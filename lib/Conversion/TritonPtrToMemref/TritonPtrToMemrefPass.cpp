//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "triton-ptr-to-memref"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONPTRTOMEMREF
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h.inc"

namespace {

static Type getMemRefElementTypeForPointer(triton::PointerType ptrType) {
  Type pointeeType = ptrType.getPointeeType();
  if (auto shapedType = dyn_cast<ShapedType>(pointeeType)) {
    return shapedType.getElementType();
  }
  return pointeeType;
}

class TritonFunctionSignatureConverter : public TypeConverter {
public:
  TritonFunctionSignatureConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return MemRefType::get({}, getMemRefElementTypeForPointer(ptrType));
    });
    addConversion([](RankedTensorType tensorType) -> std::optional<Type> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return MemRefType::get(tensorType.getShape(), ptrType.getPointeeType());
      }
      return std::nullopt;
    });

    auto createUnrealizedCast = [&](OpBuilder &builder, Type resultType,
                                    ValueRange inputs, Location loc) -> Value {
      // Keep the existing internal unranked representation while exposing a
      // statically described rank-zero memref at the function boundary.
      if (isa<triton::PointerType>(resultType) && inputs.size() == 1) {
        if (auto memref = dyn_cast<MemRefType>(inputs[0].getType())) {
          Value unranked = builder.create<memref::CastOp>(
              loc,
              UnrankedMemRefType::get(memref.getElementType(),
                                      memref.getMemorySpace()),
              inputs[0]);
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultType, unranked)
              .getResult(0);
        }
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(createUnrealizedCast);
  }
};

// Lower each tt.alloc to the allocation its placement names, a heap
// memref.alloc for `global` or a stack memref.alloca for `local`, and hand it
// to its users
// the way a pointer argument is handed to them: as a memref<*xT> cast back to
// the original !tt.ptr<T>. The loads and stores already lowered by
// StructuredToMemref and UnstructuredToMemref cast their base pointer to
// memref<*xT>, so the cast pairs fold away in reconcile-unrealized-casts and
// the accesses address the allocation directly.
//
// No dealloc is emitted here. Every compiler pipeline frees heap memrefs with
// the ownership-based --buffer-deallocation-pipeline, which rejects input that
// already contains deallocations; it frees the buffer at the end of the
// function, which is the end of the program instance.
//
// `spm` (scratchpad memory, such as TCM) has no lowering yet and is reported
// as an error rather than placed elsewhere.
static LogicalResult lowerAlloc(ModuleOp moduleOp) {
  WalkResult result = moduleOp.walk([](triton::AllocOp op) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto ptrType = cast<triton::PointerType>(op.getType());
    Type elementType = ptrType.getPointeeType();
    auto memrefType = MemRefType::get(
        {static_cast<int64_t>(op.getNumElements())}, elementType);
    IntegerAttr alignment = b.getI64IntegerAttr(op.getAlignment());
    Value alloc;
    switch (op.getPlacement()) {
    case triton::AllocPlacement::GLOBAL:
      alloc = b.create<memref::AllocOp>(loc, memrefType, alignment);
      break;
    case triton::AllocPlacement::LOCAL:
      // The op sits directly in the function body, so the stack slot lives
      // as long as the program instance.
      alloc = b.create<memref::AllocaOp>(loc, memrefType, alignment);
      break;
    case triton::AllocPlacement::SPM:
      op.emitError("tt.alloc placement 'spm' is not implemented yet");
      return WalkResult::interrupt();
    }
    Value unranked = b.create<memref::CastOp>(
        loc, UnrankedMemRefType::get(elementType, /*memorySpace=*/0), alloc);
    Value ptr = b.create<UnrealizedConversionCastOp>(loc, ptrType, unranked)
                    .getResult(0);
    op.replaceAllUsesWith(ptr);
    op.erase();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

class TritonPtrToMemrefPass
    : public ::impl::TritonPtrToMemrefBase<TritonPtrToMemrefPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
                triton::TritonDialect, tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonFunctionSignatureConverter typeConverter;

    // FuncToLLVM reconstructs the rank-zero descriptor from one data pointer.
    // Limit this calling convention to functions whose pointer ABI we convert.
    moduleOp.walk([&](FunctionOpInterface function) {
      auto type = dyn_cast<FunctionType>(function.getFunctionType());
      if (type &&
          (llvm::any_of(type.getInputs(),
                        [](Type t) { return isa<triton::PointerType>(t); }) ||
           llvm::any_of(type.getResults(),
                        [](Type t) { return isa<triton::PointerType>(t); })))
        function->setAttr("llvm.bareptr", UnitAttr::get(&getContext()));
    });

    // Update function signature and call ops to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp, triton::FuncOp>([&](auto op) {
      return typeConverter.isSignatureLegal(
          cast<FunctionType>(cast<FunctionOpInterface>(op).getFunctionType()));
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getResultTypes()) &&
             typeConverter.isLegal(op.getOperandTypes());
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    if (failed(lowerAlloc(moduleOp)))
      return signalPassFailure();

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonPtrToMemrefPass() {
  return std::make_unique<TritonPtrToMemrefPass>();
}
