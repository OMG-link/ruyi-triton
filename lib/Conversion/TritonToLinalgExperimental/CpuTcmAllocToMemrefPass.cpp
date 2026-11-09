//===----------------------------------------------------------------------===//
//
// --cpu-tcm-alloc-to-memref
//
// triton.language.extra.cpu.alloc(shape, dtype) is emitted as a call to the
// runtime symbol __triton_cpu_tcm_alloc whose result is only used as a pointer.
// After the rest of the experimental pipeline these pointers are tptr ops:
//
//   %a = func.call @__triton_cpu_tcm_alloc(%bytes, %align) : (i64, i64) -> i64
//   %p = tptr.inttoptr %a : i64 to <#tptr.default_memory_space>
//   %q = tptr.ptradd %p %off : ..., i32 to ...         (optional, bytes)
//   %m = tptr.to_memref %q : ... to memref<1xf32>
//
// which nothing downstream lowers.  This pass replaces the call by a buffer
//
//   %buf = memref.alloc() {alignment = A, cpu.tcm} : memref<BYTESxi8>
//
// and every tptr.to_memref reached from it by memref.view %buf[byte offset][],
// every tptr.ptrtoint by the address of the buffer plus the offset.  The
// `cpu.tcm` marker (a plain attribute, not a memory space, so that views and
// the vector lowering are unaffected) is consumed by buddy-mlir's
// --cpu-tcm-assign, which places the buffer in the TCM of the thread executing
// the allocation.  Without that pass the buffer is an ordinary heap
// allocation.
//
// Element sizes (tptr.type_offset) in the offsets of these pointers are
// rebuilt as constants, since buddy-mlir does not know the tptr dialect
// either.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToLinalgExperimental/CpuTcmAllocToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kAllocSymbol = "__triton_cpu_tcm_alloc";
constexpr llvm::StringLiteral kTcmAttr = "cpu.tcm";

Value toIndex(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex())
    return v;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

// Byte size of a scalar type, 0 if unknown.
int64_t scalarBytes(Type type) {
  if (type.isIntOrFloat())
    return std::max<int64_t>(1, (type.getIntOrFloatBitWidth() + 7) / 8);
  return 0;
}

// Size of tptr.type_offset `v` as an index constant, null if `v` is not one.
Value typeOffsetSize(OpBuilder &b, Location loc, Value v) {
  auto typeOffset = v.getDefiningOp<tptr::TypeOffsetOp>();
  if (!typeOffset)
    return Value();
  const int64_t bytes = scalarBytes(typeOffset.getBaseType());
  if (bytes == 0)
    return Value();
  return b.create<arith::ConstantIndexOp>(loc, bytes);
}

// Offset of a tptr.ptradd as an index.  tt.addptr is lowered to
// ptradd(p, offset * type_offset(T)); the element size is rebuilt as a
// constant here (buddy-mlir does not know tptr.type_offset).  The original
// computation is left to dead code elimination: it may be shared with
// pointers that are not cpu.alloc buffers.
Value byteOffset(OpBuilder &b, Location loc, Value offset) {
  if (Value size = typeOffsetSize(b, loc, offset))
    return size;
  if (auto mul = offset.getDefiningOp<arith::MulIOp>()) {
    for (int i = 0; i < 2; ++i) {
      if (Value size = typeOffsetSize(b, loc, mul->getOperand(i))) {
        Value count = toIndex(b, loc, mul->getOperand(1 - i));
        return b.create<arith::MulIOp>(loc, count, size);
      }
    }
  }
  return toIndex(b, loc, offset);
}

LogicalResult unsupportedUse(Operation *user, func::CallOp call) {
  InFlightDiagnostic diag =
      user->emitError("cpu.alloc: unsupported use of the returned pointer "
                      "(only loads, stores and pointer arithmetic are "
                      "supported): ")
      << user->getName();
  diag.attachNote(call.getLoc()) << "buffer allocated here";
  return failure();
}

LogicalResult rewriteAlloc(func::CallOp call) {
  Location loc = call.getLoc();
  IntegerAttr bytesAttr, alignAttr;
  if (call.getNumOperands() != 2 || call.getNumResults() != 1 ||
      !matchPattern(call.getOperand(0), m_Constant(&bytesAttr)) ||
      !matchPattern(call.getOperand(1), m_Constant(&alignAttr)))
    return call.emitError(
        "cpu.alloc: size and alignment must be compile-time constants");
  const int64_t bytes = bytesAttr.getInt();
  const int64_t align = alignAttr.getInt();
  if (bytes <= 0)
    return call.emitError("cpu.alloc: size must be positive, got ") << bytes;
  if (align <= 0 || !llvm::isPowerOf2_64(static_cast<uint64_t>(align)))
    return call.emitError("cpu.alloc: alignment must be a power of two, got ")
           << align;

  OpBuilder b(call);
  auto bufferType = MemRefType::get({bytes}, b.getI8Type());
  auto buffer = b.create<memref::AllocOp>(loc, bufferType,
                                          b.getI64IntegerAttr(align));
  buffer->setAttr(kTcmAttr, b.getUnitAttr());

  // Pointers derived from the allocation with their byte offset (null: 0).
  SmallVector<std::pair<Value, Value>> work;
  SmallVector<Operation *> dead;
  for (Operation *user : llvm::to_vector(call.getResult(0).getUsers())) {
    auto intToPtr = dyn_cast<tptr::IntToPtrOp>(user);
    if (!intToPtr)
      return unsupportedUse(user, call);
    work.push_back({intToPtr.getResult(), Value()});
    dead.push_back(intToPtr);
  }

  while (!work.empty()) {
    auto [ptr, offset] = work.pop_back_val();
    for (Operation *user : llvm::to_vector(ptr.getUsers())) {
      OpBuilder ub(user);
      Location userLoc = user->getLoc();
      if (auto add = dyn_cast<tptr::PtrAddOp>(user)) {
        Value delta = byteOffset(ub, userLoc, add.getOffset());
        Value sum = offset ? ub.create<arith::AddIOp>(userLoc, offset, delta)
                                 .getResult()
                           : delta;
        work.push_back({add.getResult(), sum});
        dead.push_back(add);
      } else if (auto toMemref = dyn_cast<tptr::ToMemrefOp>(user)) {
        auto type = cast<MemRefType>(toMemref.getType());
        if (!type.getLayout().isIdentity() || !type.hasStaticShape())
          return unsupportedUse(user, call);
        Value shift =
            offset ? offset
                   : ub.create<arith::ConstantIndexOp>(userLoc, 0).getResult();
        Value view = ub.create<memref::ViewOp>(userLoc, type, buffer, shift,
                                               ValueRange{});
        toMemref.replaceAllUsesWith(view);
        toMemref.erase();
      } else if (auto ptrToInt = dyn_cast<tptr::PtrToIntOp>(user)) {
        Value address =
            ub.create<memref::ExtractAlignedPointerAsIndexOp>(userLoc, buffer);
        if (offset)
          address = ub.create<arith::AddIOp>(userLoc, address, offset);
        Type type = ptrToInt.getType();
        if (!type.isIndex())
          address = ub.create<arith::IndexCastOp>(userLoc, type, address);
        ptrToInt.replaceAllUsesWith(address);
        ptrToInt.erase();
      } else {
        return unsupportedUse(user, call);
      }
    }
  }

  for (Operation *op : llvm::reverse(dead)) {
    if (!op->use_empty())
      return op->emitError("cpu.alloc: pointer still in use after lowering");
    op->erase();
  }
  call.erase();
  return success();
}

class CpuTcmAllocToMemrefPass
    : public CpuTcmAllocToMemrefBase<CpuTcmAllocToMemrefPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    tptr::TPtrDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() == kAllocSymbol)
        calls.push_back(call);
    });
    for (func::CallOp call : calls)
      if (failed(rewriteAlloc(call)))
        return signalPassFailure();
    if (auto decl = module.lookupSymbol<func::FuncOp>(kAllocSymbol))
      if (decl.symbolKnownUseEmpty(module))
        decl.erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createCpuTcmAllocToMemrefPass() {
  return std::make_unique<CpuTcmAllocToMemrefPass>();
}
