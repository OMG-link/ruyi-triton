// Licensed under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0

//===- DecomposeHighLevelOpsPass.cpp - Decompose tt.dot into basic ops ----===//
//
// This pass decomposes high-level Triton operations (primarily tt.dot) into
// sequences of basic operations (trans, expand_dims, broadcast, mulf/muli,
// reduce) to enable fine-grained pattern matching in downstream passes.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Transform/DecomposeHighLevelOps/DecomposeHighLevelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "decompose-high-level-ops"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Transform/DecomposeHighLevelOps/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type query helpers
//===----------------------------------------------------------------------===//

/// Enum representing the kind of type conversion needed for DotOp operands.
enum class ConversionKind {
  None,   // No conversion needed (input type == output type)
  ExtF,   // Float-to-float widening (arith.extf)
  ExtSI,  // Signed integer widening (arith.extsi)
  SIToFP, // Signed integer to floating-point (arith.sitofp)
  Unsupported
};

/// Determine the type conversion needed to cast from inputElemType to
/// outputElemType. Returns Unsupported for invalid combinations (e.g.,
/// narrowing conversions or float-to-integer).
///
/// Supported widening conversions:
///   - Float → wider Float (e.g., f16→f32): ExtF
///   - Integer → wider Integer (e.g., i8→i32): ExtSI
///   - Integer → Float (e.g., i8→f32): SIToFP
static ConversionKind determineConversion(Type inputElemType,
                                          Type outputElemType) {
  if (inputElemType == outputElemType)
    return ConversionKind::None;

  if (isa<FloatType>(inputElemType) && isa<FloatType>(outputElemType)) {
    // Guard against narrowing: output must be at least as wide as input.
    if (cast<FloatType>(outputElemType).getWidth() >=
        cast<FloatType>(inputElemType).getWidth())
      return ConversionKind::ExtF;
    return ConversionKind::Unsupported;
  }

  if (inputElemType.isInteger() && outputElemType.isInteger()) {
    // Guard against narrowing: output must be at least as wide as input.
    if (outputElemType.getIntOrFloatBitWidth() >=
        inputElemType.getIntOrFloatBitWidth())
      return ConversionKind::ExtSI;
    return ConversionKind::Unsupported;
  }

  if (inputElemType.isInteger() && isa<FloatType>(outputElemType))
    return ConversionKind::SIToFP;

  return ConversionKind::Unsupported;
}

/// Returns true if the element type is floating-point, meaning
/// arith.mulf/arith.addf should be used. Returns false for integer types,
/// meaning arith.muli/arith.addi should be used.
static bool isFloatArith(Type elemType) { return isa<FloatType>(elemType); }

/// Pattern to decompose tt.dot into basic ops:
/// trans, expand_dims, broadcast, mulf/muli, reduce.
struct DotOpDecomposer : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Extract operands.
    Value a = op.getA();
    Value b = op.getB();
    Value acc = op.getC();

    // Extract types and shapes.
    auto aType = cast<RankedTensorType>(a.getType());
    auto bType = cast<RankedTensorType>(b.getType());
    auto resultType = cast<RankedTensorType>(op.getType());

    // Only handle rank-2 (non-batched) dot operations.
    if (aType.getRank() != 2 || bType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "only rank-2 dot is supported");

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    Type inputElemType = aType.getElementType();
    Type outputElemType = resultType.getElementType();

    // Determine type conversion; bail out on unsupported combinations.
    ConversionKind convKind =
        determineConversion(inputElemType, outputElemType);
    if (convKind == ConversionKind::Unsupported)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type conversion for decomposition");

    // Step 1: Transpose b from (K, N) to (N, K).
    SmallVector<int32_t> order = {1, 0};
    auto transB = rewriter.create<triton::TransOp>(loc, b, order);
    Value bT = transB.getResult();

    // Step 2: Type conversion if needed.
    Value aCast = a;
    Value bCast = bT;

    if (convKind != ConversionKind::None) {
      auto aCastType = RankedTensorType::get(aType.getShape(), outputElemType);
      auto bCastType = RankedTensorType::get(
          cast<RankedTensorType>(bT.getType()).getShape(), outputElemType);

      switch (convKind) {
      case ConversionKind::ExtF:
        aCast = rewriter.create<arith::ExtFOp>(loc, aCastType, a);
        bCast = rewriter.create<arith::ExtFOp>(loc, bCastType, bT);
        break;
      case ConversionKind::ExtSI:
        aCast = rewriter.create<arith::ExtSIOp>(loc, aCastType, a);
        bCast = rewriter.create<arith::ExtSIOp>(loc, bCastType, bT);
        break;
      case ConversionKind::SIToFP:
        aCast = rewriter.create<arith::SIToFPOp>(loc, aCastType, a);
        bCast = rewriter.create<arith::SIToFPOp>(loc, bCastType, bT);
        break;
      case ConversionKind::None:
      case ConversionKind::Unsupported:
        llvm_unreachable("handled before the switch");
      }
    }

    // Step 3: Expand dims.
    // a: (M, K) -> (M, 1, K) via expand_dims at axis=1
    auto aExp = rewriter.create<triton::ExpandDimsOp>(loc, aCast, 1);
    // b_t: (N, K) -> (1, N, K) via expand_dims at axis=0
    auto bExp = rewriter.create<triton::ExpandDimsOp>(loc, bCast, 0);

    // Step 4: Broadcast to (M, N, K).
    auto broadcastType = RankedTensorType::get({M, N, K}, outputElemType);
    auto aBc = rewriter.create<triton::BroadcastOp>(loc, broadcastType, aExp);
    auto bBc = rewriter.create<triton::BroadcastOp>(loc, broadcastType, bExp);

    // Step 5: Elementwise multiplication.
    Value prod;
    if (isFloatArith(outputElemType)) {
      prod = rewriter.create<arith::MulFOp>(loc, aBc, bBc);
    } else {
      prod = rewriter.create<arith::MulIOp>(loc, aBc, bBc);
    }

    // Step 6: Reduce along axis=2 with addition combiner.
    auto reduceOp = rewriter.create<triton::ReduceOp>(
        loc, SmallVector<Value>{prod}, /*axis=*/2);
    {
      // Build the combiner region.
      Region &region = reduceOp.getRegion();
      Block &block = region.emplaceBlock();
      block.addArguments({outputElemType, outputElemType}, {loc, loc});
      Value arg0 = block.getArgument(0);
      Value arg1 = block.getArgument(1);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);

      Value addResult;
      if (isFloatArith(outputElemType)) {
        addResult = rewriter.create<arith::AddFOp>(loc, arg0, arg1);
      } else {
        addResult = rewriter.create<arith::AddIOp>(loc, arg0, arg1);
      }
      rewriter.create<triton::ReduceReturnOp>(loc,
                                              SmallVector<Value>{addResult});
    }

    Value result = reduceOp.getResult()[0];

    // Step 7: Handle accumulator.
    if (!isZeroAccumulator(acc)) {
      if (isFloatArith(outputElemType)) {
        result = rewriter.create<arith::AddFOp>(loc, result, acc);
      } else {
        result = rewriter.create<arith::AddIOp>(loc, result, acc);
      }
    }

    // Replace the original DotOp.
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  /// Detect whether the accumulator value is a zero tensor.
  /// Returns true if:
  ///   1. acc is defined by tt.splat(arith.constant 0), or
  ///   2. acc is defined by arith.constant with a zero dense attribute.
  bool isZeroAccumulator(Value acc) const {
    // Pattern 1: tt.splat(arith.constant 0)
    if (auto splatOp = acc.getDefiningOp<triton::SplatOp>()) {
      Value src = splatOp.getSrc();
      if (matchPattern(src, m_Zero()) || matchPattern(src, m_AnyZeroFloat()))
        return true;
    }

    // Pattern 2: arith.constant dense<0.0> (or dense<0>)
    if (auto constOp = acc.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat()) {
          Type eltType = denseAttr.getElementType();
          if (isa<FloatType>(eltType)) {
            return denseAttr.getSplatValue<APFloat>().isZero();
          }
          if (eltType.isInteger()) {
            return denseAttr.getSplatValue<APInt>().isZero();
          }
        }
      }
    }

    return false;
  }
};

class DecomposeHighLevelOpsPass
    : public DecomposeHighLevelOpsBase<DecomposeHighLevelOpsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<DotOpDecomposer>(context);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createDecomposeHighLevelOpsPass() {
  return std::make_unique<DecomposeHighLevelOpsPass>();
}
