//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs algebraic rewriting on Majority-Inverter Graphs (MIGs)
// for depth optimization using associativity and distributivity rules in
// majority-of-3 logic.
//
// The implementation (especially heuristics for pattern matching and algorithm
// structure) is heavily inspired by the MIG algebraic rewriting algorithm from
// the mockturtle library (https://github.com/lsils/mockturtle) and the paper
// "Majority-Inverter Graph: A New Paradigm for Logic Optimization" by Amarù et
// al. (2016).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "synth-mig-algebraic-rewriting"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_MIGALGEBRAICREWRITING
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace {

/// Check if two values represent the same signal (considering inversions)
/// Returns true if they are the same signal, false otherwise
/// Also sets isInverted to true if one is the inversion of the other
bool isSameSignal(Value a, Value b, bool &isInverted) {
  isInverted = false;

  // Direct equality
  if (a == b)
    return true;

  if (auto constA = a.getDefiningOp<hw::ConstantOp>()) {
    if (auto constB = b.getDefiningOp<hw::ConstantOp>()) {
      if (constA.getValue() == constB.getValue())
        return true;
      if (constA.getValue() == ~constB.getValue()) {
        isInverted = true;
        return true;
      }
    }
  }

  // Check if one is an inverted version of the other through MIG operations
  auto checkInvertedMIG = [](Value val, Value target) -> bool {
    if (auto migOp = val.getDefiningOp<synth::mig::MajorityInverterOp>()) {
      // Check for single-input inverted MIG (acts as NOT gate)
      if (migOp.getNumOperands() == 1 && migOp.isInverted(0) &&
          migOp.getOperand(0) == target)
        return true;
    }
    return false;
  };

  if (checkInvertedMIG(a, b) || checkInvertedMIG(b, a)) {
    isInverted = true;
    return true;
  }

  return false;
}

static Value createMajorityFunction(OpBuilder &rewriter, Location loc, Value a,
                                    Value b, Value c, bool invertA,
                                    bool invertB, bool invertC) {
  std::array<Value, 3> inputs = {a, b, c};
  std::array<bool, 3> inverts = {invertA, invertB, invertC};
  return rewriter.createOrFold<synth::mig::MajorityInverterOp>(loc, inputs,
                                                               inverts);
}

/// Enhanced associativity rewrite with depth checking and inversion handling
static Value
tryAssociativityRewriteWithDepth(synth::mig::MajorityInverterOp topOp,
                                 const synth::OrderedOperands &topChildren,
                                 const synth::OrderedOperands &childChildren,
                                 PatternRewriter &rewriter) {

  // Look for associativity opportunities: maj(v, w, maj(x, y, z))
  Value v = topChildren[0].getValue();
  Value w = topChildren[1].getValue();
  Value x = childChildren[0].getValue();
  Value y = childChildren[1].getValue();
  Value z = childChildren[2].getValue();
  // Helper to create rewrite with proper inversion handling
  auto tryRewrite = [&](Value common, Value other, Value childOther,
                        Value remaining, bool vIsCommon,
                        bool xIsCommon) -> Value {
    bool isInverted = false;
    if (!isSameSignal(common, xIsCommon ? x : y, isInverted))
      return nullptr;

    // Handle inversion flags properly
    SmallVector<bool> innerInversions = {
        topChildren.isInverted(vIsCommon ? 0 : 1),
        topChildren.isInverted(vIsCommon ? 1 : 0),
        static_cast<bool>(childChildren.isInverted(xIsCommon ? 1 : 0) ^
                          isInverted)};

    auto innerMaj = createMajorityFunction(
        rewriter, topOp.getLoc(), common, other, childOther, innerInversions[0],
        innerInversions[1], innerInversions[2]);

    SmallVector<bool> outerInversions = {
        static_cast<bool>(childChildren.isInverted(2) ^ isInverted), false,
        topOp.isInverted(vIsCommon ? 0 : 1)};

    auto newOp = createMajorityFunction(rewriter, topOp.getLoc(), remaining,
                                        innerMaj, common, outerInversions[0],
                                        outerInversions[1], outerInversions[2]);

    return newOp;
  };

  // Try different patterns
  if (auto result = tryRewrite(v, w, y, z, true, true))
    return result;
  if (auto result = tryRewrite(v, w, x, z, true, false))
    return result;
  if (auto result = tryRewrite(w, v, y, z, false, true))
    return result;
  if (auto result = tryRewrite(w, v, x, z, false, false))
    return result;
  return Value();
}

/// Pattern to rewrite MIG operations for depth reduction
struct MIGDepthReductionPattern
    : public OpRewritePattern<synth::mig::MajorityInverterOp> {
  bool allowAreaIncrease;
  IncrementalLongestPathAnalysis *depthAnalysis;

  MIGDepthReductionPattern(MLIRContext *context, bool allowAreaIncrease,
                           IncrementalLongestPathAnalysis *analysis)
      : OpRewritePattern<synth::mig::MajorityInverterOp>(context,
                                                         /*benefit=*/1),
        allowAreaIncrease(allowAreaIncrease), depthAnalysis(analysis) {}

  LogicalResult matchAndRewrite(synth::mig::MajorityInverterOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle 3-input majority operations for now
    if (op.getNumOperands() != 3)
      return failure();

    // Get current depth of the operation
    // Get ordered children (sorted by level) with depth analysis
    auto orderedChildrenFailureOr = OrderedOperands::get(op, depthAnalysis);
    if (failed(orderedChildrenFailureOr))
      return rewriter.notifyMatchFailure(op, "Failed to get ordered children");

    auto orderedChildren = *orderedChildrenFailureOr;

    if (orderedChildren[2].depth <= orderedChildren[1].depth + 1)
      return failure();

    // Check if the last child is a majority operation with higher level
    auto childOp = orderedChildren[2]
                       .getValue()
                       .getDefiningOp<synth::mig::MajorityInverterOp>();
    if (!childOp || childOp.getNumOperands() != 3)
      return failure();

    // For single fanout constraint (when area increase is not allowed)
    if (!allowAreaIncrease && !childOp.getResult().hasOneUse())
      return failure();

    auto childChildrenFailureOr = OrderedOperands::get(childOp, depthAnalysis);
    if (failed(childChildrenFailureOr))
      return rewriter.notifyMatchFailure(op, "Failed to get child depth");
    auto childChildren = *childChildrenFailureOr;

    if (childChildren[2].depth == childChildren[1].depth)
      return failure();

    LLVM_DEBUG({
      llvm::errs() << "  Trying: " << op << "\n";
      orderedChildren.dump(llvm::errs());
      childChildren.dump(llvm::errs());
    });

    if (auto newOp = tryAssociativityRewriteWithDepth(
            op, orderedChildren, *childChildrenFailureOr, rewriter)) {
      // Mark the new operation to prevent infinite loops
      LDBG() << "  Found associativity rewrite: " << op << " -> " << newOp
             << "\n";
      rewriter.replaceOp(op, newOp);
      return success();
    }

    // Try distributivity rewrite if area increase is allowed
    if (!allowAreaIncrease)
      return failure();

    LDBG() << "  Found distributivity rewrite: " << op << "\n";
    // maj(v, w, maj(x, y, z)) -> maj(maj(v, w, x), maj(v, w, y), z)
    // Handle inversion flags properly
    SmallVector<bool> leftInversions = {orderedChildren.isInverted(0),
                                        orderedChildren.isInverted(1),
                                        childChildren.isInverted(0)};
    SmallVector<bool> rightInversions = {orderedChildren.isInverted(0),
                                         orderedChildren.isInverted(1),
                                         childChildren.isInverted(1)};

    auto leftMaj = createMajorityFunction(
        rewriter, op.getLoc(), orderedChildren.getValue(0),
        orderedChildren.getValue(1), childChildren.getValue(0),
        leftInversions[0], leftInversions[1], leftInversions[2]);

    auto rightMaj = createMajorityFunction(
        rewriter, op.getLoc(), orderedChildren.getValue(0),
        orderedChildren.getValue(1), childChildren.getValue(1),
        rightInversions[0], rightInversions[1], rightInversions[2]);

    rewriter.replaceOp(
        op, createMajorityFunction(rewriter, op.getLoc(), leftMaj, rightMaj,
                                   childChildren.getValue(2), false, false,
                                   childChildren.isInverted(2)));
    return success();
  }
};

/// The main MIG algebraic rewriting pass
struct MIGAlgebraicRewritingPass
    : public circt::synth::impl::MIGAlgebraicRewritingBase<
          MIGAlgebraicRewritingPass> {

  using circt::synth::impl::MIGAlgebraicRewritingBase<
      MIGAlgebraicRewritingPass>::MIGAlgebraicRewritingBase;

  void runOnOperation() override {
    auto module = getOperation();
    mlir::AnalysisManager am = getAnalysisManager();

    for (size_t i = 0; i < 30; ++i) {
      // Create longest path  for depth calculation
      IncrementalLongestPathAnalysis depthAnalysis(module, am);

      RewritePatternSet patterns(&getContext());

      patterns.add<MIGDepthReductionPattern>(&getContext(), allowAreaIncrease,
                                             &depthAnalysis);

      auto config = GreedyRewriteConfig();
      config.setListener(&depthAnalysis)
          .setUseTopDownTraversal(true)
          .setMaxIterations(1);

      // Apply patterns greedily
      if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
        continue;
    }
  }
};

} // namespace
