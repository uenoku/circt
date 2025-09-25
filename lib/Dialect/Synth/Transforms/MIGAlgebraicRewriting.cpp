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
#include <mlir/IR/Location.h>

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

/// Enhanced associativity rewrite with depth checking and inversion handling
static Value tryAssociativityRewriteWithDepth(
    Location loc, const synth::OrderedValues &topOperands,
    const synth::OrderedValues &nestedOperands, PatternRewriter &rewriter) {

  // Attempt a specific associativity case.
  // Mapping:
  // - common: top operand (v or w) that matches one of {x, y}
  // - other: the other top operand
  // - childOther: the other shallow nested input (the one not matched by
  // 'common')
  // - remaining: z (deepest nested input)
  // - vIsCommon: true if common==v (false means common==w)
  // - xIsCommon: true if common matches x (false means matches y)
  auto tryRewrite = [&](InvertibleValue common, InvertibleValue other,
                        InvertibleValue childOther, InvertibleValue remaining,
                        bool vIsCommon, bool xIsCommon) -> Value {
    // Get the target nested operand to match against
    InvertibleValue targetNested =
        xIsCommon ? nestedOperands[0] : nestedOperands[1];

    bool complementedMatch = common.isComplementary(targetNested);
    // Check if common operand matches the target nested operand
    if (!complementedMatch && !common.isEquivalent(targetNested))
      return Value();

    // Apply complement match if needed
    if (complementedMatch) {
      childOther ^= true;
      remaining ^= true;
    }

    // Create inner majority: maj(common, other, childOther)

    auto innerMaj = rewriter.createOrFold<synth::mig::MajorityInverterOp>(
        loc, common, other, childOther);

    // Create outer majority: maj(innerMaj, remaining, common)
    return rewriter.createOrFold<synth::mig::MajorityInverterOp>(
        loc, InvertibleValue(innerMaj, false), remaining, common);
  };

  // Try different patterns using InvertibleOperand from
  // topOperands/nestedOperands
  InvertibleValue topV = topOperands[0];
  InvertibleValue topW = topOperands[1];
  InvertibleValue nestedX = nestedOperands[0];
  InvertibleValue nestedY = nestedOperands[1];
  InvertibleValue nestedZ = nestedOperands[2];

  // Inversion propagation.
  assert(!topOperands[2].isInverted() &&
         "Deepest operand should have no inversion");

  if (auto result = tryRewrite(topV, topW, nestedY, nestedZ, true, true))
    return result;
  if (auto result = tryRewrite(topV, topW, nestedX, nestedZ, true, false))
    return result;
  if (auto result = tryRewrite(topW, topV, nestedY, nestedZ, false, true))
    return result;
  if (auto result = tryRewrite(topW, topV, nestedX, nestedZ, false, false))
    return result;
  return {};
}

/// Pattern to rewrite MIG operations for depth reduction
struct MIGDepthReductionPattern
    : public OpRewritePattern<synth::mig::MajorityInverterOp> {
  bool allowAreaIncrease;
  IncrementalLongestPathAnalysis *depthAnalysis;

  unsigned numAssociativity = 0;
  unsigned numDistributivity = 0;

  MIGDepthReductionPattern(MLIRContext *context, bool allowAreaIncrease,
                           IncrementalLongestPathAnalysis *analysis)
      : OpRewritePattern<synth::mig::MajorityInverterOp>(context,
                                                         /*benefit=*/1),
        allowAreaIncrease(allowAreaIncrease), depthAnalysis(analysis) {}

  LogicalResult matchAndRewrite(synth::mig::MajorityInverterOp op,
                                PatternRewriter &rewriter) const override {
    assert(depthAnalysis->isOperationValidToMutate(op));
    // Only handle 3-input majority operations
    if (op.getNumOperands() != 3)
      return failure();
    LDBG() << "  Trying: " << op << "\n";

    // Get operands ordered by depth level (shallowest to deepest)
    auto topOperandsFailureOr = OrderedValues::get(op, depthAnalysis);
    if (failed(topOperandsFailureOr)) {
      LDBG() << "  Skipping: " << op << " (failed to get ordered operands)\n";
      return rewriter.notifyMatchFailure(op, "Failed to get ordered operands");
    }

    auto topOperands = *topOperandsFailureOr;

    // Skip if depth difference is not significant enough for optimization
    if (topOperands[2].depth <= topOperands[1].depth + 1) {
      LDBG() << "  Skipping: " << op << " (depth difference too small)\n";
      LDBG() << "    " << topOperands[0].depth << " " << topOperands[1].depth
             << " " << topOperands[2].depth << "\n";
      return failure();
    }

    // Check if the deepest operand is a 3-input majority operation
    auto nestedMajOp = topOperands[2]
                           .getValue()
                           .getDefiningOp<synth::mig::MajorityInverterOp>();
    if (!nestedMajOp || nestedMajOp.getNumOperands() != 3)
      return failure();

    // Respect single fanout constraint when area increase is not allowed
    if (!allowAreaIncrease && !nestedMajOp.getResult().hasOneUse())
      return failure();

    auto nestedOperandsFailureOr =
        OrderedValues::get(nestedMajOp, depthAnalysis);
    if (failed(nestedOperandsFailureOr))
      return rewriter.notifyMatchFailure(op, "Failed to get nested operands");
    auto &nestedOperands = *nestedOperandsFailureOr;

    // Skip if nested operation doesn't have significant depth difference
    if (nestedOperands[2].depth == nestedOperands[1].depth) {
      LDBG() << "  Skipping: " << op << " (nested depth are same)\n";
      return failure();
    }

    // Propagate inversion from top operand to nested operands.
    if (topOperands[2].isInverted()) {
      nestedOperands[0] ^= true;
      nestedOperands[1] ^= true;
      nestedOperands[2] ^= true;
      topOperands[2] ^= true;
      assert(!topOperands[2].isInverted());
    }

    // Try associativity rewrite first (preserves area)
    if (auto newOp = tryAssociativityRewriteWithDepth(
            op.getLoc(), topOperands, nestedOperands, rewriter)) {
      LDBG() << "  Applied associativity rewrite: " << op << " -> " << newOp
             << "\n";

      rewriter.replaceOp(op, newOp);
      return success();
    }

    // Try distributivity rewrite if area increase is allowed
    if (!allowAreaIncrease)
      return failure();

    LDBG() << "  Applying distributivity rewrite for: " << op << "\n";

    return applyDistributivityRewrite(op, topOperands, nestedOperands,
                                      rewriter);
  }

private:
  /// Apply distributivity rewrite: maj(v, w, maj(x, y, z)) -> maj(maj(v, w,
  /// x), maj(v, w, y), z)
  LogicalResult
  applyDistributivityRewrite(synth::mig::MajorityInverterOp op,
                             const synth::OrderedValues &topOperands,
                             const synth::OrderedValues &nestedOperands,
                             PatternRewriter &rewriter) const {
    LDBG() << "  Applying distributivity rewrite for: " << op << "\n";

    // Create left majority: maj(v, w, x)
    auto leftMaj = rewriter.createOrFold<synth::mig::MajorityInverterOp>(
        op.getLoc(), topOperands[0], topOperands[1], nestedOperands[0]);

    // Create right majority: maj(v, w, y)
    auto rightMaj = rewriter.createOrFold<synth::mig::MajorityInverterOp>(
        op.getLoc(), topOperands[0], topOperands[1], nestedOperands[1]);

    // Create final majority: maj(leftMaj, rightMaj, z)
    auto result = rewriter.createOrFold<synth::mig::MajorityInverterOp>(
        op.getLoc(), InvertibleValue(leftMaj), InvertibleValue(rightMaj),
        nestedOperands[2]);

    rewriter.replaceOp(op, result);
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
    PatternRewriter rewriter(&getContext());
    while (true) {
      bool changed = false;
      IncrementalLongestPathAnalysis depthAnalysis(module, am);
      MIGDepthReductionPattern pattern(&getContext(), allowAreaIncrease,
                                       &depthAnalysis);

      rewriter.setListener(&depthAnalysis);
      module.walk([&](mig::MajorityInverterOp maj) {
        rewriter.setInsertionPoint(maj);
        if (succeeded(pattern.matchAndRewrite(maj, rewriter)))
          changed = true;
      });
      if (!changed)
        break;
    }
  }
};

} // namespace
