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

/// Create a MIG majority operation from InvertibleOperand objects
static Value createMajorityFromInvertibleOperands(OpBuilder &rewriter,
                                                  Location loc,
                                                  InvertibleOperand a,
                                                  InvertibleOperand b,
                                                  InvertibleOperand c) {
  SmallVector<Value, 3> inputs{a.getValue(), b.getValue(), c.getValue()};
  SmallVector<bool, 3> inverts{a.isInverted(), b.isInverted(), c.isInverted()};

  return rewriter.createOrFold<synth::mig::MajorityInverterOp>(loc, inputs,
                                                               inverts);
}

/// Enhanced associativity rewrite with depth checking and inversion handling
static Value tryAssociativityRewriteWithDepth(
    Location loc, const synth::OrderedOperands &topOperands,
    const synth::OrderedOperands &nestedOperands, PatternRewriter &rewriter) {

  // Look for associativity opportunities: maj(v, w, maj(x, y, z))
  Value v = topOperands[0].getValue();
  Value w = topOperands[1].getValue();
  Value x = nestedOperands[0].getValue();
  Value y = nestedOperands[1].getValue();
  Value z = nestedOperands[2].getValue();
  // Helper to create rewrite with proper inversion handling
  auto tryRewrite = [&](Value common, Value other, Value childOther,
                        Value remaining, bool vIsCommon,
                        bool xIsCommon) -> Value {
    bool isInverted = false;
    if (!isSameSignal(common, xIsCommon ? x : y, isInverted))
      return nullptr;

    // Handle inversion flags properly
    SmallVector<bool> innerInversions = {
        topOperands.isInverted(vIsCommon ? 0 : 1),
        topOperands.isInverted(vIsCommon ? 1 : 0),
        static_cast<bool>(nestedOperands.isInverted(xIsCommon ? 1 : 0) ^
                          isInverted)};

    auto innerMaj = createMajorityFromInvertibleOperands(
        rewriter, loc, InvertibleOperand(common, innerInversions[0]),
        InvertibleOperand(other, innerInversions[1]),
        InvertibleOperand(childOther, innerInversions[2]));

    SmallVector<bool> outerInversions = {
        static_cast<bool>(nestedOperands.isInverted(2) ^ isInverted), false,
        topOperands.isInverted(vIsCommon ? 0 : 1)};

    auto newOp = createMajorityFromInvertibleOperands(
        rewriter, loc, InvertibleOperand(remaining, outerInversions[0]),
        InvertibleOperand(innerMaj, outerInversions[1]),
        InvertibleOperand(common, outerInversions[2]));

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
    // Only handle 3-input majority operations
    if (op.getNumOperands() != 3)
      return failure();

    // Get operands ordered by depth level (shallowest to deepest)
    auto topOperandsFailureOr = OrderedOperands::get(op, depthAnalysis);
    if (failed(topOperandsFailureOr))
      return rewriter.notifyMatchFailure(op, "Failed to get ordered operands");

    auto topOperands = *topOperandsFailureOr;

    // Skip if depth difference is not significant enough for optimization
    if (topOperands[2].depth <= topOperands[1].depth + 1)
      return failure();

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
        OrderedOperands::get(nestedMajOp, depthAnalysis);
    if (failed(nestedOperandsFailureOr))
      return rewriter.notifyMatchFailure(op, "Failed to get nested operands");
    auto nestedOperands = *nestedOperandsFailureOr;

    // Skip if nested operation doesn't have significant depth difference
    if (nestedOperands[2].depth == nestedOperands[1].depth)
      return failure();

    LLVM_DEBUG({
      llvm::errs() << "  Trying rewrite for: " << op << "\n";
      topOperands.dump(llvm::errs());
      nestedOperands.dump(llvm::errs());
    });

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

    return applyDistributivityRewrite(op, topOperands, nestedOperands,
                                      rewriter);
  }

private:
  /// Apply distributivity rewrite: maj(v, w, maj(x, y, z)) -> maj(maj(v, w, x),
  /// maj(v, w, y), z)
  LogicalResult
  applyDistributivityRewrite(synth::mig::MajorityInverterOp op,
                             const synth::OrderedOperands &topOperands,
                             const synth::OrderedOperands &nestedOperands,
                             PatternRewriter &rewriter) const {
    LDBG() << "  Applying distributivity rewrite for: " << op << "\n";

    // Create left majority: maj(v, w, x)
    auto leftMaj = createMajorityFromInvertibleOperands(
        rewriter, op.getLoc(), topOperands[0], topOperands[1],
        nestedOperands[0]);

    // Create right majority: maj(v, w, y)
    auto rightMaj = createMajorityFromInvertibleOperands(
        rewriter, op.getLoc(), topOperands[0], topOperands[1],
        nestedOperands[1]);

    // Create final majority: maj(leftMaj, rightMaj, z)
    auto result = createMajorityFromInvertibleOperands(
        rewriter, op.getLoc(), InvertibleOperand(leftMaj),
        InvertibleOperand(rightMaj, false), nestedOperands[2]);

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
