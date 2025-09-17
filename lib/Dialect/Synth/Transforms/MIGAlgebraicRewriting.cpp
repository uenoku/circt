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

/// Convenience wrappers to make call sites more readable.
static inline InvertibleOperand inv(Value v, bool inverted = false) {
  return InvertibleOperand(v, inverted);
}
static inline Value createMaj(OpBuilder &rewriter, Location loc, Value a,
                              bool ai, Value b, bool bi, Value c, bool ci) {
  return createMajorityFromInvertibleOperands(rewriter, loc, inv(a, ai),
                                              inv(b, bi), inv(c, ci));
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
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      auto v = topChildren[i];
      auto x = (*childChildren)[j];
      bool invert = false;
      if (!isSameSignal(v.getPointer(), x.getPointer(), invert))
        continue;
      invert ^= v.getInt() ^ x.getInt();
      LDBG() << "  Found common signal: " << v.getPointer()
             << " (invert: " << invert << ")\n";

      auto w = topChildren[1 - i];
      // Ok try swapping w and y or z
      auto y = (*childChildren)[2 - j];
      auto z = (*childChildren)[2];

      if (invert) {
        // Completely associativity.
        // M(v, w, M(v', y, z)) -> M(v, w, M(w, y, z))
        // Benefits when w == y
        bool isInverted;
        if (isSameSignal(w.getPointer(), y.getPointer(), isInverted)) {
          isInverted ^= w.getInt() ^ y.getInt();
          LDBG() << "  Found associativity rewrite: " << topOp << "\n";
          if (isInverted) {
            // M(v, w, M(w, w', z)) -> M(v, w, z)
            return createMajorityFunction(
                rewriter, topOp.getLoc(), v.getPointer(), w.getPointer(),
                z.getPointer(), v.getInt(), w.getInt(), z.getInt());
          }
          // M(v, w, M(w, w, z)) -> M(v, w, w)  -> w
          if (w.getInt())
            return createMajorityFunction(rewriter, topOp.getLoc(),
                                          w.getPointer(), true);
          return w.getPointer();
        }
      } else {
        // Assoc
        // M(v, w, M(v, y, z)) = M(z, w, M(v, y, w))
        int64_t depthZ = childChildren->depths[2];
        int64_t depthY = childChildren->depths[1];
        if (depthZ != depthY) {
          // M(v, y, w)
          auto newOp = createMajorityFunction(
              rewriter, topOp.getLoc(), v.getPointer(), y.getPointer(),
              w.getPointer(), v.getInt(), y.getInt(), w.getInt());
          // M(v, w, M(v, y, w))
          auto finalOp = createMajorityFunction(
              rewriter, topOp.getLoc(), z.getPointer(), w.getPointer(), newOp,
              z.getInt(), w.getInt(), false);
          return finalOp;
        }
      }
    }
  // Attempt a specific associativity case.
  // Mapping:
  // - common: top operand (v or w) that matches one of {x, y}
  // - other: the other top operand
  // - childOther: the other shallow nested input (the one not matched by
  // 'common')
  // - remaining: z (deepest nested input)
  // - vIsCommon: true if common==v (false means common==w)
  // - xIsCommon: true if common matches x (false means matches y)
  auto tryRewrite = [&](Value common, Value other, Value childOther,
                        Value remaining, bool vIsCommon,
                        bool xIsCommon) -> Value {
    bool complementedMatch = false;
    if (!isSameSignal(common, xIsCommon ? x : y, complementedMatch))
      return Value();

    // Inner majority: maj(common, other, childOther) with proper inversions.
    SmallVector<bool, 3> innerInv = {
        topOperands.isInverted(vIsCommon ? 0 : 1), // inversion for 'common'
        topOperands.isInverted(vIsCommon ? 1 : 0), // inversion for 'other'
        static_cast<bool>(nestedOperands.isInverted(xIsCommon ? 1 : 0) ^
                          complementedMatch)};

    auto innerMaj = createMaj(rewriter, loc, common, innerInv[0], other,
                              innerInv[1], childOther, innerInv[2]);

    // Outer majority: maj(remaining=z, innerMaj, common).
    const bool zInv =
        static_cast<bool>(nestedOperands.isInverted(2) ^ complementedMatch);
    const bool innerMajInv = false;
    const bool commonInv = topOperands.isInverted(vIsCommon ? 0 : 1);

    return createMaj(rewriter, loc, remaining, zInv, innerMaj, innerMajInv,
                     common, commonInv);
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
  return {};
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
    const auto &nestedOperands = *nestedOperandsFailureOr;
    LLVM_DEBUG({
      llvm::errs() << "  Trying rewrite for: " << op << "\n";
      topOperands.dump(llvm::errs());
      nestedOperands.dump(llvm::errs());
    });

    // Skip if nested operation doesn't have significant depth difference
    if (nestedOperands[2].depth == nestedOperands[1].depth)
      return failure();

    LLVM_DEBUG({
      llvm::errs() << "  Ok for: " << op << "\n";
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
    // if (!allowAreaIncrease)
    //   return failure();

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
        InvertibleOperand(rightMaj), nestedOperands[2]);

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

    for (size_t i = 0; i < 100; ++i) {
      // Create longest path  for depth calculation
      llvm::errs() << "Round " << i << "\n";
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
