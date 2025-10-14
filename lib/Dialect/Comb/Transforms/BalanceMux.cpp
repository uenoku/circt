//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BalanceMux pass, which balances and optimizes mux
// chains.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace circt;
using namespace comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_BALANCEMUX
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {

/// Helper class for building balanced priority mux trees with optimal delay
/// minimization using dynamic programming.
class BalancedPriorityMuxBuilder {
public:
  /// Build a balanced mux tree from conditions and results
  static Value build(PatternRewriter &rewriter,
                     synth::IncrementalLongestPathAnalysis *analysis,
                     ArrayRef<Value> conditions, ArrayRef<Value> results,
                     ArrayRef<Location> locs);

private:
  BalancedPriorityMuxBuilder(PatternRewriter &rewriter,
                             synth::IncrementalLongestPathAnalysis *analysis,
                             ArrayRef<Value> allConditions,
                             ArrayRef<Value> allResults,
                             ArrayRef<Location> allLocs)
      : rewriter(rewriter), analysis(analysis), allConditions(allConditions),
        allResults(allResults), allLocs(allLocs) {}
  PatternRewriter &rewriter;
  synth::IncrementalLongestPathAnalysis *analysis;

  // Assume each mux adds 2 units delay (AIG/MIG both require 2 levels)
  constexpr static unsigned muxDelay = 2;
  constexpr static unsigned orDelay = 1;

  struct PriorityMuxInfo {
    int64_t delay;
    int64_t condDelay;
    size_t splitPoint;
  };

  // Memoization cache for DP computation
  DenseMap<std::pair<size_t, size_t>, PriorityMuxInfo> memo;

  // Input data
  ArrayRef<Value> allConditions;
  ArrayRef<Value> allResults;
  ArrayRef<Location> allLocs;

  // Precomputed arrival times
  SmallVector<int64_t> conditionsArrivalTimes;
  SmallVector<int64_t> resultsArrivalTimes;

  bool useSimpleSplit() const { return !analysis; }

  /// Recursively build balanced mux tree for range [start, end)
  Value buildBalancedPriorityMux(size_t start, size_t end);

  /// Compute optimal delay and split point using dynamic programming
  PriorityMuxInfo computeOptimalDelayAndSplit(size_t start, size_t end);

  /// Precompute all arrival times for conditions and results
  void precomputeArrivalTimes(ArrayRef<Value> conditions,
                              ArrayRef<Value> results);
};

/// Mux chain with comparison folding pattern.
class MuxChainWithComparison : public OpRewritePattern<MuxOp> {
  unsigned muxChainThreshold;

public:
  // Set a higher benefit than PriorityEncoderReshape to run first.
  MuxChainWithComparison(MLIRContext *context, unsigned muxChainThreshold)
      : OpRewritePattern<MuxOp>(context, /*benefit=*/2),
        muxChainThreshold(muxChainThreshold) {}
  LogicalResult matchAndRewrite(MuxOp rootMux,
                                PatternRewriter &rewriter) const override {
    auto fn = [muxChainThreshold = muxChainThreshold](size_t indexWidth,
                                                      size_t numEntries) {
      // In this pattern, we consider it beneficial to fold mux chains
      // with more than the threshold.
      if (numEntries >= muxChainThreshold)
        return MuxChainWithComparisonFoldingStyle::BalancedMuxTree;
      return MuxChainWithComparisonFoldingStyle::None;
    };
    // Try folding on both false and true sides
    return llvm::success(foldMuxChainWithComparison(rewriter, rootMux,
                                                    /*isFalseSide=*/true, fn) ||
                         foldMuxChainWithComparison(rewriter, rootMux,
                                                    /*isFalseSide=*/false, fn));
  }
};

/// Rebalances a linear chain of muxes forming a priority encoder into a
/// balanced tree structure. This reduces the depth of the mux tree from O(n)
/// to O(log n).
///
/// For a priority encoder with n conditions, this transform:
/// - Reduces depth from O(n) to O(log n) levels
/// - Muxes: Creates exactly (n-1) muxes (same as original linear chain)
/// - OR gates: Creates additional O(n log n) OR gates to combine
class PriorityMuxReshape : public OpRewritePattern<MuxOp> {
  unsigned muxChainThreshold;
  synth::IncrementalLongestPathAnalysis *analysis;

public:
  PriorityMuxReshape(MLIRContext *context, unsigned muxChainThreshold,
                     synth::IncrementalLongestPathAnalysis *analysis)
      : OpRewritePattern<MuxOp>(context, /*benefit=*/1),
        muxChainThreshold(muxChainThreshold), analysis(analysis) {}

  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Helper function to collect a mux chain from a given side
  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Location>>
  collectChain(MuxOp op, bool isFalseSide) const;
};
}; // namespace

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BalancedPriorityMuxBuilder Implementation
//===----------------------------------------------------------------------===//

void BalancedPriorityMuxBuilder::precomputeArrivalTimes(
    ArrayRef<Value> conditions, ArrayRef<Value> results) {
  // Precompute arrival times for all results if timing analysis is available
  if (analysis) {
    resultsArrivalTimes.reserve(results.size());
    for (auto result : results) {
      auto delay = analysis->getMaxDelay(result);
      if (failed(delay)) {
        // Fall back to no timing analysis
        resultsArrivalTimes.clear();
        break;
      }
      resultsArrivalTimes.push_back(*delay);
    }
  }

  // Precompute arrival times for conditions
  conditionsArrivalTimes.reserve(conditions.size());
  for (auto cond : conditions) {
    int64_t condDelay = 0;
    if (analysis) {
      auto delay = analysis->getMaxDelay(cond);
      if (succeeded(delay))
        condDelay = *delay;
    }
    conditionsArrivalTimes.push_back(condDelay);
  }
}

// Compute optimal delay and split point for building a mux tree over range
// [start, end) with resultsArrivalTimes[end] as the default value.
// Returns PriorityMuxInfo with delay, condDelay, and splitPoint. Uses
// memoization. NOLINTNEXTLINE(misc-no-recursion)
BalancedPriorityMuxBuilder::PriorityMuxInfo
BalancedPriorityMuxBuilder::computeOptimalDelayAndSplit(size_t start,
                                                        size_t end) {
  auto key = std::make_pair(start, end);
  auto it = memo.find(key);
  if (it != memo.end())
    return it->second;

  // Base case: single element [i, i+1)
  if (end == start + 1) {
    PriorityMuxInfo result{resultsArrivalTimes[start],
                           conditionsArrivalTimes[start], start + 1};
    memo[key] = result;
    return result;
  }

  // Recursive case: try all split points k in (start, end)
  int64_t minDelay = std::numeric_limits<int64_t>::max();
  size_t bestSplit = start + 1;

  for (size_t k = start + 1; k < end; ++k) {
    auto leftInfo = computeOptimalDelayAndSplit(start, k);
    auto rightInfo = computeOptimalDelayAndSplit(k, end);
    int64_t totalDelay =
        std::max({leftInfo.delay, rightInfo.delay, leftInfo.condDelay}) +
        muxDelay;

    if (totalDelay <= minDelay) {
      minDelay = totalDelay;
      bestSplit = k;
    }
  }

  LDBG() << "DP: Computing optimal split for range [" << start << ", " << end
         << ")\n";
  LDBG() << "DP: Optimal delay = " << minDelay << " at k = " << bestSplit
         << "\n";

  // Compute condition delay for this range
  llvm::PriorityQueue<int64_t, std::vector<int64_t>, std::greater<int64_t>>
      condDelays;
  for (size_t i = start; i < end; ++i)
    condDelays.push(conditionsArrivalTimes[i]);
  while (condDelays.size() >= 2) {
    int64_t d1 = condDelays.top();
    condDelays.pop();
    int64_t d2 = condDelays.top();
    condDelays.pop();
    condDelays.push(std::max(d1, d2) + orDelay);
  }

  PriorityMuxInfo result{minDelay, condDelays.top(), bestSplit};
  memo[key] = result;
  return result;
}

// Recursively constructs a balanced binary tree of muxes for a priority
// encoder. Transforms a linear chain into a balanced tree, reducing depth
// from O(n) to O(log n).
// NOLINTNEXTLINE(misc-no-recursion)
Value BalancedPriorityMuxBuilder::buildBalancedPriorityMux(size_t start,
                                                           size_t end) {
  auto conditions = allConditions.slice(start, end - start);
  auto locs = allLocs.slice(start, end - start);
  auto results = allResults.slice(start, end - start);
  auto defaultValue = allResults[end];
  size_t size = conditions.size();

  // Base cases.
  if (size == 0)
    return defaultValue;
  if (size == 1)
    return rewriter.createOrFold<MuxOp>(locs.front(), conditions.front(),
                                        results.front(), defaultValue);

  // Find optimal split point using DP if timing analysis is available
  unsigned mid;
  if (useSimpleSplit()) {
    // Fall back to simple balanced split
    mid = llvm::divideCeil(size, 2);
  } else {
    auto info = computeOptimalDelayAndSplit(start, end);
    LDBG() << "DP: Chosen split point at k=" << info.splitPoint
           << " for range [" << start << ", " << end << ")"
           << " with optimal delay " << info.delay << "\n";
    // Adjust mid to be relative to the current slice
    mid = info.splitPoint - start;
  }

  auto loc = rewriter.getFusedLoc(locs.take_front(mid));

  // Build left and right subtrees. Use results[mid] as the default for the
  // left subtree to ensure correct priority encoding.
  Value leftTree = buildBalancedPriorityMux(start, start + mid - 1);

  Value rightTree = buildBalancedPriorityMux(start + mid, end);

  // Combine conditions from left half with OR
  Value combinedCond =
      rewriter.createOrFold<OrOp>(loc, conditions.take_front(mid), true);

  // Create mux that selects between left and right subtrees
  return rewriter.create<MuxOp>(loc, combinedCond, leftTree, rightTree);
}

Value BalancedPriorityMuxBuilder::build(
    PatternRewriter &rewriter, synth::IncrementalLongestPathAnalysis *analysis,
    ArrayRef<Value> conditions, ArrayRef<Value> results,
    ArrayRef<Location> locs) {
  assert(conditions.size() + 1 == results.size() &&
         "Expected one more result than conditions");

  BalancedPriorityMuxBuilder builder(rewriter, analysis, conditions, results,
                                     locs);

  // Precompute arrival times
  builder.precomputeArrivalTimes(conditions, results);

  // Build the balanced tree
  return builder.buildBalancedPriorityMux(0, conditions.size());
}

//===----------------------------------------------------------------------===//
// PriorityMuxReshape Implementation
//===----------------------------------------------------------------------===//

LogicalResult
PriorityMuxReshape::matchAndRewrite(MuxOp op, PatternRewriter &rewriter) const {
  // Make sure that we're not looking at the intermediate node in a mux tree.
  if (op->hasOneUse())
    if (auto userMux = dyn_cast<MuxOp>(*op->user_begin()))
      return failure();

  // Early return if both or neither side are mux chains.
  auto trueMux = op.getTrueValue().getDefiningOp<MuxOp>();
  auto falseMux = op.getFalseValue().getDefiningOp<MuxOp>();
  if ((trueMux && falseMux) || (!trueMux && !falseMux))
    return failure();
  bool useFalseSideChain = falseMux;

  auto [conditions, results, locs] = collectChain(op, useFalseSideChain);
  if (conditions.size() < muxChainThreshold)
    return failure();

  if (!useFalseSideChain) {
    // For true-side chains, we need to invert all conditions
    for (auto &cond : conditions) {
      cond = rewriter.createOrFold<comb::XorOp>(
          op.getLoc(), cond,
          rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(1, 1)), true);
    }
  }

  LDBG() << "Rebalanced priority mux with " << conditions.size()
         << " conditions, using " << (useFalseSideChain ? "false" : "true")
         << "-side chain.\n";

  // Build balanced tree using helper class
  Value balancedTree = BalancedPriorityMuxBuilder::build(
      rewriter, analysis, conditions, results, locs);
  replaceOpAndCopyNamehint(rewriter, op, balancedTree);
  return success();
}

std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Location>>
PriorityMuxReshape::collectChain(MuxOp op, bool isFalseSide) const {
  SmallVector<Value> chainConditions, chainResults;
  DenseSet<Value> seenConditions;
  SmallVector<Location> chainLocs;

  auto chainMux = isFalseSide ? op.getFalseValue().getDefiningOp<MuxOp>()
                              : op.getTrueValue().getDefiningOp<MuxOp>();

  if (!chainMux)
    return {chainConditions, chainResults, chainLocs};

  // Helper lambdas to abstract the differences between false/true side chains
  auto getChainResult = [&](MuxOp mux) -> Value {
    return isFalseSide ? mux.getTrueValue() : mux.getFalseValue();
  };

  auto getChainNext = [&](MuxOp mux) -> Value {
    return isFalseSide ? mux.getFalseValue() : mux.getTrueValue();
  };

  auto getRootResult = [&]() -> Value {
    return isFalseSide ? op.getTrueValue() : op.getFalseValue();
  };

  // Start collecting the chain
  seenConditions.insert(op.getCond());
  chainConditions.push_back(op.getCond());
  chainResults.push_back(getRootResult());
  chainLocs.push_back(op.getLoc());

  // Walk down the chain collecting all conditions and results
  MuxOp currentMux = chainMux;
  while (currentMux) {
    // Only add unique conditions (outer muxes have priority)
    if (seenConditions.insert(currentMux.getCond()).second) {
      chainConditions.push_back(currentMux.getCond());
      chainResults.push_back(getChainResult(currentMux));
      chainLocs.push_back(currentMux.getLoc());
    }

    auto nextMux = getChainNext(currentMux).getDefiningOp<MuxOp>();
    if (!nextMux || !nextMux->hasOneUse()) {
      // Add the final default value
      chainResults.push_back(getChainNext(currentMux));
      break;
    }
    currentMux = nextMux;
  }

  return {chainConditions, chainResults, chainLocs};
}

//===----------------------------------------------------------------------===//
// BalanceMuxPass Implementation
//===----------------------------------------------------------------------===//

/// Pass that performs enhanced mux chain optimizations
struct BalanceMuxPass : public impl::BalanceMuxBase<BalanceMuxPass> {
  using BalanceMuxBase::BalanceMuxBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    auto *analysis = &getAnalysis<synth::IncrementalLongestPathAnalysis>();

    LDBG() << "Running BalanceMuxPass on operation: " << *op << "\n";

    RewritePatternSet patterns(context);
    patterns.add<MuxChainWithComparison>(context, muxChainThreshold);
    patterns.add<PriorityMuxReshape>(context, muxChainThreshold, analysis);
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal().setMaxIterations(1);
    (void)applyPatternsGreedily(op, std::move(patterns), config);
  }
};
