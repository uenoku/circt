//===- LowerVariadic.cpp - Lowering Variadic to Binary Ops ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers variadic operations to binary operations using a
// delay-aware algorithm for commutative operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"

#define DEBUG_TYPE "synth-lower-variadic"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_LOWERVARIADIC
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;

//===----------------------------------------------------------------------===//
// Lower Variadic pass
//===----------------------------------------------------------------------===//

namespace {

struct LowerVariadicPass : public impl::LowerVariadicBase<LowerVariadicPass> {
  using LowerVariadicBase::LowerVariadicBase;
  void runOnOperation() override;
};

/// Key for caching binary operations. Represents a pair of operands with their
/// inversion flags, normalized (sorted) so that (a,b) and (b,a) map to the same
/// key. This is similar to StructuralHashKey in the StructuralHash pass.
struct BinaryOpKey {
  OperationName opName;
  // Sorted pair: the operand with smaller opaque value comes first.
  llvm::PointerIntPair<Value, 1> first;
  llvm::PointerIntPair<Value, 1> second;

  bool operator==(const BinaryOpKey &other) const {
    return opName == other.opName && first == other.first &&
           second == other.second;
  }

  /// Create a normalized key for a binary operation.
  static BinaryOpKey create(OperationName opName, Value lhs, bool lhsInv,
                            Value rhs, bool rhsInv) {
    auto p1 = llvm::PointerIntPair<Value, 1>(lhs, lhsInv);
    auto p2 = llvm::PointerIntPair<Value, 1>(rhs, rhsInv);
    // Sort by opaque value for consistency (commutative normalization)
    if (p1.getOpaqueValue() > p2.getOpaqueValue())
      std::swap(p1, p2);
    return BinaryOpKey{opName, p1, p2};
  }
};

} // namespace

namespace llvm {
template <>
struct DenseMapInfo<BinaryOpKey> {
  static BinaryOpKey getEmptyKey() {
    return BinaryOpKey{DenseMapInfo<OperationName>::getEmptyKey(), {}, {}};
  }
  static BinaryOpKey getTombstoneKey() {
    return BinaryOpKey{DenseMapInfo<OperationName>::getTombstoneKey(), {}, {}};
  }
  static unsigned getHashValue(const BinaryOpKey &key) {
    return llvm::hash_combine(hash_value(key.opName),
                              key.first.getOpaqueValue(),
                              key.second.getOpaqueValue());
  }
  static bool isEqual(const BinaryOpKey &lhs, const BinaryOpKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace {

/// Construct a balanced binary tree from a variadic operation using a
/// delay-aware algorithm. This function builds the tree by repeatedly combining
/// the two values with the earliest arrival times, which minimizes the critical
/// path delay.
///
/// When multiple values have the same arrival time, the algorithm checks if any
/// pair already exists in the cache and prefers those pairs. This enables
/// reusing operations across different variadic ops with overlapping operands,
/// as described in issue #9712.
static LogicalResult replaceWithBalancedTree(
    IncrementalLongestPathAnalysis *analysis, mlir::IRRewriter &rewriter,
    Operation *op, llvm::function_ref<bool(OpOperand &)> isInverted,
    llvm::function_ref<Value(ValueWithArrivalTime, ValueWithArrivalTime)>
        createBinaryOp,
    DenseMap<BinaryOpKey, Value> &binaryOpCache) {
  // Collect all operands with their arrival times and inversion flags
  SmallVector<ValueWithArrivalTime> operands;
  size_t valueNumber = 0;

  for (size_t i = 0, e = op->getNumOperands(); i < e; ++i) {
    int64_t delay = 0;
    // If analysis is available, use it to compute the delay.
    // If not available, use zero delay and `valueNumber` will be used instead.
    if (analysis) {
      auto result = analysis->getMaxDelay(op->getOperand(i));
      if (failed(result))
        return failure();
      delay = *result;
    }
    operands.push_back(ValueWithArrivalTime(op->getOperand(i), delay,
                                            isInverted(op->getOpOperand(i)),
                                            valueNumber++));
  }

  if (operands.size() == 1) {
    rewriter.replaceOp(op, operands[0].getValue());
    return success();
  }

  // Min-heap priority queue ordered by arrival time (smaller = higher priority)
  llvm::PriorityQueue<ValueWithArrivalTime, std::vector<ValueWithArrivalTime>,
                      std::greater<ValueWithArrivalTime>>
      pq(operands.begin(), operands.end());

  // Helper to create or reuse a binary operation
  auto combineValues = [&](const ValueWithArrivalTime &lhs,
                           const ValueWithArrivalTime &rhs) {
    // Create a normalized key for this pair
    auto key = BinaryOpKey::create(op->getName(), lhs.getValue(),
                                   lhs.isInverted(), rhs.getValue(),
                                   rhs.isInverted());

    // Check if this pair already exists in the cache
    auto it = binaryOpCache.find(key);
    if (it != binaryOpCache.end()) {
      Value cached = it->second;
      int64_t cachedDelay = 0;
      if (analysis) {
        auto delayResult = analysis->getMaxDelay(cached);
        if (succeeded(delayResult))
          cachedDelay = *delayResult;
      }
      return ValueWithArrivalTime(cached, cachedDelay, false, valueNumber++);
    }

    // Create new binary operation
    Value combined = createBinaryOp(lhs, rhs);
    int64_t newDelay = 0;
    if (analysis) {
      auto delayResult = analysis->getMaxDelay(combined);
      if (succeeded(delayResult))
        newDelay = *delayResult;
    }

    // Cache the new operation for future reuse
    binaryOpCache[key] = combined;

    return ValueWithArrivalTime(combined, newDelay, false, valueNumber++);
  };

  // Build balanced tree, preferring cached pairs when arrival times are equal
  while (pq.size() > 1) {
    // Pop all values with the minimum arrival time
    SmallVector<ValueWithArrivalTime> sameArrival;
    int64_t minArrival = pq.top().getArrivalTime();

    while (!pq.empty() && pq.top().getArrivalTime() == minArrival) {
      sameArrival.push_back(pq.top());
      pq.pop();
    }

    // If we only have one element with minimum arrival, we need one more
    if (sameArrival.size() == 1 && !pq.empty()) {
      sameArrival.push_back(pq.top());
      pq.pop();
    }

    // Among values with same arrival time, find a pair that exists in cache
    std::optional<std::pair<size_t, size_t>> cachedPair;
    for (size_t i = 0; i < sameArrival.size() && !cachedPair; ++i) {
      for (size_t j = i + 1; j < sameArrival.size(); ++j) {
        auto key = BinaryOpKey::create(
            op->getName(), sameArrival[i].getValue(),
            sameArrival[i].isInverted(), sameArrival[j].getValue(),
            sameArrival[j].isInverted());
        if (binaryOpCache.count(key)) {
          cachedPair = {i, j};
          break;
        }
      }
    }

    // Combine the chosen pair (cached pair if found, otherwise first two)
    size_t idx1 = cachedPair ? cachedPair->first : 0;
    size_t idx2 = cachedPair ? cachedPair->second : 1;

    ValueWithArrivalTime combined =
        combineValues(sameArrival[idx1], sameArrival[idx2]);
    pq.push(combined);

    // Push remaining elements back to the queue
    for (size_t i = 0; i < sameArrival.size(); ++i) {
      if (i != idx1 && i != idx2)
        pq.push(sameArrival[i]);
    }
  }

  rewriter.replaceOp(op, pq.top().getValue());
  return success();
}

} // namespace

void LowerVariadicPass::runOnOperation() {
  // Topologically sort operations in graph regions to ensure operands are
  // defined before uses.
  if (!mlir::sortTopologically(
          getOperation().getBodyBlock(), [](Value val, Operation *op) -> bool {
            if (isa_and_nonnull<hw::HWDialect>(op->getDialect()))
              return isa<hw::InstanceOp>(op);
            return !isa_and_nonnull<comb::CombDialect, synth::SynthDialect>(
                op->getDialect());
          })) {
    mlir::emitError(getOperation().getLoc())
        << "Failed to topologically sort graph region blocks";
    return signalPassFailure();
  }

  // Get longest path analysis if timing-aware lowering is enabled.
  synth::IncrementalLongestPathAnalysis *analysis = nullptr;
  if (timingAware.getValue())
    analysis = &getAnalysis<synth::IncrementalLongestPathAnalysis>();

  auto moduleOp = getOperation();

  // Build set of operation names to lower if specified.
  SmallVector<OperationName> names;
  for (const auto &name : opNames)
    names.push_back(OperationName(name, &getContext()));

  // Return true if the operation should be lowered.
  auto shouldLower = [&](Operation *op) {
    // If no names specified, lower all variadic ops.
    if (names.empty())
      return true;
    return llvm::find(names, op->getName()) != names.end();
  };

  mlir::IRRewriter rewriter(&getContext());
  rewriter.setListener(analysis);

  // Cache for reusing binary operations across lowerings.
  // This enables the optimization described in issue #9712: when lowering
  // operations with overlapping operand sets, reuse intermediate results.
  DenseMap<BinaryOpKey, Value> binaryOpCache;

  // FIXME: Currently only top-level operations are lowered due to the lack of
  //        topological sorting in across nested regions.
  for (auto &opRef :
       llvm::make_early_inc_range(moduleOp.getBodyBlock()->getOperations())) {
    auto *op = &opRef;
    // Skip operations that don't need lowering or are already binary.
    if (!shouldLower(op) || op->getNumOperands() <= 2)
      continue;

    rewriter.setInsertionPoint(op);

    // Handle AndInverterOp specially to preserve inversion flags.
    if (auto andInverterOp = dyn_cast<aig::AndInverterOp>(op)) {
      auto result = replaceWithBalancedTree(
          analysis, rewriter, op,
          // Check if each operand is inverted.
          [&](OpOperand &operand) {
            return andInverterOp.isInverted(operand.getOperandNumber());
          },
          // Create binary AndInverterOp with inversion flags.
          [&](ValueWithArrivalTime lhs, ValueWithArrivalTime rhs) {
            return aig::AndInverterOp::create(
                rewriter, op->getLoc(), lhs.getValue(), rhs.getValue(),
                lhs.isInverted(), rhs.isInverted());
          },
          binaryOpCache);
      if (failed(result))
        return signalPassFailure();
      continue;
    }

    // Handle commutative operations (and, or, xor, mul, add, etc.) using
    // delay-aware lowering to minimize critical path.
    if (isa_and_nonnull<comb::CombDialect>(op->getDialect()) &&
        op->hasTrait<OpTrait::IsCommutative>()) {
      auto result = replaceWithBalancedTree(
          analysis, rewriter, op,
          // No inversion flags for standard commutative operations.
          [](OpOperand &) { return false; },
          // Create binary operation with the same operation type.
          [&](ValueWithArrivalTime lhs, ValueWithArrivalTime rhs) {
            OperationState state(op->getLoc(), op->getName());
            state.addOperands(ValueRange{lhs.getValue(), rhs.getValue()});
            state.addTypes(op->getResult(0).getType());
            auto *newOp = Operation::create(state);
            rewriter.insert(newOp);
            return newOp->getResult(0);
          },
          binaryOpCache);
      if (failed(result))
        return signalPassFailure();
    }
  }
}
