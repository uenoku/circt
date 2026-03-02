//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the Synth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHOPS_H
#define CIRCT_DIALECT_SYNTH_SYNTHOPS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/IR/Value.h"
#include <mlir/IR/Value.h>

#include "llvm/ADT/PriorityQueue.h"

namespace circt {
namespace synth {
namespace aig {
class AndInverterOp;
} // namespace aig
namespace mig {
class MajorityInverterOp;
} // namespace mig

struct AndInverterVariadicOpConversion
    : mlir::OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(aig::AndInverterOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// This function performs a topological sort on the operations within each
/// block of graph regions in the given operation. It uses MLIR's topological
/// sort utility as a wrapper, ensuring that operations are ordered such that
/// all operands are defined before their uses. The `isOperandReady` callback
/// allows customization of when an operand is considered ready for sorting.
LogicalResult topologicallySortGraphRegionBlocks(
    mlir::Operation *op,
    llvm::function_ref<bool(mlir::Value, mlir::Operation *)> isOperandReady);

/// Look up the `synth.liberty.library` attribute for a Liberty cell.
///
/// The search order is:
///   1. The `hw::HWModuleOp` cell itself (set by the LinkLiberty pass for
///      per-cell provenance).
///   2. Each ancestor `mlir::ModuleOp` walking upward (covers the case where
///      cells live inside a nested ModuleOp that carries the library attr, or
///      a top-level ModuleOp with a merged value).
///
/// Returns a null `DictionaryAttr` when no attribute is found.
inline mlir::DictionaryAttr getLibertyLibraryAttr(hw::HWModuleOp cellOp) {
  // 1. Check the cell itself first.
  if (auto attr =
          cellOp->getAttrOfType<mlir::DictionaryAttr>("synth.liberty.library"))
    return attr;

  // 2. Walk ancestor ModuleOps (nested library module, then top-level).
  mlir::Operation *parent = cellOp->getParentOp();
  while (parent) {
    if (auto mod = mlir::dyn_cast<mlir::ModuleOp>(parent)) {
      if (auto attr =
              mod->getAttrOfType<mlir::DictionaryAttr>("synth.liberty.library"))
        return attr;
    }
    parent = parent->getParentOp();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Delay-Aware Tree Building Utilities
//===----------------------------------------------------------------------===//

/// Helper struct to represent a value that may be inverted.
struct InvertibleValue {
  llvm::PointerIntPair<mlir::Value, 1, bool> value;
  InvertibleValue(mlir::Value value, bool inverted = false)
      : value({value, inverted}) {}

  void operator^=(bool invert) { value.setInt(value.getInt() ^ invert); }
  InvertibleValue operator^(bool invert) const {
    return InvertibleValue(value.getPointer(), value.getInt() ^ invert);
  }
  bool operator==(const InvertibleValue &other) const {
    return value == other.value;
  }
  InvertibleValue operator!() const {
    return InvertibleValue(value.getPointer(), !value.getInt());
  }

  /// Returns true if the value is the same as the other value and inverted.
  bool isComplementary(mlir::Value other) const {
    return value.getPointer() == other && value.getInt();
  }

  bool isComplementary(const InvertibleValue &other) const {
    return value.getPointer() == other.value.getPointer() &&
           value.getInt() != other.value.getInt();
  }

  /// Returns true if the value is the same as the other value and not inverted.
  bool isEquivalent(mlir::Value other) const {
    return value.getPointer() == other && !value.getInt();
  }

  bool isEquivalent(const InvertibleValue &other) const {
    return value == other.value;
  }

  InvertibleValue &flipInversion() {
    value.setInt(!value.getInt());
    return *this;
  }

  bool isInverted() const { return value.getInt(); }
  mlir::Value getValue() const { return value.getPointer(); }
};

/// Helper struct to represent a value that may be inverted and has timing info.
/// Used for delay-aware tree building. Stores a value along with its arrival
/// time (depth) and an optional value numbering for deterministic ordering.
struct TimedInvertibleValue : public InvertibleValue {
  /// The arrival time (delay/depth) of this value in the circuit.
  int64_t depth;

  /// Value numbering for deterministic ordering when arrival times are equal.
  /// This ensures consistent results across runs when multiple values have
  /// the same delay.
  size_t valueNumbering = 0;

  TimedInvertibleValue(mlir::Value value, bool inverted, int64_t depth,
                       size_t valueNumbering = 0)
      : InvertibleValue(value, inverted), depth(depth),
        valueNumbering(valueNumbering) {}

  int64_t getArrivalTime() const { return depth; }
  int64_t getDepth() const { return depth; }

  TimedInvertibleValue &flipInversion() {
    InvertibleValue::flipInversion();
    return *this;
  }

  /// Comparison operators for sorting and priority queue usage.
  bool operator<(const TimedInvertibleValue &other) const {
    return depth < other.depth;
  }

  /// Values with earlier arrival times have higher priority.
  /// When arrival times are equal, use value numbering for determinism.
  bool operator>(const TimedInvertibleValue &other) const {
    return std::tie(depth, valueNumbering) >
           std::tie(other.depth, other.valueNumbering);
  }
};

/// Build a balanced binary tree using a priority queue to greedily pair
/// elements with earliest arrival times. This minimizes the critical path
/// delay.
///
/// Template parameters:
///   T - The element type (must have operator> defined)
///
/// The algorithm uses a min-heap to repeatedly combine the two elements with
/// the earliest arrival times, which is optimal for minimizing maximum delay.
template <typename T>
T buildBalancedTreeWithArrivalTimes(llvm::ArrayRef<T> elements,
                                    llvm::function_ref<T(T, T)> combine) {
  assert(!elements.empty() && "Cannot build tree from empty elements");

  if (elements.size() == 1)
    return elements[0];
  if (elements.size() == 2)
    return combine(elements[0], elements[1]);

  // Min-heap priority queue ordered by operator>
  llvm::PriorityQueue<T, std::vector<T>, std::greater<T>> pq(elements.begin(),
                                                             elements.end());

  // Greedily pair the two earliest-arriving elements
  while (pq.size() > 1) {
    T e1 = pq.top();
    pq.pop();
    T e2 = pq.top();
    pq.pop();

    // Combine the two elements
    T combined = combine(e1, e2);
    pq.push(combined);
  }

  return pq.top();
}

/// Helper struct to represent values that may be inverted and have a depth.
struct OrderedValues {
  SmallVector<TimedInvertibleValue, 3> invertibleValues;

  OrderedValues(OperandRange operands, ArrayRef<bool> inversions,
                ArrayRef<int64_t> depths);

  static FailureOr<OrderedValues> get(mig::MajorityInverterOp op,
                                      IncrementalLongestPathAnalysis *analysis);

  void dump(llvm::raw_ostream &os) const {
    for (size_t i = 0; i < invertibleValues.size(); ++i) {
      os << "  Child " << i << ": " << invertibleValues[i].value.getPointer()
         << " (inverted: " << invertibleValues[i].value.getInt()
         << ", depth: " << invertibleValues[i].depth << ")\n";
    }
  }

  Value getValue(size_t idx) const { return invertibleValues[idx].getValue(); }
  bool isInverted(size_t idx) const {
    return invertibleValues[idx].isInverted();
  }
  int64_t getDepth(size_t idx) const { return invertibleValues[idx].depth; }

  auto &operator[](size_t idx) { return invertibleValues[idx]; }
  auto operator[](size_t idx) const { return invertibleValues[idx]; }
};
} // namespace synth
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.h.inc"

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_H
