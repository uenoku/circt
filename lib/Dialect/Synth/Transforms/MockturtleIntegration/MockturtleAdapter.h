//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a mockturtle network adapter for CIRCT's IR.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_ADAPTER_H
#define LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_ADAPTER_H

#define DEBUG_TYPE "mockturtle-adapter"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"

// Forward declare comparison operators for mlir::Value before mockturtle
// includes
namespace mlir {
class Value;

/// Comparison operators for mlir::Value
inline bool operator<(const Value &lhs, const Value &rhs) {
  return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
}

inline bool operator>(const Value &lhs, const Value &rhs) {
  return lhs.getAsOpaquePointer() > rhs.getAsOpaquePointer();
}

inline bool operator<=(const Value &lhs, const Value &rhs) {
  return lhs.getAsOpaquePointer() <= rhs.getAsOpaquePointer();
}

inline bool operator>=(const Value &lhs, const Value &rhs) {
  return lhs.getAsOpaquePointer() >= rhs.getAsOpaquePointer();
}

} // namespace mlir

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"

#include <mockturtle/algorithms/reconv_cut.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>
// Include traits first to define network concepts
#include <mockturtle/traits.hpp>
// Include views for algorithm compatibility
#include <mockturtle/views/cut_view.hpp>
#include <mockturtle/views/fanout_view.hpp>
#include <mockturtle/views/mffc_view.hpp>
// Kitty is included as part of mockturtle
#include <mockturtle/utils/truth_table_cache.hpp>

namespace circt {
namespace synth {
namespace mockturtle_integration {

/// A mockturtle-compatible network adapter for CIRCT IR.
/// This class provides the interface mockturtle expects while operating
/// on CIRCT's MLIR operations.
class CIRCTNetworkAdapter {
public:
  // Mockturtle type aliases with complementable signal support
  using node = mlir::Value;
  // Use PointerIntPair to pack complement bit with Value pointer
  // This matches mockturtle's signal semantics where signals can be
  // complemented
  using signal = llvm::PointerIntPair<mlir::Value, 1, bool>;
  using storage = void; // Dummy storage type for mockturtle compatibility
  using base_type = CIRCTNetworkAdapter; // Required for is_network_type trait

  // Required constants for network type trait
  static constexpr uint32_t max_fanin_size = 2;
  static constexpr uint32_t min_fanin_size = 2;

  // Node and signal indexing for mockturtle compatibility
  using node_type = mlir::Value;
  using signal_type = llvm::PointerIntPair<mlir::Value, 1, bool>;

  // Signal creation utilities

private:
  bool isInput(node n) const {
    assert(n);
    auto *defOp = n.getDefiningOp();
    return !defOp || !isa<aig::AndInverterOp>(defOp);
  }

  bool isGate(node n) const {
    assert(n);
    auto *defOp = n.getDefiningOp();
    return llvm::isa_and_nonnull<aig::AndInverterOp>(defOp) &&
           defOp->getNumOperands() == 2;
  }

  // Traversal state for mockturtle algorithms
  mutable llvm::DenseMap<mlir::Value, uint64_t> visitedMap;
  mutable llvm::DenseMap<mlir::Value, uint64_t> valueMap;
  mutable uint64_t currentTravId = 1;

  // Level cache for timing-aware algorithms
  mutable llvm::DenseMap<mlir::Value, unsigned> levelCache;

  // The root module we're operating on
  Block *block;

  // Node numbering system
  mutable llvm::MapVector<mlir::Value, uint32_t> nodeIndexMap;
  mutable uint32_t nextNodeIndex = 0;
  mutable size_t numGate = 0;

  /// Get or assign an index to a node
  uint32_t getOrAssignNodeIndex(node n) const {
    auto it = nodeIndexMap.find(n);
    if (it != nodeIndexMap.end()) {
      return it->second;
    }

    if (auto definingOp = n.getDefiningOp<aig::AndInverterOp>()) {
      if (definingOp->getNumOperands() == 1) {
        return getOrAssignNodeIndex(definingOp->getOperand(0));
      }
      numGate++;
    }

    uint32_t index = nextNodeIndex++;
    nodeIndexMap[n] = index;
    return index;
  }

public:
  explicit CIRCTNetworkAdapter(Block *block) : block(block) {
    LLVM_DEBUG(llvm::dbgs()
               << "CIRCTNetworkAdapter constructor called with block: " << block
               << "\n");
    collectBits();
  }
  bool isI1Value(mlir::Value v) const { return v.getType().isInteger(1); }

  // Collect all bits (Values) in the block to ensure they are indexed.
  void collectBits() {
    LLVM_DEBUG(llvm::dbgs() << "collectBits called\n");
    for (auto &arg : block->getArguments()) {
      if (isI1Value(arg)) {
        LLVM_DEBUG(llvm::dbgs() << "Indexing block argument: " << arg << "\n");
        getOrAssignNodeIndex(arg);
      }
    }
    for (auto &op : *block) {
      for (auto result : op.getResults()) {
        if (isI1Value(result)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Indexing operation result: " << result << "\n");
          getOrAssignNodeIndex(result);
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "collectBits finished, total nodes: "
                            << nodeIndexMap.size() << "\n");
  }

  //===--------------------------------------------------------------------===//
  // Mockturtle Network Interface
  //===--------------------------------------------------------------------===//

  /// Get the size of the network (number of nodes)
  uint32_t size() const {
    LLVM_DEBUG(llvm::dbgs() << "size() called, returning " << nodeIndexMap.size() << "\n");
    return nodeIndexMap.size();
  }

  /// Get number of gates (logic nodes)
  uint32_t num_gates() const {
    LLVM_DEBUG(llvm::dbgs() << "num_gates() called, returning " << numGate << "\n");
    return numGate;
  }

  /// Check if a node represents a constant
  bool is_constant(node value) const {
    assert(value);
    bool result = isa<hw::ConstantOp>(value.getDefiningOp());
    LLVM_DEBUG(llvm::dbgs() << "is_constant(" << value << ") = " << result << "\n");
    return result;
  }

  /// Check if a node is a combinational input (primary input or block argument)
  bool is_ci(node value) const {
    auto *defOp = value.getDefiningOp();
    bool result = !defOp || !isa<aig::AndInverterOp>(defOp);
    LLVM_DEBUG(llvm::dbgs() << "is_ci(" << value << ") = " << result << "\n");
    return result;
  }

  /// Get the number of fanins for a node
  uint32_t fanin_size(node n) const {
    if (!n)
      return 0;
    auto *defOp = n.getDefiningOp();
    if (!defOp)
      return 0; // Block arguments have no fanins
    uint32_t result = defOp->getNumOperands();
    LLVM_DEBUG(llvm::dbgs() << "fanin_size(" << n << ") = " << result << "\n");
    return result;
  }

  /// Get fanout size (expensive operation, cached if needed)
  uint32_t fanout_size(node n) const {
    if (!n)
      return 0;
    uint32_t result = n.getNumUses();
    LLVM_DEBUG(llvm::dbgs() << "fanout_size(" << n << ") = " << result << "\n");
    return result;
  }

  /// Execute function for each fanin
  template <typename Fn>
  void foreach_fanin(node n, Fn &&fn) const {
    assert(n);
    if (isInput(n))
      return;

    auto andInv = n.getDefiningOp<aig::AndInverterOp>();
    assert(andInv && andInv->getNumOperands() == 2 &&
           "Node is not a valid AND gate with 2 operands");
    auto inverted = andInv.getInverted();

    // Check if the function expects two parameters (signal and index)
    if constexpr (std::is_invocable_v<Fn, signal, size_t>) {
      for (size_t i = 0; i < andInv.getNumOperands(); ++i) {
        auto operand = andInv.getOperand(i);
        auto striped = strip(operand);
        striped.setInt(inverted[i] ^ striped.getInt());
        fn(striped, i);
      }
    } else {
      // Function expects only one parameter (signal)
      for (size_t i = 0; i < andInv.getNumOperands(); ++i) {
        auto operand = andInv.getOperand(i);
        auto striped = strip(operand);
        striped.setInt(inverted[i] ^ striped.getInt());
        fn(striped);
      }
    }
  }

  /// Execute function for each node
  template <typename Fn>
  void foreach_node(Fn &&fn) const {
    // Check if the function expects two parameters (node and index)
    if constexpr (std::is_invocable_v<Fn, node const &>) {
      for (const auto &pair : nodeIndexMap) {
        fn(pair.first);
      }
    } else if constexpr (std::is_invocable_v<Fn, node const &, size_t>) {
      for (const auto &pair : nodeIndexMap) {
        fn(pair.first, pair.second);
      }
    } else {
      // Fallback: try without const reference
      for (const auto &pair : nodeIndexMap) {
        fn(pair.first, pair.second);
      }
    }
  }

  /// Execute function for each gate (logic node)
  template <typename Fn>
  void foreach_gate(Fn &&fn) const {
    // Try to call with both node and index first
    for (const auto &pair : nodeIndexMap) {
      if constexpr (std::is_invocable_v<Fn, node, size_t>) {
        fn(pair.first, pair.second);
      } else if constexpr (std::is_invocable_v<Fn, node const &, size_t>) {
        fn(pair.first, pair.second);
      } else if constexpr (std::is_invocable_v<Fn, node>) {
        fn(pair.first);
      } else if constexpr (std::is_invocable_v<Fn, node const &>) {
        fn(pair.first);
      } else {
        // Fallback: try without const reference
        fn(pair.first, pair.second);
      }
    }
  }

  /// Make a signal from a node
  signal make_signal(node n) const {
    assert(n && "Cannot create signal from null node");
    signal result = signal(n, false); // Return signal with no complement
    LLVM_DEBUG(llvm::dbgs() << "make_signal(" << n << ") = (" << result.getPointer() << ", " << result.getInt() << ")\n");
    return result;
  }

  /// Substitute one node with another
  void substitute_node(node old_node, signal new_signal) {

    assert(old_node && "Old node must be valid");
    auto value = new_signal.getPointer();
    assert(value && "New signal must be valid");

    if (new_signal.getInt()) {
      OpBuilder builder(old_node.getContext());
      builder.setInsertionPointAfterValue(value);
      value = builder.createOrFold<synth::aig::AndInverterOp>(
          value.getLoc(), new_signal.getPointer());
    }

    // Replace all uses of the old node with the new signal's value
    LLVM_DEBUG(llvm::dbgs() << "substitute_node: replacing " << old_node << " with " << value << "\n");
    old_node.replaceAllUsesWith(value);
    take_out_node(old_node);
  }

  /// Substitute for pairs (for fanout_view compatibility)
  void substitute_node(const std::pair<node, signal> &old_pair,
                       signal new_signal) {
    substitute_node(old_pair.first, new_signal);
  }

  /// Get the level of a node (for timing-aware algorithms)
  uint32_t level(node n) const {
    if (isInput(n))
      return 0;

    auto *op = n.getDefiningOp();
    if (op->getNumOperands() == 1)
      return level(op->getOperand(0));

    auto it = levelCache.find(n);
    if (it != levelCache.end()) {
      LLVM_DEBUG(llvm::dbgs() << "level(" << n << ") = " << it->second << " (cached)\n");
      return it->second;
    }

    // Prevent infinite recursion by setting a temporary value
    levelCache[n] = 0;

    // Compute level recursively
    uint32_t maxLevel = 0;
    auto *defOp = n.getDefiningOp();
    if (defOp) {
      for (auto operand : defOp->getOperands()) {
        maxLevel = std::max(maxLevel, level(operand));
      }
    }

    uint32_t nodeLevel = maxLevel + 1;
    levelCache[n] = nodeLevel;
    LLVM_DEBUG(llvm::dbgs() << "level(" << n << ") = " << nodeLevel << " (computed)\n");
    return nodeLevel;
  }

  //===--------------------------------------------------------------------===//
  // Traversal Interface for Mockturtle
  //===--------------------------------------------------------------------===//

  /// Increment traversal ID (starts a new traversal)
  void incr_trav_id() const { 
    ++currentTravId; 
    LLVM_DEBUG(llvm::dbgs() << "incr_trav_id: new trav_id = " << currentTravId << "\n");
  }

  /// Get current traversal ID
  uint64_t trav_id() const { 
    LLVM_DEBUG(llvm::dbgs() << "trav_id() = " << currentTravId << "\n");
    return currentTravId; 
  }

  /// Check if node was visited in current traversal
  uint64_t visited(node n) const {
    auto it = visitedMap.find(n);
    uint64_t result = it != visitedMap.end() ? it->second : 0;
    LLVM_DEBUG(llvm::dbgs() << "visited(" << n << ") = " << result << "\n");
    return result;
  }

  /// Mark node as visited with current traversal ID
  void set_visited(node n, uint64_t travId) const { 
    visitedMap[n] = travId; 
    LLVM_DEBUG(llvm::dbgs() << "set_visited(" << n << ", " << travId << ")\n");
  }

  /// Clear visited flags
  void clear_visited() const { 
    visitedMap.clear(); 
    LLVM_DEBUG(llvm::dbgs() << "clear_visited: cleared visitedMap\n");
  }

  /// Get value for a node
  uint64_t value(node n) const {
    auto it = valueMap.find(n);
    return it != valueMap.end() ? it->second : 0;
  }

  /// Set value for a node
  void set_value(node n, uint64_t val) const { valueMap[n] = val; }

  /// Clear all values
  void clear_values() const { valueMap.clear(); }

  llvm::PointerIntPair<Value, 1, bool> strip(Value value) const {
    if (isInput(value))
      return llvm::PointerIntPair<Value, 1, bool>(value, false);
    auto defOp = value.getDefiningOp<aig::AndInverterOp>();
    if (defOp.getNumOperands() == 1) {
      auto stripped = strip(defOp->getOperand(0));
      return llvm::PointerIntPair<Value, 1, bool>(
          stripped.getPointer(), stripped.getInt() ^ defOp.isInverted(0));
    }
    return llvm::PointerIntPair<Value, 1, bool>(value, false);
  }

  //===--------------------------------------------------------------------===//
  // Additional Mockturtle Network Requirements
  //===--------------------------------------------------------------------===//

  /// Get node index for mockturtle algorithms that require indexing
  uint64_t node_to_index(node n) const {
    // Convert Value to index using its internal pointer representation
    auto stripped = strip(n);
    return getOrAssignNodeIndex(stripped.getPointer());
  }

  /// Get index of a node (alternative name for compatibility)
  uint64_t index_of(node n) const { return node_to_index(n); }

  /// Signal-related methods (compatibility with mockturtle)
  node get_node(signal const &f) const { return f.getPointer(); }

  bool is_complemented(signal const &f) const {
    // Use the complement bit stored in the PointerIntPair
    return f.getInt();
  }

  // Node creation methods with proper signal handling
  signal create_not(signal const &a) {
    // For NOT, we can simply flip the complement bit
    return signal(a.getPointer(), !a.getInt());
  }

  signal create_and(signal const &a, signal const &b) {
    // Create an AND gate in CIRCT
    auto *op = a.getPointer().getDefiningOp();
    if (!op)
      return signal(mlir::Value(), false);

    auto builder = mlir::OpBuilder(op->getContext());
    builder.setInsertionPointAfter(op);

    // This is a placeholder - actual implementation would create AND
    // operation
    return signal(a.getPointer(), false); // Return the first input for now
  }

  signal create_or(signal const &a, signal const &b) {
    // Create an OR gate in CIRCT
    auto *op = a.getPointer().getDefiningOp();
    if (!op)
      return signal(mlir::Value(), false);

    auto builder = mlir::OpBuilder(op->getContext());
    builder.setInsertionPointAfter(op);

    // This is a placeholder - actual implementation would create OR operation
    return signal(a.getPointer(), false); // Return the first input for now
  }

  signal create_xor(signal const &a, signal const &b) {
    // Create an XOR gate in CIRCT
    auto *op = a.getPointer().getDefiningOp();
    if (!op)
      return signal(mlir::Value(), false);

    auto builder = mlir::OpBuilder(op->getContext());
    builder.setInsertionPointAfter(op);

    // This is a placeholder - actual implementation would create XOR
    // operation
    return signal(a.getPointer(), false); // Return the first input for now
  }

  // Node status methods
  bool is_dead(node const &n) const {
    // Check if node is dead (has no fanout)
    return n.use_empty();
  }

  // Node replacement methods - matching aig.hpp signature
  std::optional<std::pair<node, signal>>
  replace_in_node(node const &n, node const &old_node, signal new_signal) {
    // Replace old_node with new_signal in node n
    // This is a placeholder implementation for our read-only adapter
    (void)n;
    (void)old_node;
    (void)new_signal;
    return std::nullopt; // No replacement in our read-only adapter
  }

  void replace_in_outputs(node const &old_node, signal const &new_signal) {
    // Replace old_node with new_signal in outputs
    // This is a placeholder implementation for our read-only adapter
    (void)old_node;
    (void)new_signal;
  }

  /// Check if network has node-to-index mapping capability
  static constexpr bool has_node_to_index_v = true;

  /// Events placeholder for mockturtle compatibility (not implemented)
  struct events_placeholder {
    // For add events - takes a node parameter
    template <typename Fn>
    std::shared_ptr<std::function<void(node const &)>>
    register_add_event(Fn &&fn) {
      return std::make_shared<std::function<void(node const &)>>(
          [fn = std::forward<Fn>(fn)](node const &n) { fn(n); });
    }

    // For modified events - takes node and previous fanins parameter
    template <typename Fn>
    std::shared_ptr<
        std::function<void(node const &, std::vector<signal> const &)>>
    register_modified_event(Fn &&fn) {
      return std::make_shared<
          std::function<void(node const &, std::vector<signal> const &)>>(
          [fn = std::forward<Fn>(fn)](
              node const &n, std::vector<signal> const &prev) { fn(n, prev); });
    }

    // For delete events - takes a node parameter
    template <typename Fn>
    std::shared_ptr<std::function<void(node const &)>>
    register_delete_event(Fn &&fn) {
      return std::make_shared<std::function<void(node const &)>>(
          [fn = std::forward<Fn>(fn)](node const &n) { fn(n); });
    }

    template <typename T>
    void release_add_event(const std::shared_ptr<T> &) {}
    template <typename T>
    void release_modified_event(const std::shared_ptr<T> &) {}
    template <typename T>
    void release_delete_event(const std::shared_ptr<T> &) {}
  };

  /// Get events system (placeholder)
  static events_placeholder events() { return {}; }

  // Compute method for simulation support (required for mockturtle
  // algorithms) Compute methods for simulation support - matching mockturtle
  // signatures exactly
  template <typename Iterator>
  mockturtle::iterates_over_t<Iterator, bool>
  compute(node const &n, Iterator begin, Iterator end) const {
    (void)end;
    if (!is_logic_op(n)) {
      return false; // Default for non-logic ops
    }

    auto *op = n.getDefiningOp();
    if (!op)
      return false;

    if (isa<aig::AndInverterOp>(op)) {
      auto andOp = cast<aig::AndInverterOp>(op);
      auto inverted = andOp.getInverted();

      auto v1 = *begin++;
      auto v2 = *begin;

      bool lhsInv = inverted[0];
      bool rhsInv = inverted[1];

      bool result = (v1 ^ lhsInv) && (v2 ^ rhsInv);
      return result;
    }

    // Default to false for other ops
    return false;
  }

  template <typename Iterator>
  mockturtle::iterates_over_truth_table_t<Iterator>
  compute(node const &n, Iterator begin, Iterator end) const {
    (void)end;
    if (!is_logic_op(n) || begin == end) {
      kitty::dynamic_truth_table result(1);
      return result; // Default to single bit
    }

    if (isa<aig::AndInverterOp>(n.getDefiningOp())) {
      auto andOp = cast<aig::AndInverterOp>(n.getDefiningOp());
      auto inverted = andOp.getInverted();

      auto tt1 = *begin++;
      auto tt2 = *begin;

      bool lhsInv = inverted[0];
      bool rhsInv = inverted[1];

      return (lhsInv ? ~tt1 : tt1) & (rhsInv ? ~tt2 : tt2);
    }

    // Default fallback
    llvm::report_fatal_error("Unsupported operation in compute");
  }

  template <typename Iterator>
  void compute(node const &n, kitty::partial_truth_table &result,
               Iterator begin, Iterator end) const {
    static_assert(
        mockturtle::iterates_over_v<Iterator, kitty::partial_truth_table>,
        "begin and end have to iterate over partial_truth_tables");

    (void)end;
    if (!is_logic_op(n) || begin == end) {
      return; // No-op for non-logic ops
    }

    if (isa<aig::AndInverterOp>(n.getDefiningOp())) {
      auto andOp = cast<aig::AndInverterOp>(n.getDefiningOp());
      auto inverted = andOp.getInverted();

      auto tt1 = *begin++;
      auto tt2 = *begin;

      bool lhsInv = inverted[0];
      bool rhsInv = inverted[1];

      assert(tt1.num_bits() > 0 && "truth tables must not be empty");
      assert(tt1.num_bits() == tt2.num_bits());
      assert(tt1.num_bits() >= result.num_bits());

      result.resize(tt1.num_bits());
      result._bits.back() = (lhsInv ? ~(tt1._bits.back()) : tt1._bits.back()) &
                            (rhsInv ? ~(tt2._bits.back()) : tt2._bits.back());
      result.mask_bits();
    }

    // Default fallback
    llvm::report_fatal_error("Unsupported operation in compute");
  }

  // Revive node method (required for fanout_view)
  void revive_node(node const &n) {
    // In our adapter, this is a no-op since we don't track dead nodes
    // In a full implementation, this would mark the node as alive again
    llvm::report_fatal_error("revive_node is not supported in this adapter");
  }

  /// Take out node (placeholder for mockturtle view compatibility)
  void take_out_node(node n) {
    // Mockturtle expects this method to remove a node from the network.
    // However in mockturtle detached nodes could be referred or even revived
    // later. Revive won't be handled in this adapter but we cannot delete
    // nodes either. So currently it detaches the node from the network.
    LDBG() << "take_out_node: detaching node " << n << "\n";
    block->dump();
    auto andOp = n.getDefiningOp<aig::AndInverterOp>();
    if (!andOp)
      return;
    assert(n.use_empty());
    andOp->dropAllReferences();
  }

  /// Check if a node is a primary input
  bool is_pi(node n) const {
    // In CIRCT, primary inputs are represented as block arguments
    return n == nullptr; // Block arguments don't have defining ops
  }

  /// Get constant signal (false)

  mutable Value constants[2];
  signal get_constant(bool value) const {
    if (constants[value])
      return signal(constants[value], false);
    OpBuilder rewriter(block->getParentOp()->getContext());
    rewriter.setInsertionPointToStart(block);
    auto constOp = rewriter.createOrFold<hw::ConstantOp>(
        block->getParentOp()->getLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), value));
    constants[value] = constOp;
    return signal(constOp, false);
  }

  /// Get the value of a constant node
  bool constant_value(node n) const {
    if (auto constOp = dyn_cast_or_null<hw::ConstantOp>(n.getDefiningOp())) {
      return constOp.getValue().getBoolValue();
    }
    assert(false && "Node is not a constant");
  }

  /// Get node function (for simulation)
  uint32_t node_function(node n) const {
    if (!n)
      return 0;

    // Simple function mapping for basic gates
    if (llvm::isa_and_nonnull<aig::AndInverterOp>(n.getDefiningOp()))
      return 0x8; // AND function
    return 0;     // Unknown function
  }

  /// Increment/decrement fanout size (placeholder for view compatibility)
  uint32_t incr_fanout_size(node n) {
    (void)n;
    return 1;
  }
  uint32_t decr_fanout_size(node n) {
    (void)n;
    return 0;
  }

  /// Increment/decrement value (placeholder for reference counting)
  void incr_value(node n) const { (void)n; }
  void decr_value(node n) const { (void)n; }

  //===--------------------------------------------------------------------===//
  // CIRCT-specific Helper Methods
  //===--------------------------------------------------------------------===//

  /// Check if an operation is a supported logic operation for refactoring
  bool is_logic_op(node n) const {
    if (!n)
      return false;
    auto *defOp = n.getDefiningOp();
    if (!defOp)
      return false;
    return isa<aig::AndInverterOp>(defOp);
  }

  /// Get all logic operations in the module
  void foreach_logic_node(const std::function<void(node)> &fn) const {
    for (const auto &pair : nodeIndexMap) {
      fn(pair.first);
    }
  }
};

} // namespace mockturtle_integration
} // namespace synth
} // namespace circt

// Signal operator overloads for mockturtle-style signal manipulation
namespace circt {
namespace synth {
namespace mockturtle_integration {

// Complement operator (NOT)
inline CIRCTNetworkAdapter::signal
operator!(CIRCTNetworkAdapter::signal const &s) {
  return CIRCTNetworkAdapter::signal(s.getPointer(), !s.getInt());
}

// Positive operator (remove complement)
inline CIRCTNetworkAdapter::signal
operator+(CIRCTNetworkAdapter::signal const &s) {
  return CIRCTNetworkAdapter::signal(s.getPointer(), false);
}

// Negative operator (add complement)
inline CIRCTNetworkAdapter::signal
operator-(CIRCTNetworkAdapter::signal const &s) {
  return CIRCTNetworkAdapter::signal(s.getPointer(), true);
}

// XOR with boolean (conditional complement)
inline CIRCTNetworkAdapter::signal
operator^(CIRCTNetworkAdapter::signal const &s, bool complement) {
  return CIRCTNetworkAdapter::signal(s.getPointer(), s.getInt() ^ complement);
}

} // namespace mockturtle_integration
} // namespace synth
} // namespace circt

//===----------------------------------------------------------------------===//
// Mockturtle Trait Specializations
//===----------------------------------------------------------------------===//

namespace mockturtle {

// Specialize network type trait for CIRCT adapter
template <>
struct is_network_type<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

// Specialize has_node_to_index trait
template <>
struct has_node_to_index<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

// Specialize other required traits
template <>
struct has_foreach_node<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_foreach_gate<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_foreach_fanin<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_get_node<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_size<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_is_constant<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_is_ci<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_fanin_size<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_fanout_size<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_substitute_node<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_is_pi<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_get_constant<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_incr_fanout_size<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_decr_fanout_size<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_incr_value<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_decr_value<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

// Specialize traits for fanout_view as well
template <>
struct is_network_type<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_node_to_index<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_foreach_node<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_foreach_gate<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_foreach_fanin<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_get_node<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_size<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_is_constant<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_is_ci<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_fanin_size<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_fanout_size<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_substitute_node<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_is_pi<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_get_constant<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_incr_fanout_size<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_decr_fanout_size<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_incr_value<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_decr_value<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

// Add creation method traits for basic adapter
template <>
struct has_create_not<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_create_and<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_create_or<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_create_xor<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_is_complemented<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_is_dead<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_replace_in_node<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_replace_in_outputs<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_make_signal<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_constant_value<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

template <>
struct has_node_function<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>
    : std::true_type {};

// Add creation method traits for fanout_view
template <>
struct has_create_not<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_create_and<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_create_or<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_create_xor<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_is_complemented<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_is_dead<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_replace_in_node<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_replace_in_outputs<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_make_signal<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_constant_value<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

template <>
struct has_node_function<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>
    : std::true_type {};

// Network type specializations for mffc_view and cut_view
template <>
struct is_network_type<mockturtle::mffc_view<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>>
    : std::true_type {};

template <>
struct is_network_type<mockturtle::cut_view<mockturtle::fanout_view<
    circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>>
    : std::true_type {};

// Compute trait specializations for views
template <>
struct has_compute<
    mockturtle::mffc_view<mockturtle::fanout_view<
        circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>,
    kitty::dynamic_truth_table> : std::true_type {};

template <>
struct has_compute<
    mockturtle::cut_view<mockturtle::fanout_view<
        circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>,
    kitty::dynamic_truth_table> : std::true_type {};

// Add placeholder for satisfiability_dont_cares function used by refactoring
template <typename Ntk>
inline auto
satisfiability_dont_cares(Ntk const &ntk,
                          std::vector<typename Ntk::node> const &pivots,
                          uint32_t max_tfi_inputs = 8u) {
  // Return empty don't care set for now
  return kitty::dynamic_truth_table(0);
}

} // namespace mockturtle

// STL specializations for mlir::Value
namespace std {
template <>
struct hash<mlir::Value> {
  size_t operator()(const mlir::Value &v) const {
    return std::hash<void *>{}(v.getAsOpaquePointer());
  }
};

template <>
struct equal_to<mlir::Value> {
  bool operator()(const mlir::Value &lhs, const mlir::Value &rhs) const {
    return lhs == rhs;
  }
};

template <>
struct less<mlir::Value> {
  bool operator()(const mlir::Value &lhs, const mlir::Value &rhs) const {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};
} // namespace std

#endif // LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_ADAPTER_H
