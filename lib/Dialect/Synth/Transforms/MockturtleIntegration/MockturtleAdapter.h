//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a mockturtle network adapter for CIRCT's IR.
// This allows us to use mockturtle's algorithms (like reconvergence-driven
// cuts) directly on CIRCT IR structures.
//
// This is an internal implementation detail and not part of the public API.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_ADAPTER_H
#define LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_ADAPTER_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <mockturtle/algorithms/reconv_cut.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>
// Include traits first to define network concepts
#include <mockturtle/traits.hpp>
// Include views for algorithm compatibility
#include <mockturtle/views/fanout_view.hpp>
#include <mockturtle/views/mffc_view.hpp>
#include <mockturtle/views/cut_view.hpp>
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
  // Mockturtle type aliases
  using node = mlir::Operation *;
  using signal = mlir::Value;
  using storage = void; // Dummy storage type for mockturtle compatibility
  using base_type = CIRCTNetworkAdapter; // Required for is_network_type trait
  
  // Required constants for network type trait
  static constexpr uint32_t max_fanin_size = 32;
  static constexpr uint32_t min_fanin_size = 0;
  
  // Node and signal indexing for mockturtle compatibility
  using node_type = mlir::Operation *;
  using signal_type = mlir::Value;

private:
  // Traversal state for mockturtle algorithms
  mutable llvm::DenseMap<mlir::Operation *, uint64_t> visitedMap;
  mutable llvm::DenseMap<mlir::Operation *, uint64_t> valueMap;
  mutable uint64_t currentTravId = 1;

  // Level cache for timing-aware algorithms
  mutable llvm::DenseMap<mlir::Operation *, unsigned> levelCache;

  // The root module we're operating on
  hw::HWModuleOp module;

  // Cache of all nodes for iteration
  mutable std::vector<node> allNodes;
  mutable bool nodesCacheValid = false;

  void updateNodesCache() const {
    if (nodesCacheValid) return;
    
    allNodes.clear();
    auto moduleCopy = module;
    moduleCopy->walk([&](mlir::Operation *op) {
      if (is_logic_op(op)) {
        allNodes.push_back(op);
      }
    });
    nodesCacheValid = true;
  }

public:
  explicit CIRCTNetworkAdapter(hw::HWModuleOp mod) : module(mod) {}

  //===--------------------------------------------------------------------===//
  // Mockturtle Network Interface
  //===--------------------------------------------------------------------===//

  /// Get the size of the network (number of nodes)
  uint32_t size() const {
    updateNodesCache();
    return allNodes.size();
  }

  /// Get number of gates (logic nodes)
  uint32_t num_gates() const {
    updateNodesCache();
    return allNodes.size();
  }

  /// Check if a node represents a constant
  bool is_constant(node n) const { return isa<hw::ConstantOp>(n); }

  /// Check if a node is a combinational input (primary input or block argument)
  bool is_ci(node n) const {
    // Primary inputs are represented as block arguments in CIRCT
    return n == nullptr; // Block arguments don't have defining ops
  }

  /// Get the number of fanins for a node
  uint32_t fanin_size(node n) const {
    if (!n)
      return 0;
    return n->getNumOperands();
  }

  /// Get fanout size (expensive operation, cached if needed)
  uint32_t fanout_size(node n) const {
    if (!n)
      return 0;
    uint32_t count = 0;
    for (auto result : n->getResults())
      count += result.getNumUses();
    return count;
  }

  /// Execute function for each fanin
  template <typename Fn>
  void foreach_fanin(node n, Fn &&fn) const {
    if (!n)
      return;

    // Check if the function expects two parameters (signal and index)
    if constexpr (std::is_invocable_v<Fn, signal, size_t>) {
      for (size_t i = 0; i < n->getNumOperands(); ++i) {
        fn(n->getOperand(i), i);
      }
    } else {
      // Function expects only one parameter (signal)
      for (auto operand : n->getOperands()) {
        fn(operand);
      }
    }
  }

  /// Execute function for each node
  template <typename Fn>
  void foreach_node(Fn &&fn) const {
    updateNodesCache();
    
    // Check if the function expects two parameters (node and index)
    if constexpr (std::is_invocable_v<Fn, node const&>) {
      for (const auto& nodePtr : allNodes) {
        fn(nodePtr);
      }
    } else if constexpr (std::is_invocable_v<Fn, node const&, size_t>) {
      for (size_t i = 0; i < allNodes.size(); ++i) {
        fn(allNodes[i], i);
      }
    } else {
      // Fallback: try without const reference
      for (size_t i = 0; i < allNodes.size(); ++i) {
        fn(allNodes[i], i);
      }
    }
  }

  /// Execute function for each gate (logic node)
  template <typename Fn>
  void foreach_gate(Fn &&fn) const {
    updateNodesCache();
    
    // Check if the function expects two parameters (node and index)
    if constexpr (std::is_invocable_v<Fn, node, size_t>) {
      for (size_t i = 0; i < allNodes.size(); ++i) {
        fn(allNodes[i], i);
      }
    } else if constexpr (std::is_invocable_v<Fn, node>) {
      for (const auto& nodePtr : allNodes) {
        fn(nodePtr);
      }
    } else {
      // Fallback: try node with index
      for (size_t i = 0; i < allNodes.size(); ++i) {
        fn(allNodes[i], i);
      }
    }
  }

  /// Make a signal from a node
  signal make_signal(node n) const {
    if (!n || n->getNumResults() == 0)
      return {};
    return n->getResult(0);
  }

  /// Substitute one node with another
  void substitute_node(node old_node, signal new_signal) {
    if (!old_node || old_node->getNumResults() == 0)
      return;
    
    // Replace all uses of the old node with the new signal
    old_node->getResult(0).replaceAllUsesWith(new_signal);
    nodesCacheValid = false; // Invalidate cache
  }

  /// Substitute for pairs (for fanout_view compatibility)
  void substitute_node(const std::pair<node, signal>& old_pair, signal new_signal) {
    substitute_node(old_pair.first, new_signal);
  }

  /// Get the level of a node (for timing-aware algorithms)
  uint32_t level(node n) const {
    if (!n)
      return 0;

    auto it = levelCache.find(n);
    if (it != levelCache.end()) {
      return it->second;
    }

    // Prevent infinite recursion by setting a temporary value
    levelCache[n] = 0;

    // Compute level recursively
    uint32_t maxLevel = 0;
    for (auto operand : n->getOperands()) {
      if (auto *definingOp = operand.getDefiningOp()) {
        maxLevel = std::max(maxLevel, level(definingOp));
      }
    }

    uint32_t nodeLevel = maxLevel + 1;
    levelCache[n] = nodeLevel;
    return nodeLevel;
  }

  //===--------------------------------------------------------------------===//
  // Traversal Interface for Mockturtle
  //===--------------------------------------------------------------------===//

  /// Increment traversal ID (starts a new traversal)
  void incr_trav_id() const { ++currentTravId; }

  /// Get current traversal ID
  uint64_t trav_id() const { return currentTravId; }

  /// Check if node was visited in current traversal
  uint64_t visited(node n) const {
    auto it = visitedMap.find(n);
    return it != visitedMap.end() ? it->second : 0;
  }

  /// Mark node as visited with current traversal ID
  void set_visited(node n, uint64_t travId) const { visitedMap[n] = travId; }

  /// Clear visited flags
  void clear_visited() const { visitedMap.clear(); }

  /// Get value for a node
  uint64_t value(node n) const {
    auto it = valueMap.find(n);
    return it != valueMap.end() ? it->second : 0;
  }

  /// Set value for a node
  void set_value(node n, uint64_t val) const { valueMap[n] = val; }

  /// Clear all values
  void clear_values() const { valueMap.clear(); }

  //===--------------------------------------------------------------------===//
  // Additional Mockturtle Network Requirements  
  //===--------------------------------------------------------------------===//

  /// Get node index for mockturtle algorithms that require indexing
  uint64_t node_to_index(node n) const {
    return reinterpret_cast<uint64_t>(n);
  }

  /// Get index of a node (alternative name for compatibility)
  uint64_t index_of(node n) const {
    return node_to_index(n);
  }
  
  /// Signal-related methods (compatibility with mockturtle)
  node get_node(signal const& f) const {
    return f.getDefiningOp();
  }
  
  bool is_complemented(signal const& f) const {
    // For CIRCT, all signals are positive
    return false;
  }
  
  // Node creation methods
  signal create_not(signal const& a) {
    // Create a NOT gate in CIRCT
    auto op = a.getDefiningOp();
    if (!op) return {};
    
    auto builder = mlir::OpBuilder(op->getContext());
    builder.setInsertionPointAfter(op);
    
    // This is a placeholder - actual implementation would create NOT operation
    return a; // Return the input for now
  }
  
  signal create_and(signal const& a, signal const& b) {
    // Create an AND gate in CIRCT
    auto op = a.getDefiningOp();
    if (!op) return {};
    
    auto builder = mlir::OpBuilder(op->getContext());
    builder.setInsertionPointAfter(op);
    
    // This is a placeholder - actual implementation would create AND operation
    return a; // Return the first input for now
  }
  
  signal create_or(signal const& a, signal const& b) {
    // Create an OR gate in CIRCT
    auto op = a.getDefiningOp();
    if (!op) return {};
    
    auto builder = mlir::OpBuilder(op->getContext());
    builder.setInsertionPointAfter(op);
    
    // This is a placeholder - actual implementation would create OR operation
    return a; // Return the first input for now
  }
  
  signal create_xor(signal const& a, signal const& b) {
    // Create an XOR gate in CIRCT
    auto op = a.getDefiningOp();
    if (!op) return {};
    
    auto builder = mlir::OpBuilder(op->getContext());
    builder.setInsertionPointAfter(op);
    
    // This is a placeholder - actual implementation would create XOR operation
    return a; // Return the first input for now
  }
  
  // Node status methods
  bool is_dead(node const& n) const {
    // Check if node is dead (has no fanout)
    return n->use_empty();
  }
  
  // Node replacement methods - matching aig.hpp signature
  std::optional<std::pair<node, signal>> replace_in_node(node const& n, node const& old_node, signal new_signal) {
    // Replace old_node with new_signal in node n
    // This is a placeholder implementation for our read-only adapter
    (void)n; (void)old_node; (void)new_signal;
    return std::nullopt; // No replacement in our read-only adapter
  }
  
  void replace_in_outputs(node const& old_node, signal const& new_signal) {
    // Replace old_node with new_signal in outputs
    // This is a placeholder implementation for our read-only adapter
    (void)old_node; (void)new_signal;
  }

  /// Check if network has node-to-index mapping capability
  static constexpr bool has_node_to_index_v = true;

  /// Events placeholder for mockturtle compatibility (not implemented)
  struct events_placeholder {
    // For add events - takes a node parameter
    template<typename Fn>
    std::shared_ptr<std::function<void(node const&)>> register_add_event(Fn&& fn) {
      return std::make_shared<std::function<void(node const&)>>(
        [fn = std::forward<Fn>(fn)](node const& n) { fn(n); }
      );
    }
    
    // For modified events - takes node and previous fanins parameter
    template<typename Fn>
    std::shared_ptr<std::function<void(node const&, std::vector<signal> const&)>> register_modified_event(Fn&& fn) {
      return std::make_shared<std::function<void(node const&, std::vector<signal> const&)>>(
        [fn = std::forward<Fn>(fn)](node const& n, std::vector<signal> const& prev) { fn(n, prev); }
      );
    }
    
    // For delete events - takes a node parameter  
    template<typename Fn>
    std::shared_ptr<std::function<void(node const&)>> register_delete_event(Fn&& fn) {
      return std::make_shared<std::function<void(node const&)>>(
        [fn = std::forward<Fn>(fn)](node const& n) { fn(n); }
      );
    }
    
    template<typename T>
    void release_add_event(const std::shared_ptr<T>&) {}
    template<typename T>
    void release_modified_event(const std::shared_ptr<T>&) {}
    template<typename T>
    void release_delete_event(const std::shared_ptr<T>&) {}
  };

  /// Get events system (placeholder)
  static events_placeholder events() { return {}; }

  // Compute method for simulation support (required for mockturtle algorithms)
  // Compute methods for simulation support - matching mockturtle signatures exactly
  template<typename Iterator>
  mockturtle::iterates_over_t<Iterator, bool>
  compute(node const& n, Iterator begin, Iterator end) const {
    (void)end;
    if (!is_logic_op(n)) {
      return false; // Default for non-logic ops
    }
    
    if (isa<comb::AndOp>(n)) {
      auto v1 = *begin++;
      if (begin == end) return v1;
      auto v2 = *begin;
      return v1 && v2;
    }
    
    // Default to false for other ops
    return false;
  }

  template<typename Iterator>
  mockturtle::iterates_over_truth_table_t<Iterator>
  compute(node const& n, Iterator begin, Iterator end) const {
    (void)end;
    if (!is_logic_op(n) || begin == end) {
      kitty::dynamic_truth_table result(1);
      return result; // Default to single bit
    }
    
    if (isa<comb::AndOp>(n)) {
      auto tt1 = *begin++;
      if (begin == end) return tt1;
      auto tt2 = *begin;
      return tt1 & tt2;
    }
    
    // Default fallback
    auto result = *begin;
    for (auto it = begin + 1; it != end; ++it) {
      result = result & (*it);
    }
    return result;
  }

  template<typename Iterator>
  void compute(node const& n, kitty::partial_truth_table& result, Iterator begin, Iterator end) const {
    static_assert(mockturtle::iterates_over_v<Iterator, kitty::partial_truth_table>, 
                  "begin and end have to iterate over partial_truth_tables");
    
    (void)end;
    if (!is_logic_op(n) || begin == end) {
      return; // No-op for non-logic ops
    }
    
    if (isa<comb::AndOp>(n)) {
      auto tt1 = *begin++;
      if (begin == end) {
        result = tt1;
        return;
      }
      auto tt2 = *begin;
      
      assert(tt1.num_bits() > 0 && "truth tables must not be empty");
      assert(tt1.num_bits() == tt2.num_bits());
      assert(tt1.num_bits() >= result.num_bits());
      
      result.resize(tt1.num_bits());
      result._bits.back() = tt1._bits.back() & tt2._bits.back();
      result.mask_bits();
    }
  }
  
  // Revive node method (required for fanout_view)
  void revive_node(node const& n) {
    // In our adapter, this is a no-op since we don't track dead nodes
    // In a full implementation, this would mark the node as alive again
    (void)n;
  }

  /// Take out node (placeholder for mockturtle view compatibility)
  void take_out_node(node n) {
    // This is a no-op for our read-only adapter
    (void)n;
  }

  /// Check if a node is a primary input
  bool is_pi(node n) const {
    // In CIRCT, primary inputs are represented as block arguments
    return n == nullptr; // Block arguments don't have defining ops
  }

  /// Get constant signal (false)
  signal get_constant(bool value) const {
    // For now, return an empty signal - would need to create actual constant ops
    (void)value;
    return {};
  }

  /// Get the value of a constant node
  bool constant_value(node n) const {
    if (auto constOp = dyn_cast<hw::ConstantOp>(n)) {
      return constOp.getValue().getBoolValue();
    }
    return false; // Default to false for non-constants
  }

  /// Get node function (for simulation)
  uint32_t node_function(node n) const {
    if (!n) return 0;
    
    // Simple function mapping for basic gates
    if (isa<comb::AndOp>(n)) return 0x8; // AND function
    if (isa<comb::OrOp>(n)) return 0xE;  // OR function  
    if (isa<comb::XorOp>(n)) return 0x6; // XOR function
    return 0; // Unknown function
  }

  /// Increment/decrement fanout size (placeholder for view compatibility)
  uint32_t incr_fanout_size(node n) { (void)n; return 1; }
  uint32_t decr_fanout_size(node n) { (void)n; return 0; }

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
    return isa<comb::AndOp, comb::OrOp, comb::XorOp>(n);
  }

  /// Get all logic operations in the module
  void foreach_logic_node(const std::function<void(node)> &fn) const {
    updateNodesCache();
    for (auto *n : allNodes) {
      fn(n);
    }
  }
};

} // namespace mockturtle_integration
} // namespace synth
} // namespace circt

//===----------------------------------------------------------------------===//
// Mockturtle Trait Specializations
//===----------------------------------------------------------------------===//

namespace mockturtle {

// Specialize network type trait for CIRCT adapter
template<>
struct is_network_type<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

// Specialize has_node_to_index trait
template<>
struct has_node_to_index<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

// Specialize other required traits
template<>
struct has_foreach_node<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_foreach_gate<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_foreach_fanin<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_get_node<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_size<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_is_constant<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_is_ci<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_fanin_size<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_fanout_size<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_substitute_node<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_is_pi<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_get_constant<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_incr_fanout_size<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_decr_fanout_size<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_incr_value<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_decr_value<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

// Specialize traits for fanout_view as well
template<>
struct is_network_type<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_node_to_index<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_foreach_node<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_foreach_gate<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_foreach_fanin<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_get_node<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_size<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_is_constant<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_is_ci<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_fanin_size<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_fanout_size<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_substitute_node<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_is_pi<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_get_constant<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_incr_fanout_size<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_decr_fanout_size<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_incr_value<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_decr_value<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

// Add creation method traits for basic adapter
template<>
struct has_create_not<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_create_and<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_create_or<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_create_xor<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_is_complemented<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_is_dead<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_replace_in_node<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_replace_in_outputs<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_make_signal<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_constant_value<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

template<>
struct has_node_function<circt::synth::mockturtle_integration::CIRCTNetworkAdapter> : std::true_type {};

// Add creation method traits for fanout_view
template<>
struct has_create_not<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_create_and<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_create_or<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_create_xor<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_is_complemented<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_is_dead<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_replace_in_node<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_replace_in_outputs<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_make_signal<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_constant_value<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

template<>
struct has_node_function<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>> : std::true_type {};

// Network type specializations for mffc_view and cut_view 
template<>
struct is_network_type<mockturtle::mffc_view<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>> : std::true_type {};

template<>
struct is_network_type<mockturtle::cut_view<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>> : std::true_type {};

// Compute trait specializations for views
template<>
struct has_compute<mockturtle::mffc_view<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>, kitty::dynamic_truth_table> : std::true_type {};

template<>
struct has_compute<mockturtle::cut_view<mockturtle::fanout_view<circt::synth::mockturtle_integration::CIRCTNetworkAdapter>>, kitty::dynamic_truth_table> : std::true_type {};

// Add placeholder for satisfiability_dont_cares function used by refactoring
template<typename Ntk>
inline auto satisfiability_dont_cares(Ntk const& ntk, std::vector<typename Ntk::node> const& pivots, uint32_t max_tfi_inputs = 8u) {
  // Return empty don't care set for now
  return kitty::dynamic_truth_table(0);
}

} // namespace mockturtle

#endif // LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_ADAPTER_H
