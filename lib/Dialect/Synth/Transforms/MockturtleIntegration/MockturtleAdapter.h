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

    for (auto operand : n->getOperands()) {
      fn(operand);
    }
  }

  /// Execute function for each node
  template <typename Fn>
  void foreach_node(Fn &&fn) const {
    updateNodesCache();
    for (size_t i = 0; i < allNodes.size(); ++i) {
      fn(allNodes[i], i);
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
    } else {
      // Function expects only one parameter (node)
      for (auto *nodePtr : allNodes) {
        fn(nodePtr);
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
  signal make_signal(node const& n) const {
    // In CIRCT, a signal is typically the first result of an operation
    if (n && n->getNumResults() > 0) {
      return n->getResult(0);
    }
    return {};
  }
  
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
  
  // Node replacement methods
  std::optional<signal> replace_in_node(node const& n, signal old_f, signal new_f) {
    // Replace old signal with new signal in node
    // Return the new signal if replacement happened
    if (old_f != new_f) {
      old_f.replaceUsesWithIf(new_f, [&](mlir::OpOperand& use) {
        return use.getOwner() == n;
      });
      return new_f;
    }
    return std::nullopt;
  }
  
  void replace_in_outputs(signal old_f, signal new_f) {
    // Replace old signal with new signal in outputs
    old_f.replaceAllUsesWith(new_f);
  }

  /// Check if network has node-to-index mapping capability
  static constexpr bool has_node_to_index_v = true;

  /// Events placeholder for mockturtle compatibility (not implemented)
  struct events_placeholder {
    template<typename Fn>
    std::shared_ptr<Fn> register_add_event(Fn&& fn) { 
      return std::make_shared<Fn>(std::forward<Fn>(fn)); 
    }
    template<typename Fn>
    std::shared_ptr<Fn> register_modified_event(Fn&& fn) { 
      return std::make_shared<Fn>(std::forward<Fn>(fn)); 
    }
    template<typename Fn>
    std::shared_ptr<Fn> register_delete_event(Fn&& fn) { 
      return std::make_shared<Fn>(std::forward<Fn>(fn)); 
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

// Add placeholder for satisfiability_dont_cares function used by refactoring
template<typename Ntk>
inline auto satisfiability_dont_cares(Ntk const& ntk, std::vector<typename Ntk::node> const& pivots, uint32_t max_tfi_inputs = 8u) {
  // Return empty don't care set for now
  return kitty::dynamic_truth_table(0);
}

} // namespace mockturtle

#endif // LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_ADAPTER_H
