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
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#ifdef CIRCT_MOCKTURTLE_INTEGRATION_ENABLED
#include <mockturtle/algorithms/reconv_cut.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>
#endif

namespace circt {
namespace synth {
namespace mockturtle_integration {

#ifdef CIRCT_MOCKTURTLE_INTEGRATION_ENABLED

/// A mockturtle-compatible network adapter for CIRCT IR.
/// This class provides the interface mockturtle expects while operating
/// on CIRCT's MLIR operations.
class CIRCTNetworkAdapter {
public:
  // Mockturtle type aliases
  using node = mlir::Operation *;
  using signal = mlir::Value;

private:
  // Traversal state for mockturtle algorithms
  mutable llvm::DenseMap<mlir::Operation *, uint64_t> visitedMap;
  mutable uint64_t currentTravId = 1;

  // Level cache for timing-aware algorithms
  mutable llvm::DenseMap<mlir::Operation *, unsigned> levelCache;

  // The root module we're operating on
  hw::HWModuleOp module;

public:
  explicit CIRCTNetworkAdapter(hw::HWModuleOp mod) : module(mod) {}

  //===--------------------------------------------------------------------===//
  // Mockturtle Network Interface
  //===--------------------------------------------------------------------===//

  /// Check if a node represents a constant
  bool is_constant(node n) const { return isa<hw::ConstantOp>(n); }

  /// Check if a node is a combinational input (primary input or block argument)
  bool is_ci(node n) const {
    // Primary inputs are represented as block arguments in CIRCT
    return n == nullptr; // Block arguments don't have defining ops
  }

  /// Get the node that defines a signal
  node get_node(signal s) const { return s.getDefiningOp(); }

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
    for (auto result : n->getResults()) {
      count += std::distance(result.use_begin(), result.use_end());
    }
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

  /// Get the level of a node (for timing-aware algorithms)
  uint32_t level(node n) const {
    if (!n)
      return 0;

    auto it = levelCache.find(n);
    if (it != levelCache.end()) {
      return it->second;
    }

    // Compute level recursively
    uint32_t maxLevel = 0;
    for (auto operand : n->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
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

  //===--------------------------------------------------------------------===//
  // CIRCT-specific Helper Methods
  //===--------------------------------------------------------------------===//

  /// Check if an operation is a supported logic operation for refactoring
  bool is_logic_op(node n) const {
    if (!n)
      return false;
    return isa<aig::AndInverterOp, mig::MajorityInverterOp>(n);
  }

  /// Get all logic operations in the module
  void foreach_logic_node(std::function<void(node)> fn) const {
    module.walk([&](mlir::Operation *op) {
      if (is_logic_op(op)) {
        fn(op);
      }
    });
  }

  /// Convert mockturtle result back to CIRCT Cut structure
  Cut createCutFromMockturtle(node root, const std::vector<node> &leaves,
                              const std::vector<node> &nodes) const {
    Cut cut;

    // Add operations to cut (nodes in the cone)
    for (auto op : nodes) {
      if (op && is_logic_op(op)) {
        cut.operations.insert(op);
      }
    }

    // Add inputs to cut (leaves/boundary)
    for (auto leafOp : leaves) {
      if (leafOp) {
        // Add all results of leaf operations as cut inputs
        for (auto result : leafOp->getResults()) {
          cut.inputs.insert(result);
        }
      } else {
        // Handle block arguments (primary inputs)
        // This needs to be handled based on the specific traversal context
      }
    }

    return cut;
  }
};

/// Compute reconvergence-driven cuts using mockturtle algorithms
std::optional<Cut> computeReconvergenceDrivenCut(hw::HWModuleOp module, 
                                                 mlir::Operation *root);

#else // !CIRCT_MOCKTURTLE_INTEGRATION_ENABLED

/// Stub implementation when mockturtle integration is disabled
class CIRCTNetworkAdapter {
public:
  explicit CIRCTNetworkAdapter(hw::HWModuleOp mod) {
    (void)mod; // Suppress unused parameter warning
  }
};

/// Fallback implementation that returns empty result
inline std::optional<Cut> computeReconvergenceDrivenCut(hw::HWModuleOp module, 
                                                        mlir::Operation *root) {
  (void)module; (void)root; // Suppress unused parameter warnings
  return std::nullopt;
}

#endif // CIRCT_MOCKTURTLE_INTEGRATION_ENABLED

} // namespace mockturtle_integration
} // namespace synth
} // namespace circt

#endif // LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_ADAPTER_H
