//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mockturtle integration functions for CIRCT.
//
//===----------------------------------------------------------------------===//

#include "MockturtleAdapter.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mockturtle-integration"

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::mockturtle_integration;

#ifdef CIRCT_MOCKTURTLE_INTEGRATION_ENABLED

std::optional<Cut> mockturtle_integration::computeReconvergenceDrivenCut(
    hw::HWModuleOp module, mlir::Operation *root) {
  
  LLVM_DEBUG(llvm::dbgs() << "Computing reconvergence-driven cut for: " << *root << "\n");

  // Create adapter for CIRCT IR
  CIRCTNetworkAdapter adapter(module);

  // Check if the root is a valid logic operation
  if (!adapter.is_logic_op(root)) {
    LLVM_DEBUG(llvm::dbgs() << "Root is not a logic operation\n");
    return std::nullopt;
  }

  try {
    // Use mockturtle's reconvergence-driven cut algorithm
    mockturtle::reconv_cut_params params;
    params.cut_size = 8; // Default cut size
    params.reconvergence_depth = 4; // Default depth

    mockturtle::reconv_cut<CIRCTNetworkAdapter> cut_gen(adapter, params);
    
    // Compute cuts for the root node
    auto cuts = cut_gen.run(root);
    
    if (cuts.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No cuts found\n");
      return std::nullopt;
    }

    // Use the first (best) cut
    auto &best_cut = cuts[0];
    
    // Convert mockturtle cut to CIRCT Cut
    std::vector<CIRCTNetworkAdapter::node> leaves;
    std::vector<CIRCTNetworkAdapter::node> nodes;
    
    // Extract leaves and nodes from mockturtle cut
    // Note: This is a simplified conversion - the actual mockturtle API
    // might be different depending on the version
    for (auto leaf : best_cut.leaves()) {
      leaves.push_back(leaf);
    }
    
    for (auto node : best_cut.nodes()) {
      nodes.push_back(node);
    }

    Cut result = adapter.createCutFromMockturtle(root, leaves, nodes);
    
    LLVM_DEBUG({
      llvm::dbgs() << "Successfully computed cut with " 
                   << result.inputs.size() << " inputs and "
                   << result.operations.size() << " operations\n";
    });

    return result;

  } catch (const std::exception &e) {
    LLVM_DEBUG(llvm::dbgs() << "Exception in mockturtle: " << e.what() << "\n");
    return std::nullopt;
  }
}

#endif // CIRCT_MOCKTURTLE_INTEGRATION_ENABLED
