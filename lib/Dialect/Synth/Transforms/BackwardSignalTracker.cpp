//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs backward signal tracking from specified objects to
// identify all users in the dataflow graph. It uses bit-sensitive Object
// tracking from LongestPathAnalysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "synth-backward-signal-tracker"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_BACKWARDSIGNALTRACKER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

namespace {

struct BackwardSignalTrackerPass
    : public impl::BackwardSignalTrackerBase<BackwardSignalTrackerPass> {
  using impl::BackwardSignalTrackerBase<
      BackwardSignalTrackerPass>::BackwardSignalTrackerBase;

  void runOnOperation() override;

private:
  // Track users backward from a given Object
  void trackBackwardFromObject(const Object &obj,
                                igraph::InstancePathCache &pathCache,
                                hw::InstanceGraph &instanceGraph,
                                llvm::DenseSet<Object> &visited,
                                int depth = 0);

  // Find all Objects matching the target signal paths (with instance hierarchy)
  void findTargetObjects(hw::HWModuleOp topModule,
                         igraph::InstancePathCache &pathCache,
                         llvm::SmallVectorImpl<Object> &targets);

  // Parse a target path like "inst1/inst2/signal_name" and find the object
  bool findObjectFromPath(StringRef pathStr, hw::HWModuleOp topModule,
                          igraph::InstancePathCache &pathCache,
                          llvm::SmallVectorImpl<Object> &results);

  // Find a value (port or register) by name in a module
  Value findValueInModule(hw::HWModuleOp module, StringRef name);
};

} // namespace

Value BackwardSignalTrackerPass::findValueInModule(hw::HWModuleOp module,
                                                   StringRef name) {
  // Check module inputs
  for (auto [idx, argName] : llvm::enumerate(module.getInputNames())) {
    if (cast<StringAttr>(argName).getValue() == name)
      return module.getBodyBlock()->getArgument(idx);
  }

  // Check module outputs
  for (auto [idx, argName] : llvm::enumerate(module.getOutputNames())) {
    if (cast<StringAttr>(argName).getValue() == name) {
      // For outputs, we need to find the value being output
      // Walk to find hw.output and get the corresponding operand
      Value result;
      module.walk([&](hw::OutputOp output) {
        result = output.getOperand(idx);
        return WalkResult::interrupt();
      });
      return result;
    }
  }

  // Check for registers and wires with matching names
  Value result;
  module.walk([&](Operation *op) {
    // Check for registers
    if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
        if (nameAttr.getValue() == name) {
          result = reg.getResult();
          return WalkResult::interrupt();
        }
      }
    }
    // Check for any operation with a name attribute
    if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
      if (nameAttr.getValue() == name && op->getNumResults() > 0) {
        result = op->getResult(0);
        return WalkResult::interrupt();
      }
    }
    // Check sym_name
    if (auto nameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
      if (nameAttr.getValue() == name && op->getNumResults() > 0) {
        result = op->getResult(0);
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return result;
}

bool BackwardSignalTrackerPass::findObjectFromPath(
    StringRef pathStr, hw::HWModuleOp topModule,
    igraph::InstancePathCache &pathCache,
    llvm::SmallVectorImpl<Object> &results) {

  // Split the path by '/'
  SmallVector<StringRef> pathParts;
  pathStr.split(pathParts, '/');

  if (pathParts.empty())
    return false;

  // The last element is the signal name
  StringRef signalName = pathParts.back();

  // Navigate through instance hierarchy
  hw::HWModuleOp currentModule = topModule;
  igraph::InstancePath instancePath;
  SmallVector<igraph::InstanceOpInterface> instances;

  // Walk through instance names (all but the last element)
  for (size_t i = 0; i < pathParts.size() - 1; ++i) {
    StringRef instName = pathParts[i];

    // Find the instance in the current module
    hw::InstanceOp foundInst;
    currentModule.walk([&](hw::InstanceOp inst) {
      if (inst.getInstanceName() == instName) {
        foundInst = inst;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!foundInst) {
      llvm::errs() << "Could not find instance '" << instName << "' in module '"
                   << currentModule.getModuleName() << "'\n";
      return false;
    }

    // Append to instance path immediately
    instancePath = pathCache.appendInstance(instancePath, foundInst);
    instances.push_back(foundInst);

    // Get the referenced module using the instance graph
    // Look up the module by name (convert FlatSymbolRefAttr to StringAttr)
    auto *targetNode = pathCache.instanceGraph.lookup(
        foundInst.getModuleNameAttr().getAttr());
    if (!targetNode) {
      llvm::errs() << "Could not find module '" << foundInst.getModuleName()
                   << "' in instance graph\n";
      return false;
    }

    auto referencedModule = dyn_cast<hw::HWModuleOp>(*targetNode->getModule());
    if (!referencedModule) {
      llvm::errs() << "Referenced module '" << foundInst.getModuleName()
                   << "' is not an HWModuleOp\n";
      return false;
    }

    currentModule = referencedModule;
  }

  // Now find the signal in the final module
  Value targetValue = findValueInModule(currentModule, signalName);

  if (!targetValue) {
    llvm::errs() << "Could not find signal '" << signalName << "' in module '"
                 << currentModule.getModuleName() << "'\n";
    llvm::errs() << "Available input ports:\n";
    for (auto argName : currentModule.getInputNames()) {
      llvm::errs() << "  - " << cast<StringAttr>(argName).getValue() << "\n";
    }
    llvm::errs() << "Available output ports:\n";
    for (auto argName : currentModule.getOutputNames()) {
      llvm::errs() << "  - " << cast<StringAttr>(argName).getValue() << "\n";
    }
    return false;
  }

  // Create Objects for each bit
  auto type = targetValue.getType();
  if (auto intType = dyn_cast<IntegerType>(type)) {
    for (size_t bit = 0; bit < intType.getWidth(); ++bit) {
      results.push_back(Object(instancePath, targetValue, bit));
    }
  } else {
    // For non-integer types, create a single bit-0 object
    results.push_back(Object(instancePath, targetValue, 0));
  }

  return true;
}

void BackwardSignalTrackerPass::findTargetObjects(
    hw::HWModuleOp topModule, igraph::InstancePathCache &pathCache,
    llvm::SmallVectorImpl<Object> &targets) {

  for (const auto &pathStr : targetSignals) {
    if (!findObjectFromPath(pathStr, topModule, pathCache, targets)) {
      llvm::errs() << "Warning: Could not find target '" << pathStr << "'\n";
    }
  }
}

void BackwardSignalTrackerPass::trackBackwardFromObject(
    const Object &obj, igraph::InstancePathCache &pathCache,
    hw::InstanceGraph &instanceGraph, llvm::DenseSet<Object> &visited,
    int depth) {
  // Avoid revisiting the same object
  if (!visited.insert(obj).second)
    return;

  // Print the current object in Verilog-style path format
  llvm::outs() << std::string(depth * 2, ' ') << "User: $root";

  // Print instance path
  for (auto inst : obj.instancePath) {
    llvm::outs() << "/" << cast<hw::InstanceOp>(inst.getOperation()).getInstanceName();
  }

  // Print value name if available
  if (auto *defOp = obj.value.getDefiningOp()) {
    if (auto nameAttr = defOp->getAttrOfType<StringAttr>("name")) {
      llvm::outs() << "/" << nameAttr.getValue();
    } else if (auto nameAttr = defOp->getAttrOfType<StringAttr>("sym_name")) {
      llvm::outs() << "/" << nameAttr.getValue();
    }
  } else if (auto blockArg = dyn_cast<BlockArgument>(obj.value)) {
    // For block arguments (module inputs), get the port name
    if (auto module = dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp())) {
      auto inputNames = module.getInputNames();
      auto argNum = blockArg.getArgNumber();
      if (argNum < inputNames.size()) {
        llvm::outs() << "/" << cast<StringAttr>(inputNames[argNum]).getValue();
      }
    }
  }

  llvm::outs() << "[" << obj.bitPos << "]\n";

  // Find all users of this value - these are operations that use obj.value
  for (auto &use : obj.value.getUses()) {
    Operation *userOp = use.getOwner();

    llvm::outs() << std::string((depth + 1) * 2, ' ')
                 << "Used by operation: " << userOp->getName() << "\n";

    // Check if this is an output port being used by an instance
    // If so, we need to pop the instance path and track in the parent module
    if (auto instanceOp = dyn_cast<hw::InstanceOp>(userOp)) {
      // This value is used as an input to an instance
      // We need to push the instance onto the path and track the corresponding port
      auto operandIdx = use.getOperandNumber();
      auto *targetNode = instanceGraph.lookup(
          instanceOp.getModuleNameAttr().getAttr());
      if (targetNode) {
        if (auto targetModule = dyn_cast<hw::HWModuleOp>(*targetNode->getModule())) {
          auto *bodyBlock = targetModule.getBodyBlock();
          if (!bodyBlock)
            continue;

          // Get the corresponding argument in the target module
          Value targetArg = bodyBlock->getArgument(operandIdx);

          // Push the instance onto the path
          igraph::InstancePath newPath =
              pathCache.appendInstance(obj.instancePath, instanceOp);

          // Track backward from this argument in the instance's module
          Object newObj(newPath, targetArg, obj.bitPos);
          trackBackwardFromObject(newObj, pathCache, instanceGraph, visited,
                                  depth + 1);
        }
      }
      continue;
    }

    // Check if this is a module output port
    // If the value is a block argument (input port) and used by hw.output,
    // we need to pop the instance path
    if (auto outputOp = dyn_cast<hw::OutputOp>(userOp)) {
      // The value is driven to an output port
      // We need to find instances of this module and track backward
      if (auto currentModule = userOp->getParentOfType<hw::HWModuleOp>()) {
        auto *moduleNode = instanceGraph[currentModule];

        // Find which output index this is
        auto outputIdx = use.getOperandNumber();

        // For each instance of this module, track the corresponding result
        for (auto *instanceRecord : moduleNode->uses()) {
          auto instOp = instanceRecord->getInstance();
          auto inst = dyn_cast_or_null<hw::InstanceOp>(instOp.getOperation());
          if (!inst)
            continue;

          // Pop the instance from the path - must match the leaf
          if (obj.instancePath.empty() || obj.instancePath.leaf() != inst)
            continue;

          igraph::InstancePath parentPath = obj.instancePath.dropBack();

          // Track backward from the instance's result
          Value instResult = inst.getResult(outputIdx);
          Object newObj(parentPath, instResult, obj.bitPos);
          trackBackwardFromObject(newObj, pathCache, instanceGraph, visited,
                                  depth + 1);
        }
      }
      continue;
    }

    // Handle extract: result[i] comes from input[lowBit + i]
    if (auto extractOp = dyn_cast<comb::ExtractOp>(userOp)) {
      unsigned lowBit = extractOp.getLowBit();
      auto resultType = cast<IntegerType>(extractOp.getResult().getType());

      // Check if obj.bitPos is within the extracted range [lowBit, lowBit + width)
      if (obj.bitPos >= lowBit && obj.bitPos < lowBit + resultType.getWidth()) {
        // This bit is extracted to result[obj.bitPos - lowBit]
        Object userObj(obj.instancePath, extractOp.getResult(),
                       obj.bitPos - lowBit);
        trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                                depth + 1);
      }
      continue;
    }

    // Handle concat: MSB to LSB ordering
    if (auto concatOp = dyn_cast<comb::ConcatOp>(userOp)) {
      unsigned operandIdx = use.getOperandNumber();
      unsigned bitOffset = 0;

      // Calculate offset: sum widths of all operands after this one
      for (unsigned i = concatOp.getNumOperands() - 1; i > operandIdx; --i) {
        auto opType = cast<IntegerType>(concatOp.getOperand(i).getType());
        bitOffset += opType.getWidth();
      }

      // obj.bitPos in this operand becomes (bitOffset + obj.bitPos) in result
      Object userObj(obj.instancePath, concatOp.getResult(),
                     bitOffset + obj.bitPos);
      trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                              depth + 1);
      continue;
    }

    // Handle replicate: input repeated multiple times
    if (auto replicateOp = dyn_cast<comb::ReplicateOp>(userOp)) {
      auto inputType = cast<IntegerType>(replicateOp.getInput().getType());
      auto resultType = cast<IntegerType>(replicateOp.getResult().getType());
      unsigned inputWidth = inputType.getWidth();
      unsigned multiple = resultType.getWidth() / inputWidth;

      // obj.bitPos appears at: bitPos, bitPos + inputWidth, bitPos + 2*inputWidth, ...
      for (unsigned i = 0; i < multiple; ++i) {
        Object userObj(obj.instancePath, replicateOp.getResult(),
                       obj.bitPos + i * inputWidth);
        trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                                depth + 1);
      }
      continue;
    }

    // Handle AIG operations - bitwise: result[i] depends on all operands[i]
    if (auto aigOp = dyn_cast<synth::aig::AndInverterOp>(userOp)) {
      auto resultType = cast<IntegerType>(aigOp.getResult().getType());

      // The same bit position in result depends on this bit
      if (obj.bitPos < resultType.getWidth()) {
        Object userObj(obj.instancePath, aigOp.getResult(), obj.bitPos);
        trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                                depth + 1);
      }
      continue;
    }

    // For other operations, track conservatively (all bits to all bits)
    for (auto result : userOp->getResults()) {
      auto type = result.getType();
      if (auto intType = dyn_cast<IntegerType>(type)) {
        for (size_t bit = 0; bit < intType.getWidth(); ++bit) {
          Object userObj(obj.instancePath, result, bit);
          trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                                  depth + 1);
        }
      } else {
        Object userObj(obj.instancePath, result, 0);
        trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                                depth + 1);
      }
    }
  }
}

void BackwardSignalTrackerPass::runOnOperation() {
  auto moduleOp = getOperation();

  llvm::outs() << "=== Backward Signal Tracker ===\n";

  // Top module is mandatory
  if (topModuleName.empty()) {
    moduleOp.emitError("top-module-name option is required");
    return signalPassFailure();
  }

  llvm::outs() << "Target signal paths: ";
  for (const auto &sig : targetSignals)
    llvm::outs() << sig << " ";
  llvm::outs() << "\nTop module: " << topModuleName << "\n\n";

  // Build instance graph
  hw::InstanceGraph instanceGraph(moduleOp);
  igraph::InstancePathCache pathCache(instanceGraph);

  // Find the top module using instance graph lookup
  auto *topNode = instanceGraph.lookup(
      StringAttr::get(moduleOp.getContext(), topModuleName));

  if (!topNode) {
    moduleOp.emitError("could not find top module '") << topModuleName << "'";
    return signalPassFailure();
  }

  auto topModule = dyn_cast<hw::HWModuleOp>(*topNode->getModule());
  if (!topModule) {
    moduleOp.emitError("top module '") << topModuleName << "' is not an HWModuleOp";
    return signalPassFailure();
  }

  // Find target objects
  llvm::SmallVector<Object> targetObjects;
  findTargetObjects(topModule, pathCache, targetObjects);

  if (targetObjects.empty()) {
    llvm::outs() << "No target objects found.\n";
    return;
  }

  llvm::outs() << "Found " << targetObjects.size() << " target object(s)\n\n";

  // Track backward from each target
  llvm::DenseSet<Object> visited;
  for (const auto &target : targetObjects) {
    llvm::outs() << "Tracking from: ";
    target.print(llvm::outs());
    llvm::outs() << "\n";

    trackBackwardFromObject(target, pathCache, instanceGraph, visited, 1);
    llvm::outs() << "\n";
  }

  llvm::outs() << "Total unique users found: " << visited.size() << "\n";
}
