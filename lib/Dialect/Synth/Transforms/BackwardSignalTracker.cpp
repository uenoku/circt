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

#include "circt/Dialect/Comb/CombDialect.h"
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
                               llvm::DenseSet<Object> &visited, int depth = 0);

  // Track forward from a given Object (what it drives)
  void trackForwardFromObject(const Object &obj,
                              igraph::InstancePathCache &pathCache,
                              hw::InstanceGraph &instanceGraph,
                              llvm::DenseSet<Object> &visited, int depth = 0);

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

  // The last element is the signal name, possibly with bit index
  StringRef signalWithBit = pathParts.back();
  StringRef signalName = signalWithBit;
  std::optional<unsigned> bitIndex;

  // Check for bit index like signal_name[9]
  size_t bracketPos = signalWithBit.find('[');
  if (bracketPos != StringRef::npos) {
    signalName = signalWithBit.substr(0, bracketPos);
    size_t endBracket = signalWithBit.find(']', bracketPos);
    if (endBracket != StringRef::npos) {
      StringRef bitStr =
          signalWithBit.substr(bracketPos + 1, endBracket - bracketPos - 1);
      unsigned bit;
      if (!bitStr.getAsInteger(10, bit)) {
        bitIndex = bit;
      }
    }
  }

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
    auto *targetNode =
        pathCache.instanceGraph.lookup(foundInst.getModuleNameAttr().getAttr());
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

  // Create Objects for specified bit(s)
  auto type = targetValue.getType();
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (bitIndex) {
      // Only create object for the specified bit
      if (*bitIndex < intType.getWidth()) {
        results.push_back(Object(instancePath, targetValue, *bitIndex));
      } else {
        llvm::errs() << "Bit index " << *bitIndex
                     << " out of range for signal '" << signalName
                     << "' (width: " << intType.getWidth() << ")\n";
        return false;
      }
    } else {
      // Create objects for all bits
      for (size_t bit = 0; bit < intType.getWidth(); ++bit) {
        results.push_back(Object(instancePath, targetValue, bit));
      }
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
  bool shouldShow = true;
  if (auto op = obj.value.getDefiningOp()) {
    if (isa_and_nonnull<comb::CombDialect, synth::SynthDialect>(
            op->getDialect()))
      shouldShow = false;
  }
  if (shouldShow) {
    std::string kind = "input port";
    if (auto op = obj.value.getDefiningOp()) {
      if (isa<hw::InstanceOp>(op))
        kind = "output port";
      else
        kind = op->getName().getStringRef();
    }
    llvm::outs() << std::string(depth * 2, ' ') << kind << " ";

    // Print instance path
    for (auto inst : obj.instancePath) {
      llvm::outs()
          << "/" << cast<hw::InstanceOp>(inst.getOperation()).getInstanceName();
    }

    // Print value name if available
    if (auto *defOp = obj.value.getDefiningOp()) {
      if (auto instance = dyn_cast<hw::InstanceOp>(defOp)) {
        llvm::outs() << "/" << instance.getInstanceName() << "/"
                     << cast<StringAttr>(
                            instance.getResultNames()[cast<OpResult>(obj.value)
                                                          .getResultNumber()])
                            .getValue();
      } else if (auto nameAttr = defOp->getAttrOfType<StringAttr>("name")) {
        llvm::outs() << "/" << nameAttr.getValue();
      } else if (auto nameAttr = defOp->getAttrOfType<StringAttr>("sym_name")) {
        llvm::outs() << "/" << nameAttr.getValue();
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(obj.value)) {
      // For block arguments (module inputs), get the port name
      if (auto module =
              dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp())) {
        auto inputNames = module.getInputNames();
        auto argNum = blockArg.getArgNumber();
        assert(argNum < inputNames.size());
        llvm::outs() << "/" << cast<StringAttr>(inputNames[argNum]).getValue();
      }
    }

    llvm::outs() << "[" << obj.bitPos << "] (hasEmptyUse="
                 << (obj.value.use_empty() ? "true" : "false") << ") "
                 << obj.value.getLoc() << "\n";
  }

  if (auto op = obj.value.getDefiningOp()) {
    if (isa<seq::FirRegOp>(op))
      return;
  }

  // Find all users of this value - these are operations that use obj.value
  for (auto &use : obj.value.getUses()) {
    Operation *userOp = use.getOwner();

    // llvm::outs() << std::string((depth + 1) * 2, ' ')
    //              << "Used by operation: " << userOp->getName() << "\n";

    // Check if this is an output port being used by an instance
    // If so, we need to pop the instance path and track in the parent module
    if (auto instanceOp = dyn_cast<hw::InstanceOp>(userOp)) {
      // This value is used as an input to an instance
      // We need to push the instance onto the path and track the corresponding
      // port
      auto operandIdx = use.getOperandNumber();
      auto *targetNode =
          instanceGraph.lookup(instanceOp.getModuleNameAttr().getAttr());
      if (targetNode) {
        if (auto targetModule =
                dyn_cast<hw::HWModuleOp>(*targetNode->getModule())) {
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

      // Check if obj.bitPos is within the extracted range [lowBit, lowBit +
      // width)
      if (obj.bitPos >= lowBit && obj.bitPos < lowBit + resultType.getWidth()) {
        // This bit is extracted to result[obj.bitPos - lowBit]
        Object userObj(obj.instancePath, extractOp.getResult(),
                       obj.bitPos - lowBit);
        trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                                depth);
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
                              depth);
      continue;
    }

    // Handle replicate: input repeated multiple times
    if (auto replicateOp = dyn_cast<comb::ReplicateOp>(userOp)) {
      auto inputType = cast<IntegerType>(replicateOp.getInput().getType());
      auto resultType = cast<IntegerType>(replicateOp.getResult().getType());
      unsigned inputWidth = inputType.getWidth();
      unsigned multiple = resultType.getWidth() / inputWidth;

      // obj.bitPos appears at: bitPos, bitPos + inputWidth, bitPos +
      // 2*inputWidth, ...
      for (unsigned i = 0; i < multiple; ++i) {
        Object userObj(obj.instancePath, replicateOp.getResult(),
                       obj.bitPos + i * inputWidth);
        trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                                depth);
      }
      continue;
    }

    // Handle bitwise operations: result[i] depends on operands[i]
    if (isa<synth::aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp>(
            userOp)) {
      auto resultType = cast<IntegerType>(userOp->getResult(0).getType());

      // The same bit position in result depends on this bit
      if (obj.bitPos < resultType.getWidth()) {
        Object userObj(obj.instancePath, userOp->getResult(0), obj.bitPos);
        trackBackwardFromObject(userObj, pathCache, instanceGraph, visited,
                                depth);
      }
      continue;
    }

    if (isa<seq::FirRegOp, hw::WireOp>(userOp)) {
      trackBackwardFromObject(
          Object(obj.instancePath, userOp->getResult(0), obj.bitPos), pathCache,
          instanceGraph, visited, depth);
      continue;
    }

    // For other operations, just report unknown
    llvm::outs() << std::string((depth + 1) * 2, ' ')
                 << "Unknown operation for bit-precise tracking: "
                 << userOp->getName() << "\n";
  }
}

void BackwardSignalTrackerPass::trackForwardFromObject(
    const Object &obj, igraph::InstancePathCache &pathCache,
    hw::InstanceGraph &instanceGraph, llvm::DenseSet<Object> &visited,
    int depth) {
  // Avoid revisiting the same object
  if (!visited.insert(obj).second)
    return;

  // Print the current object in same format as backward tracking
  bool shouldShow = true;
  if (auto op = obj.value.getDefiningOp()) {
    if (isa_and_nonnull<comb::CombDialect, synth::SynthDialect>(
            op->getDialect()))
      shouldShow = false;
  }

  if (shouldShow) {
    std::string kind = "input port";
    if (auto op = obj.value.getDefiningOp()) {
      if (isa<hw::InstanceOp>(op))
        kind = "output port";
      else
        kind = op->getName().getStringRef();
    }
    llvm::outs() << std::string(depth * 2, ' ') << kind << " ";

    // Print instance path
    for (auto inst : obj.instancePath) {
      llvm::outs()
          << "/" << cast<hw::InstanceOp>(inst.getOperation()).getInstanceName();
    }

    // Print value name if available
    if (auto *defOp = obj.value.getDefiningOp()) {
      if (auto instance = dyn_cast<hw::InstanceOp>(defOp)) {
        llvm::outs() << "/" << instance.getInstanceName() << "/"
                     << cast<StringAttr>(
                            instance.getResultNames()[cast<OpResult>(obj.value)
                                                          .getResultNumber()])
                            .getValue();
      } else if (auto nameAttr = defOp->getAttrOfType<StringAttr>("name")) {
        llvm::outs() << "/" << nameAttr.getValue();
      } else if (auto nameAttr = defOp->getAttrOfType<StringAttr>("sym_name")) {
        llvm::outs() << "/" << nameAttr.getValue();
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(obj.value)) {
      // For block arguments (module inputs), get the port name
      if (auto module =
              dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp())) {
        auto inputNames = module.getInputNames();
        auto argNum = blockArg.getArgNumber();
        assert(argNum < inputNames.size());
        llvm::outs() << "/" << cast<StringAttr>(inputNames[argNum]).getValue();
      }
    }

    llvm::outs() << "[" << obj.bitPos << "] (hasEmptyUse="
                 << (obj.value.use_empty() ? "true" : "false") << ") "
                 << obj.value.getLoc() << "\n";
  }

  // Forward tracking: find what this value drives (its operands define it)
  if (auto *defOp = obj.value.getDefiningOp()) {
    // Handle instance outputs - track into the instance
    if (auto instanceOp = dyn_cast<hw::InstanceOp>(defOp)) {
      // Find which output this is
      auto resultNum = cast<OpResult>(obj.value).getResultNumber();

      // Get the target module
      auto *targetNode =
          instanceGraph.lookup(instanceOp.getModuleNameAttr().getAttr());
      if (targetNode) {
        if (auto targetModule =
                dyn_cast<hw::HWModuleOp>(*targetNode->getModule())) {
          // Find the hw.output operation in the target module
          Value outputValue;
          targetModule.walk([&](hw::OutputOp output) {
            outputValue = output.getOperand(resultNum);
            return WalkResult::interrupt();
          });

          if (outputValue) {
            // Push instance onto path
            igraph::InstancePath newPath =
                pathCache.appendInstance(obj.instancePath, instanceOp);
            Object sourceObj(newPath, outputValue, obj.bitPos);
            trackForwardFromObject(sourceObj, pathCache, instanceGraph, visited,
                                   depth + 1);
          }
        }
      }
      return;
    }

    // Handle different operation types for bit-precise tracking
    if (auto extractOp = dyn_cast<comb::ExtractOp>(defOp)) {
      // For extract: result[i] comes from input[lowBit + i]
      unsigned lowBit = extractOp.getLowBit();
      Object sourceObj(obj.instancePath, extractOp.getInput(), lowBit + obj.bitPos);
      trackForwardFromObject(sourceObj, pathCache, instanceGraph, visited, depth + 1);
    } else if (auto concatOp = dyn_cast<comb::ConcatOp>(defOp)) {
      // Find which operand contributes to this bit
      unsigned bitOffset = 0;
      for (int i = concatOp.getNumOperands() - 1; i >= 0; --i) {
        auto opType = cast<IntegerType>(concatOp.getOperand(i).getType());
        unsigned opWidth = opType.getWidth();
        if (obj.bitPos >= bitOffset && obj.bitPos < bitOffset + opWidth) {
          Object sourceObj(obj.instancePath, concatOp.getOperand(i),
                           obj.bitPos - bitOffset);
          trackForwardFromObject(sourceObj, pathCache, instanceGraph, visited, depth + 1);
          break;
        }
        bitOffset += opWidth;
      }
    } else if (auto replicateOp = dyn_cast<comb::ReplicateOp>(defOp)) {
      auto inputType = cast<IntegerType>(replicateOp.getInput().getType());
      unsigned inputWidth = inputType.getWidth();
      unsigned sourceBit = obj.bitPos % inputWidth;
      Object sourceObj(obj.instancePath, replicateOp.getInput(), sourceBit);
      trackForwardFromObject(sourceObj, pathCache, instanceGraph, visited, depth + 1);
    } else if (auto firregOp = dyn_cast<seq::FirRegOp>(defOp)) {
      // FirReg - track through getNext() only
      Value nextVal = firregOp.getNext();
      if (auto intType = dyn_cast<IntegerType>(nextVal.getType())) {
        if (obj.bitPos < intType.getWidth()) {
          Object sourceObj(obj.instancePath, nextVal, obj.bitPos);
          trackForwardFromObject(sourceObj, pathCache, instanceGraph, visited, depth + 1);
        }
      }
    } else if (auto compregOp = dyn_cast<seq::CompRegOp>(defOp)) {
      // CompReg - track through getInput() only
      Value inputVal = compregOp.getInput();
      if (auto intType = dyn_cast<IntegerType>(inputVal.getType())) {
        if (obj.bitPos < intType.getWidth()) {
          Object sourceObj(obj.instancePath, inputVal, obj.bitPos);
          trackForwardFromObject(sourceObj, pathCache, instanceGraph, visited, depth + 1);
        }
      }
    } else if (isa<synth::aig::AndInverterOp>(defOp)) {
      // AIG bitwise - track through all operands with same bit position
      for (auto operand : defOp->getOperands()) {
        if (auto intType = dyn_cast<IntegerType>(operand.getType())) {
          if (obj.bitPos < intType.getWidth()) {
            Object sourceObj(obj.instancePath, operand, obj.bitPos);
            trackForwardFromObject(sourceObj, pathCache, instanceGraph, visited, depth + 1);
          }
        }
      }
    } else {
      // Unknown operation - just report it
      llvm::outs() << std::string((depth + 1) * 2, ' ')
                   << "Unknown operation for bit-precise tracking: "
                   << defOp->getName() << "\n";
    }
  } else if (auto blockArg = dyn_cast<BlockArgument>(obj.value)) {
    // For block arguments (input ports), track backward to find the driving
    // instance
    auto module = dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp());
    if (!module)
      return;

    auto argNum = blockArg.getArgNumber();

    // Pop the instance path to get to parent
    if (obj.instancePath.empty())
      return; // Already at top level, nothing drives this

    igraph::InstancePath parentPath = obj.instancePath.dropBack();
    auto lastInst =
        cast<hw::InstanceOp>(obj.instancePath.leaf().getOperation());

    // The input is driven by the corresponding operand of the instance
    Value drivingValue = lastInst.getOperand(argNum);
    Object sourceObj(parentPath, drivingValue, obj.bitPos);
    trackForwardFromObject(sourceObj, pathCache, instanceGraph, visited,
                           depth + 1);
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
    moduleOp.emitError("top module '")
        << topModuleName << "' is not an HWModuleOp";
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
  llvm::outs() << "=== BACKWARD TRACKING (Users) ===\n";
  llvm::DenseSet<Object> visitedBackward;
  for (const auto &target : targetObjects) {
    llvm::outs() << "Tracking from: ";
    target.print(llvm::outs());
    llvm::outs() << "\n";

    trackBackwardFromObject(target, pathCache, instanceGraph, visitedBackward,
                            1);
    llvm::outs() << "\n";
  }

  llvm::outs() << "Total unique users found: " << visitedBackward.size()
               << "\n\n";

  // Track forward from each target
  llvm::outs() << "=== FORWARD TRACKING (Drivers) ===\n";
  llvm::DenseSet<Object> visitedForward;
  for (const auto &target : targetObjects) {
    llvm::outs() << "Tracking from: ";
    target.print(llvm::outs());
    llvm::outs() << "\n";

    trackForwardFromObject(target, pathCache, instanceGraph, visitedForward, 1);
    llvm::outs() << "\n";
  }

  llvm::outs() << "Total unique drivers found: " << visitedForward.size()
               << "\n";
}
