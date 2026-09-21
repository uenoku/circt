//===- TimingPressureAnalysis.cpp - Timing pressure analysis -------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/TimingPressureAnalysis.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace {

bool isLibModule(hw::HWModuleOp module) {
  return module->hasAttr("synth.liberty.cell") ||
         module->hasAttr("synth.dont_use") ||
         module->hasAttr("synth.mapping_cost") ||
         module->hasAttr("synth.supergate_candidate") ||
         module.getModuleName().starts_with("__supergate_");
}

bool hasFlop(hw::HWModuleOp module) {
  if (!module.getBodyBlock())
    return false;
  return module.getBodyBlock()
      ->walk([](Operation *op) {
        if (isa<synth::FFOp>(op))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// Structural delay model (ranking-only levels, see header).
// -1 = unsupported boundary (unknown).
int opDelay(Operation *op) {
  if (isa<synth::TimingProbeOp>(op))
    return 0;
  if (op->hasTrait<OpTrait::ConstantLike>())
    return 0;
  if (isa<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp, hw::ArrayGetOp,
          hw::ArraySliceOp, hw::ArrayCreateOp, hw::ArrayConcatOp>(op))
    return 0;
  if (isa<comb::CombDialect>(op->getDialect()))
    return 1;
  if (isa<datapath::DatapathDialect>(op->getDialect()))
    return 1;
  if (isa<synth::SynthDialect>(op->getDialect())) {
    if (isa<synth::FFOp, synth::LatchOp, synth::ClockGateOp>(op))
      return -1;
    return 1;
  }
  return -1;
}

void splitPorts(hw::HWModuleOp module, SmallVector<hw::PortInfo> &inputs,
                SmallVector<hw::PortInfo> &outputs) {
  for (const auto &port : module.getPortList()) {
    if (port.isInput())
      inputs.push_back(port);
    else if (port.isOutput())
      outputs.push_back(port);
  }
}

// Implementation over the per-module tables declared in the header.
class PressureBuilder {
public:
  PressureBuilder(
      DenseMap<Operation *, TimingPressureAnalysis::ModuleTable> &tables,
      unsigned &numUnknownBoundaries, unsigned &numHierarchicalEdges,
      unsigned &numSharedContexts)
      : tables(tables), numUnknownBoundaries(numUnknownBoundaries),
        numHierarchicalEdges(numHierarchicalEdges),
        numSharedContexts(numSharedContexts) {}

  DenseMap<Operation *, TimingPressureAnalysis::ModuleTable> &tables;
  unsigned &numUnknownBoundaries;
  unsigned &numHierarchicalEdges;
  unsigned &numSharedContexts;
  DenseMap<Operation *, char> flopMemo;

  bool moduleHasFlop(hw::HWModuleOp module) {
    auto it = flopMemo.find(module);
    if (it != flopMemo.end())
      return it->second;
    bool result = hasFlop(module);
    flopMemo[module] = result;
    return result;
  }

  hw::HWModuleOp resolveChild(ModuleOp top, hw::InstanceOp inst) {
    auto *target = top.lookupSymbol(inst.getModuleName());
    auto child = dyn_cast_or_null<hw::HWModuleOp>(target);
    if (!child || !child.getBodyBlock() || isLibModule(child))
      return hw::HWModuleOp();
    return child;
  }

  // One forward arrival sweep over a module; existing entries are seeds
  // (max-monotone, never reset). Returns true on any change.
  bool forwardSweep(hw::HWModuleOp module) {
    ModuleOp top = module->getParentOfType<ModuleOp>();
    auto tableIt = tables.find(module);
    if (tableIt == tables.end())
      return false;
    auto &table = tableIt->second;
    Block *body = module.getBodyBlock();
    bool changed = false;
    for (BlockArgument arg : body->getArguments()) {
      if (!table.arrival.contains(arg)) {
        table.arrival[arg] = 0;
        changed = true;
      }
    }
    for (auto &op : body->getOperations()) {
      if (isa<hw::OutputOp>(op))
        continue;
      if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
        auto *target = top.lookupSymbol(inst.getModuleName());
        auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
        if (targetModule && moduleHasFlop(targetModule)) {
          SmallVector<hw::PortInfo> inputs, outputs;
          splitPorts(targetModule, inputs, outputs);
          for (unsigned i = 0; i < outputs.size() && i < inst.getNumResults();
               ++i) {
            // Flop outputs launch new timing paths; some cells expose
            // only the inverted output (e.g. ASAP7 DFFHQNx1.*QN).
            StringRef portName = outputs[i].name.getValue();
            if ((portName == "Q" || portName == "QN") &&
                !table.arrival.contains(inst.getResult(i))) {
              table.arrival[inst.getResult(i)] = 0;
              changed = true;
            }
          }
        }
        continue;
      }
      int delay = opDelay(&op);
      if (delay < 0) {
        if (op.getNumResults() > 0)
          ++numUnknownBoundaries;
        continue;
      }
      if (op.getNumOperands() == 0) {
        for (Value result : op.getResults()) {
          if (!table.arrival.contains(result)) {
            table.arrival[result] = (int64_t)delay;
            changed = true;
          }
        }
        continue;
      }
      std::optional<int64_t> latest;
      for (Value operand : op.getOperands()) {
        auto it = table.arrival.find(operand);
        if (it == table.arrival.end()) {
          latest.reset();
          break;
        }
        latest = !latest ? it->second : std::max(*latest, it->second);
      }
      if (!latest)
        continue;
      for (Value result : op.getResults()) {
        int64_t arrival = *latest + delay;
        auto it = table.arrival.find(result);
        if (it == table.arrival.end() || arrival > it->second) {
          table.arrival[result] = arrival;
          changed = true;
        }
      }
    }
    return changed;
  }

  // Reseed probe inputs from current arrivals (min with existing).
  void seedProbes(hw::HWModuleOp module) {
    auto tableIt = tables.find(module);
    if (tableIt == tables.end())
      return;
    auto &table = tableIt->second;
    for (auto &op : module.getBodyBlock()->getOperations()) {
      auto probe = dyn_cast<synth::TimingProbeOp>(op);
      if (!probe)
        continue;
      auto it = table.arrival.find(probe.getInput());
      if (it == table.arrival.end())
        continue;
      auto reqIt = table.required.find(probe.getInput());
      if (reqIt == table.required.end() || it->second < reqIt->second)
        table.required[probe.getInput()] = it->second;
    }
  }

  // One backward required-time sweep; returns true on any change.
  bool backwardSweep(hw::HWModuleOp module) {
    auto tableIt = tables.find(module);
    if (tableIt == tables.end())
      return false;
    auto &table = tableIt->second;
    Block *body = module.getBodyBlock();
    bool changed = false;
    for (auto it = body->getOperations().rbegin(),
              e = body->getOperations().rend();
         it != e; ++it) {
      Operation *op = &*it;
      if (isa<hw::OutputOp, hw::InstanceOp>(op))
        continue;
      int delay = opDelay(op);
      if (delay < 0)
        continue;
      std::optional<int64_t> tightest;
      for (Value result : op->getResults()) {
        auto reqIt = table.required.find(result);
        if (reqIt == table.required.end())
          continue;
        tightest =
            !tightest ? reqIt->second : std::min(*tightest, reqIt->second);
      }
      if (!tightest)
        continue;
      for (Value operand : op->getOperands()) {
        int64_t required = *tightest - delay;
        auto reqIt = table.required.find(operand);
        if (reqIt == table.required.end() || required < reqIt->second) {
          table.required[operand] = required;
          changed = true;
        }
      }
    }
    return changed;
  }

  // Backward DFS from `root` within one module, attributing `probe`.
  // Returns true when the reachable set grew.
  bool traceCone(hw::HWModuleOp module, Value root,
                 synth::TimingProbeOp probe) {
    auto tableIt = tables.find(module);
    if (tableIt == tables.end())
      return false;
    auto &table = tableIt->second;
    bool grown = false;
    SmallVector<Value> worklist{root};
    DenseSet<Value> visited;
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;
      if (table.reaches.insert(current).second)
        grown = true;
      auto &list = table.probes[current];
      if (!llvm::is_contained(list, probe)) {
        list.push_back(probe);
        grown = true;
      }
      Operation *defOp = current.getDefiningOp();
      if (!defOp || isa<hw::InstanceOp>(defOp))
        continue; // Local boundary; hierarchy links cross separately.
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
    return grown;
  }
};

struct InstanceEdge {
  hw::HWModuleOp parent;
  hw::InstanceOp inst;
  hw::HWModuleOp child;
};

// Hierarchical summary propagation over the design instance DAG.
// Conservative shared-module policy: arrivals take the maximum over
// instance contexts, requirements the minimum. Instance cycles are
// detected upfront and left at local tables (unknown across the cycle).
// Runs reachability links, then forward arrivals, then backward
// requirements; each phase runs to a deterministic bounded fixed-point.
void propagateHierarchical(
    ModuleOp top, SmallVector<hw::HWModuleOp> &designModules,
    DenseMap<Operation *, TimingPressureAnalysis::ModuleTable> &tables,
    PressureBuilder &builder, unsigned &numHierarchicalEdges,
    unsigned &numSharedContexts) {
  // Collect instance edges in deterministic (source) order.
  DenseMap<Operation *, SmallVector<Operation *>> instanceGraph;
  SmallVector<InstanceEdge> edges;
  DenseMap<Operation *, unsigned> instanceCount;
  for (auto parent : designModules) {
    for (auto &op : parent.getBodyBlock()->getOperations()) {
      auto inst = dyn_cast<hw::InstanceOp>(op);
      if (!inst)
        continue;
      auto child = builder.resolveChild(top, inst);
      if (!child)
        continue;
      // Flop instances are launches/endpoints, not hierarchy.
      if (builder.moduleHasFlop(child))
        continue;
      edges.push_back({parent, inst, child});
      instanceGraph[parent].push_back(child);
      ++instanceCount[child];
    }
  }
  if (edges.empty())
    return;
  numHierarchicalEdges = (unsigned)edges.size();
  for (const auto &[child, count] : instanceCount)
    if (count > 1)
      ++numSharedContexts;

  // Cycle detection (iterative DFS over modules in source order).
  DenseSet<Operation *> cyclic;
  {
    DenseMap<Operation *, char> color; // 0=unvisited 1=open 2=done
    SmallVector<std::pair<Operation *, size_t>> stack;
    for (auto root : designModules) {
      if (color.lookup(root))
        continue;
      stack.push_back({root, 0});
      color[root] = 1;
      while (!stack.empty()) {
        auto &[node, nextIdx] = stack.back();
        auto childIt = instanceGraph.find(node);
        ArrayRef<Operation *> children =
            childIt == instanceGraph.end()
                ? ArrayRef<Operation *>()
                : ArrayRef<Operation *>(childIt->second);
        if (nextIdx >= children.size()) {
          color[node] = 2;
          stack.pop_back();
          continue;
        }
        Operation *child = children[nextIdx++];
        char childColor = color.lookup(child);
        if (childColor == 1) {
          // Back edge: everything on the stack is cyclic.
          bool inCycle = false;
          for (const auto &[stackNode, idx] : stack) {
            (void)idx;
            if (stackNode == child)
              inCycle = true;
            if (inCycle)
              cyclic.insert(stackNode);
          }
        } else if (childColor == 0) {
          color[child] = 1;
          stack.push_back({child, 0});
        }
      }
    }
  }
  auto edgeUsable = [&](const InstanceEdge &edge) {
    return !cyclic.contains(edge.parent) && !cyclic.contains(edge.child);
  };

  auto childOutput = [](hw::HWModuleOp child, unsigned portIdx) -> Value {
    Operation *terminator = child.getBodyBlock()->getTerminator();
    if (!terminator || portIdx >= terminator->getNumOperands())
      return Value();
    return terminator->getOperand(portIdx);
  };

  unsigned bound = 4 * (unsigned)designModules.size() + 8;

  // Phase 1: reachability links.
  {
    bool changed = true;
    unsigned iter = 0;
    while (changed && iter++ < bound) {
      changed = false;
      for (auto &edge : edges) {
        if (!edgeUsable(edge))
          continue;
        auto parentIt = tables.find(edge.parent);
        auto childIt = tables.find(edge.child);
        if (parentIt == tables.end() || childIt == tables.end())
          continue;
        // Up: parent instance result reaches => child output reaches.
        for (unsigned j = 0; j < edge.inst.getNumResults(); ++j) {
          Value result = edge.inst.getResult(j);
          if (!parentIt->second.reaches.contains(result))
            continue;
          Value output = childOutput(edge.child, j);
          if (!output)
            continue;
          for (auto probe : parentIt->second.probes.lookup(result))
            changed |= builder.traceCone(edge.child, output, probe);
        }
        // Down: child input reaches => parent operand reaches.
        Block *childBody = edge.child.getBodyBlock();
        for (unsigned i = 0;
             i < edge.inst.getNumOperands() && i < childBody->getNumArguments();
             ++i) {
          Value arg = childBody->getArgument(i);
          if (!childIt->second.reaches.contains(arg))
            continue;
          Value operand = edge.inst.getOperand(i);
          for (auto probe : childIt->second.probes.lookup(arg))
            changed |= builder.traceCone(edge.parent, operand, probe);
        }
      }
    }
  }

  // Phase 2: forward arrival links (maximum over contexts).
  {
    bool changed = true;
    unsigned iter = 0;
    while (changed && iter++ < bound) {
      changed = false;
      for (auto &edge : edges) {
        if (!edgeUsable(edge))
          continue;
        auto parentIt = tables.find(edge.parent);
        auto childIt = tables.find(edge.child);
        if (parentIt == tables.end() || childIt == tables.end())
          continue;
        Block *childBody = edge.child.getBodyBlock();
        for (unsigned i = 0;
             i < edge.inst.getNumOperands() && i < childBody->getNumArguments();
             ++i) {
          auto operandIt =
              parentIt->second.arrival.find(edge.inst.getOperand(i));
          if (operandIt == parentIt->second.arrival.end())
            continue;
          Value arg = childBody->getArgument(i);
          auto argIt = childIt->second.arrival.find(arg);
          if (argIt == childIt->second.arrival.end() ||
              operandIt->second > argIt->second) {
            childIt->second.arrival[arg] = operandIt->second;
            changed = true;
          }
        }
        for (unsigned j = 0; j < edge.inst.getNumResults(); ++j) {
          Value output = childOutput(edge.child, j);
          if (!output)
            continue;
          auto outputIt = childIt->second.arrival.find(output);
          if (outputIt == childIt->second.arrival.end())
            continue;
          Value result = edge.inst.getResult(j);
          auto resultIt = parentIt->second.arrival.find(result);
          if (resultIt == parentIt->second.arrival.end() ||
              outputIt->second != resultIt->second) {
            parentIt->second.arrival[result] = outputIt->second;
            changed = true;
          }
        }
        if (changed) {
          while (builder.forwardSweep(edge.child)) {
          }
          while (builder.forwardSweep(edge.parent)) {
          }
        }
      }
      // Reseed probes after arrivals settle each pass (arrival grows).
      for (auto module : designModules)
        builder.seedProbes(module);
    }
  }

  // Phase 3: backward requirement links (minimum over contexts).
  {
    for (auto module : designModules)
      builder.seedProbes(module);
    bool changed = true;
    unsigned iter = 0;
    while (changed && iter++ < bound) {
      changed = false;
      for (auto &edge : edges) {
        if (!edgeUsable(edge))
          continue;
        auto parentIt = tables.find(edge.parent);
        auto childIt = tables.find(edge.child);
        if (parentIt == tables.end() || childIt == tables.end())
          continue;
        for (unsigned j = 0; j < edge.inst.getNumResults(); ++j) {
          auto resultIt =
              parentIt->second.required.find(edge.inst.getResult(j));
          if (resultIt == parentIt->second.required.end())
            continue;
          Value output = childOutput(edge.child, j);
          if (!output)
            continue;
          auto outputIt = childIt->second.required.find(output);
          if (outputIt == childIt->second.required.end() ||
              resultIt->second < outputIt->second) {
            childIt->second.required[output] = resultIt->second;
            changed = true;
          }
        }
        Block *childBody = edge.child.getBodyBlock();
        for (unsigned i = 0;
             i < edge.inst.getNumOperands() && i < childBody->getNumArguments();
             ++i) {
          Value arg = childBody->getArgument(i);
          auto argIt = childIt->second.required.find(arg);
          if (argIt == childIt->second.required.end())
            continue;
          Value operand = edge.inst.getOperand(i);
          auto operandIt = parentIt->second.required.find(operand);
          if (operandIt == parentIt->second.required.end() ||
              argIt->second < operandIt->second) {
            parentIt->second.required[operand] = argIt->second;
            changed = true;
          }
        }
        if (changed) {
          builder.seedProbes(edge.child);
          while (builder.backwardSweep(edge.child)) {
          }
          builder.seedProbes(edge.parent);
          while (builder.backwardSweep(edge.parent)) {
          }
        }
      }
    }
  }
}

} // namespace

TimingPressureAnalysis::TimingPressureAnalysis(Operation *op) : scope(op) {
  analyze();
}

TimingPoint TimingPressureAnalysis::getPoint(Value value) const {
  TimingPoint point;
  Operation *parent = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    parent = blockArg.getOwner()->getParentOp();
  } else if (Operation *defOp = value.getDefiningOp()) {
    parent = defOp->getParentOfType<hw::HWModuleOp>();
  }
  auto it = tables.find(parent);
  if (it == tables.end())
    return point;
  const ModuleTable &table = it->second;
  auto arrIt = table.arrival.find(value);
  if (arrIt != table.arrival.end())
    point.arrivalLevels = arrIt->second;
  auto reqIt = table.required.find(value);
  if (reqIt != table.required.end())
    point.requiredLevels = reqIt->second;
  if (point.arrivalLevels && point.requiredLevels) {
    point.slackLevels = *point.requiredLevels - *point.arrivalLevels;
    point.confidence = TimingConfidence::StructuralEstimate;
  } else if (table.reaches.contains(value)) {
    point.confidence = TimingConfidence::StructuralEstimate;
  }
  return point;
}

bool TimingPressureAnalysis::reachesProbe(Value value) const {
  Operation *parent = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    parent = blockArg.getOwner()->getParentOp();
  } else if (Operation *defOp = value.getDefiningOp()) {
    parent = defOp->getParentOfType<hw::HWModuleOp>();
  }
  auto it = tables.find(parent);
  return it != tables.end() && it->second.reaches.contains(value);
}

SmallVector<TimingProbeOp>
TimingPressureAnalysis::getConstrainingProbes(Value value) const {
  Operation *parent = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    parent = blockArg.getOwner()->getParentOp();
  } else if (Operation *defOp = value.getDefiningOp()) {
    parent = defOp->getParentOfType<hw::HWModuleOp>();
  }
  auto it = tables.find(parent);
  if (it == tables.end())
    return {};
  auto probeIt = it->second.probes.find(value);
  if (probeIt == it->second.probes.end())
    return {};
  return probeIt->second;
}

bool TimingPressureAnalysis::isPressured(Value value,
                                         int64_t marginLevels) const {
  TimingPoint point = getPoint(value);
  if (!point.slackLevels || !reachesProbe(value))
    return false;
  return *point.slackLevels >= -marginLevels;
}

std::optional<int64_t>
TimingPressureAnalysis::getWorstSlackLevels(Operation *op) const {
  std::optional<int64_t> worst;
  for (Value result : op->getResults()) {
    TimingPoint point = getPoint(result);
    if (!point.slackLevels)
      continue;
    if (!worst || *point.slackLevels < *worst)
      worst = *point.slackLevels;
  }
  return worst;
}

void TimingPressureAnalysis::analyze() {
  auto moduleOp = dyn_cast<ModuleOp>(scope);
  SmallVector<hw::HWModuleOp> designModules;
  if (moduleOp) {
    for (auto module : moduleOp.getOps<hw::HWModuleOp>()) {
      if (isLibModule(module) || !module.getBodyBlock())
        continue;
      designModules.push_back(module);
      tables.try_emplace(module);
    }
  } else if (auto module = dyn_cast<hw::HWModuleOp>(scope)) {
    if (module.getBodyBlock() && !isLibModule(module)) {
      designModules.push_back(module);
      tables.try_emplace(module);
    }
  }

  PressureBuilder builder(tables, numUnknownBoundaries, numHierarchicalEdges,
                          numSharedContexts);
  // Local fixed-points.
  for (auto module : designModules) {
    while (builder.forwardSweep(module)) {
    }
    builder.seedProbes(module);
    while (builder.backwardSweep(module)) {
    }
    // Local reachability from each probe (preorder => sorted probe lists).
    for (auto &op : module.getBodyBlock()->getOperations())
      if (auto probe = dyn_cast<synth::TimingProbeOp>(op))
        builder.traceCone(module, probe.getInput(), probe);
  }
  if (moduleOp && designModules.size() > 1)
    propagateHierarchical(moduleOp, designModules, tables, builder,
                          numHierarchicalEdges, numSharedContexts);

  // Statistics over the final tables.
  for (const auto &[module, table] : tables) {
    (void)module;
    numWithArrival += (unsigned)table.arrival.size();
    numWithRequired += (unsigned)table.required.size();
    for (const auto &[value, probes] : table.probes) {
      (void)probes;
      if (table.arrival.contains(value) && table.required.contains(value))
        ++numWithSlack;
    }
  }
}
