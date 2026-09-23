//===- BufferHighFanout.cpp - Buffer high-fanout nets ----------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Nothing buffers high-fanout nets today; the fifo/ChipTop scoreboards are
// dominated by broadcast paths (a flop fanning out through hierarchy to a
// ~500-wide memory-decoder mux forest). This pass splits nets whose
// per-copy fanout exceeds `--max-fanout`: local gate loads are chunked
// behind one buffer instance per chunk, and output ports with large
// downstream (through-hierarchy) loads get one buffer between driver and
// port. The buffer cell is auto-detected (smallest-area identity cell),
// so the pass stays PDK-agnostic. Nets driving clock pins are never
// touched; constant (tie-cell) drivers need no buffers.
//
// Fanout is counted per flat copy, including loads reached through
// output ports (subject-graph refcounts understate broadcasts by ~100x
// once hierarchy is crossed, so plain refcounts are not used). All
// ordering is file/IR order; no pointer-ordered containers affect
// decisions, keeping the pass deterministic.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthOpInterfaces.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace circt {
namespace synth {
#define GEN_PASS_DECL_BUFFERHIGHFANOUT
#define GEN_PASS_DEF_BUFFERHIGHFANOUT
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

#define DEBUG_TYPE "synth-buffer-high-fanout"

namespace {

// A design value's local (same-module) loads, partitioned for buffering.
struct LocalLoads {
  SmallVector<OpOperand *> dataLoads;
  bool hasClockLoad = false;
  bool hasUnknownTarget = false;
};

static bool isLibModule(hw::HWModuleOp module) {
  return module->hasAttr("synth.liberty.cell") ||
         module->hasAttr("synth.dont_use") ||
         module->hasAttr("synth.mapping_cost") ||
         module->hasAttr("synth.supergate_candidate") ||
         module.getModuleName().starts_with("__supergate_");
}

// Input-port index of an instance operand (operands align with input
// ports in order).
static unsigned inputPortIndex(OpOperand *use) {
  return use->getOperandNumber();
}

struct BufferHighFanoutPass
    : public circt::synth::impl::BufferHighFanoutBase<BufferHighFanoutPass> {
  using BufferHighFanoutBase<BufferHighFanoutPass>::BufferHighFanoutBase;

  void runOnOperation() override {
    ModuleOp top = getOperation();
    if (maxFanout < 1) {
      getOperation()->emitError("max-fanout must be at least 1");
      signalPassFailure();
      return;
    }
    // Timing-guided mode: resolve STA violator launches to values.
    // Only violator nets are buffered (still max-fanout-gated).
    if (!staViolators.empty() && failed(resolveViolators(top))) {
      signalPassFailure();
      return;
    }
    // Probes anywhere also enable guided mode (set union with STA).
    guidedMode = !staViolators.empty();
    if (!guidedMode) {
      for (auto module : top.getOps<hw::HWModuleOp>()) {
        if (isLibModule(module))
          continue;
        bool found = false;
        for (auto &block : module->getRegion(0)) {
          for (auto &op : block)
            if (isa<synth::TimingProbeOp>(op)) {
              found = true;
              break;
            }
          if (found)
            break;
        }
        if (found) {
          guidedMode = true;
          break;
        }
      }
    }
    auto buffer = findBufferCell(top);
    if (!buffer) {
      LLVM_DEBUG(llvm::dbgs() << "no identity buffer cell found\n");
      return markAllAnalysesPreserved();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "buffer cell: " << buffer->getModuleName() << "\n");
    buildChildren();

    // Phase 1: collect probe launches GLOBALLY first, then expand to
    // the forward broadcast distribution (sibling loads through
    // data-ops and ports). Per-module collection would gate early
    // modules by incomplete sets.
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isLibModule(module))
        continue;
      collectProbeLaunches(module);
    }
    collectDistribution(top);
    bool changed = false;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isLibModule(module))
        continue;
      changed |= bufferModule(module, *buffer);
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

  // Static child instances indexed by target module name (file order),
  // so hierarchy queries touch only relevant instances.
  DenseMap<StringAttr, SmallVector<std::pair<hw::HWModuleOp, hw::InstanceOp>>>
      instancesByTarget;
  // STA violator launch nets (timing-guided mode). Empty unless
  // --sta-violators is set, in which case only these nets are buffered.
  DenseSet<Value> violatorLaunches;
  // Guided mode: STA file set, or probes derived anywhere. Only
  // violator launches are actionable (legacy fanout-only otherwise).
  bool guidedMode = false;
  // Memoized stateful (flop/latch/clock-gate) target modules.
  DenseMap<Operation *, char> statefulMemo;
  // Memoized per-copy downstream loads per (module, output-port index),
  // and whether that downstream reaches a clock pin.
  DenseMap<std::pair<Operation *, unsigned>, uint64_t> downMemo;
  DenseMap<std::pair<Operation *, unsigned>, bool> clockDownMemo;
  DenseSet<std::pair<Operation *, unsigned>> downInProgress;

  void buildChildren() {
    for (auto module : getOperation().getOps<hw::HWModuleOp>()) {
      for (auto &block : module->getRegion(0))
        for (auto &op : block)
          if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
            auto *target = getOperation().lookupSymbol(inst.getModuleName());
            auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
            if (!targetModule || isa<hw::HWModuleExternOp>(targetModule))
              continue;
            instancesByTarget[targetModule.getModuleNameAttr()].push_back(
                {module, inst});
          }
    }
  }

  // Output-port list of a module (ports in order); input-port list likewise.
  static void splitPorts(hw::HWModuleOp module,
                         SmallVector<hw::PortInfo> &inputs,
                         SmallVector<hw::PortInfo> &outputs) {
    for (const auto &port : module.getPortList()) {
      if (port.isInput())
        inputs.push_back(port);
      else if (port.isOutput())
        outputs.push_back(port);
    }
  }

  // Parse --sta-violators (JSON list of {startpoint, endpoint,
  // slack_ns}) and resolve each startpoint to a launch value.
  LogicalResult resolveViolators(ModuleOp top) {
    auto file = llvm::MemoryBuffer::getFile(staViolators);
    if (!file) {
      top->emitError("cannot read sta-violators file '") << staViolators << "'";
      return failure();
    }
    auto json = llvm::json::parse(file.get()->getBuffer());
    if (!json) {
      top->emitError("cannot parse sta-violators file '")
          << staViolators << "'";
      return failure();
    }
    auto *list = json->getAsArray();
    if (!list) {
      top->emitError("sta-violators file must contain a JSON list");
      return failure();
    }
    // Collect placements first (endpoint probes mutate operands).
    SmallVector<std::pair<std::string, double>> endpointProbes;
    for (const auto &entry : *list) {
      auto *obj = entry.getAsObject();
      if (!obj)
        continue;
      auto start = obj->getString("startpoint");
      if (start) {
        if (Value launch = resolveLaunch(top, *start)) {
          violatorLaunches.insert(launch);
          ++numViolatorsResolved;
        } else {
          ++numViolatorsUnresolved;
        }
      }
      auto end = obj->getString("endpoint");
      auto required = obj->getNumber("required_ns");
      if (end && required)
        endpointProbes.push_back({end->str(), *required});
    }
    // STA endpoint probes carry the timer's own required times: sparse,
    // exact selectivity for cone membership (no in-flow modeling).
    for (auto &[endpoint, requiredNs] : endpointProbes) {
      if (placeEndpointProbe(top, endpoint, requiredNs))
        ++numEndpointsProbed;
      else
        ++numEndpointsUnresolved;
    }
    return success();
  }

  // Resolve one STA startpoint (hierarchical instance path, or a
  // top-level port name) to its launch value: the Q output of the
  // launch flop, or the matching top-level input block argument.
  // Returns null on any ambiguity or lookup failure (skip conservatively).
  Value resolveLaunch(ModuleOp top, StringRef startpoint) {
    SmallVector<StringRef> segments;
    startpoint.split(segments, '/');
    if (segments.empty())
      return Value();
    if (segments.size() == 1)
      return resolvePortLaunch(top, segments[0]);
    return resolveInstanceLaunch(top, segments);
  }

  // Single-segment startpoint: a top-level input port. Requires the
  // port name to be unique across design modules.
  Value resolvePortLaunch(ModuleOp top, StringRef portName) {
    Value found;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isLibModule(module))
        continue;
      SmallVector<hw::PortInfo> inputs, outputs;
      splitPorts(module, inputs, outputs);
      for (unsigned i = 0; i < inputs.size(); ++i) {
        if (inputs[i].name.getValue() == portName) {
          if (found)
            return Value(); // Ambiguous: skip.
          found = module.getBodyBlock()->getArgument(i);
        }
      }
    }
    return found;
  }

  // Multi-segment startpoint: hierarchical instance path from a root
  // design module down to a flop instance; yields its Q output.
  // Final instance of a hierarchical instance path (null on any
  // ambiguity or lookup failure).
  hw::InstanceOp findInstance(ModuleOp top, ArrayRef<StringRef> segments) {
    // Roots: design modules never instantiated by another design module.
    DenseSet<Operation *> hasDesignParent;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isLibModule(module))
        continue;
      for (auto &block : module->getRegion(0))
        for (auto &op : block)
          if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
            auto *target = top.lookupSymbol(inst.getModuleName());
            auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
            if (targetModule && !isLibModule(targetModule))
              hasDesignParent.insert(target);
          }
    }
    SmallVector<hw::HWModuleOp> worklist;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (!isLibModule(module) && !hasDesignParent.contains(module))
        worklist.push_back(module);
    }
    // Descend one level per segment; duplicate instance names at any
    // level are ambiguous: skip.
    for (size_t level = 0; level < segments.size(); ++level) {
      hw::InstanceOp match;
      for (auto parent : worklist) {
        for (auto &block : parent->getRegion(0))
          for (auto &op : block) {
            auto inst = dyn_cast<hw::InstanceOp>(op);
            if (!inst || inst.getInstanceName() != segments[level])
              continue;
            if (match)
              return hw::InstanceOp(); // Ambiguous: skip.
            match = inst;
          }
      }
      if (!match)
        return hw::InstanceOp();
      if (level + 1 == segments.size())
        return match;
      auto *target = top.lookupSymbol(match.getModuleName());
      auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
      if (!targetModule || isLibModule(targetModule))
        return hw::InstanceOp();
      worklist.clear();
      worklist.push_back(targetModule);
    }
    return hw::InstanceOp();
  }

  Value resolveInstanceLaunch(ModuleOp top, ArrayRef<StringRef> segments) {
    hw::InstanceOp inst = findInstance(top, segments);
    if (!inst)
      return Value();
    return flopQ(top, inst);
  }

  // Place a timing probe carrying STA required time on an endpoint flop
  // D input. Returns false when the endpoint cannot be resolved.
  bool placeEndpointProbe(ModuleOp top, StringRef endpoint, double requiredNs) {
    SmallVector<StringRef> segments;
    endpoint.split(segments, '/');
    if (segments.empty())
      return false;
    hw::InstanceOp inst = findInstance(top, segments);
    if (!inst)
      return false;
    auto *target = top.lookupSymbol(inst.getModuleName());
    auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
    if (!targetModule)
      return false;
    SmallVector<hw::PortInfo> inputs, outputs;
    splitPorts(targetModule, inputs, outputs);
    int dataIdx = -1;
    for (unsigned i = 0; i < inputs.size(); ++i)
      if (inputs[i].name.getValue() == "D")
        dataIdx = (int)i;
    if (dataIdx < 0 || (unsigned)dataIdx >= inst.getNumOperands())
      return false;
    int64_t required = llround(requiredNs * 1000.0);
    if (required <= 0)
      return false;
    OpBuilder builder(inst->getContext());
    builder.setInsertionPoint(inst);
    auto probe = synth::TimingProbeOp::create(
        builder, inst.getLoc(), inst.getOperand((unsigned)dataIdx), required);
    inst.setOperand((unsigned)dataIdx, probe.getResult());
    return true;
  }

  // Q output of a flop instance (output port named "Q", else "QN" for
  // cells exposing only the inverted output); null when the instance has
  // no such output (not a flop launch).
  static Value flopQ(ModuleOp top, hw::InstanceOp inst) {
    auto *target = top.lookupSymbol(inst.getModuleName());
    auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
    if (!targetModule)
      return Value();
    SmallVector<hw::PortInfo> inputs, outputs;
    splitPorts(targetModule, inputs, outputs);
    for (unsigned i = 0; i < outputs.size(); ++i)
      if (outputs[i].name.getValue() == "Q")
        return i < inst.getNumResults() ? inst.getResult(i) : Value();
    for (unsigned i = 0; i < outputs.size(); ++i)
      if (outputs[i].name.getValue() == "QN")
        return i < inst.getNumResults() ? inst.getResult(i) : Value();
    return Value();
  }

  // Data-closure of a net within one module: the net plus results of
  // data-passthrough ops (extract/concat/array/comb) reachable from it.
  // Deterministic (operand-order DFS).
  static void dataClosure(hw::HWModuleOp module, Value root,
                          DenseSet<Value> &closure) {
    SmallVector<Value> worklist{root};
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!closure.insert(current).second)
        continue;
      for (auto &use : current.getUses()) {
        Operation *user = use.getOwner();
        if (isa<hw::OutputOp>(user))
          continue;
        // Instances and stateful ops terminate the closure (their input
        // pins are the loads); data ops are transparent.
        if (isa<hw::InstanceOp>(user))
          continue;
        if (user->hasTrait<mlir::OpTrait::ConstantLike>())
          continue;
        if (isa<FFOp, LatchOp, ClockGateOp>(user))
          continue;
        for (auto result : user->getResults())
          worklist.push_back(result);
      }
    }
    (void)module;
  }

  // Instance-input pin loads on a net's data closure (same module).
  static unsigned dataClosurePinLoads(hw::HWModuleOp module, Value net,
                                      ArrayRef<hw::PortInfo> inputs) {
    DenseSet<Value> closure;
    dataClosure(module, net, closure);
    unsigned count = 0;
    for (Value member : closure) {
      for (auto &use : member.getUses()) {
        Operation *user = use.getOwner();
        if (isa<hw::OutputOp>(user))
          continue;
        if (isa<hw::InstanceOp>(user))
          ++count;
      }
    }
    (void)inputs;
    return count;
  }

  // Whether a terminator operand is in a net's data closure.
  static bool closureDrives(hw::HWModuleOp module, Value net, Value operand) {
    DenseSet<Value> closure;
    dataClosure(module, net, closure);
    return closure.contains(operand);
  }

  // Non-terminator uses of a net inside its module.
  static unsigned gateUses(Value value) {
    unsigned count = 0;
    for (auto &use : value.getUses()) {
      if (isa<hw::OutputOp>(use.getOwner()))
        continue;
      ++count;
    }
    return count;
  }

  // Per-copy downstream loads of an output port: static instances of the
  // module, each contributing its net's local gate uses plus whatever
  // that net reaches through the parent's own ports (recursively).
  // Mult-free: every flat copy sees the same per-copy loads.
  uint64_t downstreamLoads(hw::HWModuleOp module, unsigned outputIdx) {
    auto key = std::make_pair(module.getOperation(), outputIdx);
    auto memo = downMemo.find(key);
    if (memo != downMemo.end())
      return memo->second;
    if (!downInProgress.insert(key).second)
      return 0; // Recursive hierarchy: break the cycle.
    uint64_t total = 0;
    SmallVector<hw::PortInfo> inputs, outputs;
    splitPorts(module, inputs, outputs);
    if (outputIdx < outputs.size()) {
      auto instIt = instancesByTarget.find(module.getModuleNameAttr());
      if (instIt == instancesByTarget.end()) {
        downMemo[key] = 0;
        return 0;
      }
      // Parent port lists cached per module (rebuilt never; file order).
      DenseMap<Operation *,
               std::pair<SmallVector<hw::PortInfo>, SmallVector<hw::PortInfo>>>
          portCache;
      for (auto [parent, inst] : instIt->second) {
        if (outputIdx >= inst.getNumResults())
          continue;
        auto cacheIt = portCache.find(parent.getOperation());
        if (cacheIt == portCache.end()) {
          SmallVector<hw::PortInfo> parentInputs, parentOutputs;
          splitPorts(parent, parentInputs, parentOutputs);
          cacheIt =
              portCache
                  .insert({parent.getOperation(),
                           {std::move(parentInputs), std::move(parentOutputs)}})
                  .first;
        }
        Value net = inst.getResult(outputIdx);
        // Direct pin loads reachable through data operators
        // (extract/concat/array/comb passthrough): same electrical net
        // for load purposes. Gates/instances terminate (their input
        // pins ARE the counted loads); parent ports recurse below.
        total += dataClosurePinLoads(parent, net, cacheIt->second.first);
        // The net (or its data-closure) may drive the parent's own
        // output ports: recurse.
        if (auto outputOp = dyn_cast_or_null<hw::OutputOp>(
                parent.getBodyBlock()->getTerminator())) {
          for (unsigned q = 0; q < outputOp->getNumOperands(); ++q) {
            if (q < cacheIt->second.second.size() &&
                closureDrives(parent, net, outputOp->getOperand(q)))
              total += downstreamLoads(parent, q);
          }
        }
      }
    }
    downInProgress.erase(key);
    downMemo[key] = total;
    return total;
  }

  // Whether anything downstream of an output port is a clock pin
  // (unknown targets count as clock-affected: conservative).
  bool downstreamClock(hw::HWModuleOp module, unsigned outputIdx) {
    auto key = std::make_pair(module.getOperation(), outputIdx);
    auto memo = clockDownMemo.find(key);
    if (memo != clockDownMemo.end())
      return memo->second;
    if (!downInProgress.insert(key).second)
      return false;
    bool found = false;
    SmallVector<hw::PortInfo> inputs, outputs;
    splitPorts(module, inputs, outputs);
    if (outputIdx < outputs.size()) {
      auto instIt = instancesByTarget.find(module.getModuleNameAttr());
      if (instIt == instancesByTarget.end()) {
        clockDownMemo[key] = false;
        return false;
      }
      DenseMap<Operation *, SmallVector<hw::PortInfo>> outCache;
      for (auto [parent, inst] : instIt->second) {
        if (found)
          break;
        if (outputIdx >= inst.getNumResults())
          continue;
        auto cacheIt = outCache.find(parent.getOperation());
        if (cacheIt == outCache.end()) {
          SmallVector<hw::PortInfo> parentInputs, parentOutputs;
          splitPorts(parent, parentInputs, parentOutputs);
          cacheIt =
              outCache.insert({parent.getOperation(), std::move(parentOutputs)})
                  .first;
        }
        Value net = inst.getResult(outputIdx);
        for (auto &use : net.getUses()) {
          Operation *user = use.getOwner();
          if (isa<hw::OutputOp>(user)) {
            unsigned q = use.getOperandNumber();
            if (q < cacheIt->second.size() && downstreamClock(parent, q)) {
              found = true;
              break;
            }
            continue;
          }
          if (isClockPinUse(user, &use)) {
            found = true;
            break;
          }
        }
      }
    }
    downInProgress.erase(key);
    clockDownMemo[key] = found;
    return found;
  }

  // True when an operand drives a liberty clock pin, or its target
  // cannot be resolved (conservative: unknown counts as clock).
  // Non-instance users (residual comb) are plain data loads.
  bool isClockPinUse(Operation *user, OpOperand *use) {
    auto inst = dyn_cast<hw::InstanceOp>(user);
    if (!inst)
      return false;
    if (inst->hasAttrOfType<UnitAttr>("doNotPrint"))
      return false;
    auto *target = getOperation().lookupSymbol(inst.getModuleName());
    auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
    if (!targetModule || isa<hw::HWModuleExternOp>(targetModule))
      return true;
    SmallVector<hw::PortInfo> inputs, outputs;
    splitPorts(targetModule, inputs, outputs);
    unsigned idx = inputPortIndex(use);
    if (idx >= inputs.size())
      return true;
    const auto &port = inputs[idx];
    if (port.attrs) {
      if (auto pin = dyn_cast_or_null<DictionaryAttr>(
              port.attrs.get("synth.liberty.pin")))
        if (auto clock = dyn_cast_or_null<StringAttr>(pin.get("clock")))
          return clock.getValue() == "true";
    }
    return false;
  }

  // Partition a net's local loads. Unknown-target instance inputs mark
  // the net unbufferable (conservative); doNotPrint bind artifacts are
  // neither counted nor touched.
  LocalLoads collectLoads(Value value) {
    LocalLoads loads;
    for (auto &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<hw::OutputOp>(user))
        continue;
      if (auto inst = dyn_cast<hw::InstanceOp>(user)) {
        if (inst->hasAttrOfType<UnitAttr>("doNotPrint"))
          continue;
        auto *target = getOperation().lookupSymbol(inst.getModuleName());
        auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
        if (!targetModule || isa<hw::HWModuleExternOp>(targetModule)) {
          loads.hasUnknownTarget = true;
          continue;
        }
        if (isClockPinUse(user, &use)) {
          loads.hasClockLoad = true;
          continue;
        }
      }
      loads.dataLoads.push_back(&use);
    }
    return loads;
  }

  bool isStatefulTarget(Operation *target) {
    auto memo = statefulMemo.find(target);
    if (memo != statefulMemo.end())
      return memo->second;
    bool stateful = false;
    if (auto mod = dyn_cast_or_null<hw::HWModuleOp>(target)) {
      if (mod.getBodyBlock())
        stateful = mod.getBodyBlock()
                       ->walk([](Operation *op) {
                         if (isa<FFOp, LatchOp, ClockGateOp>(op))
                           return WalkResult::interrupt();
                         return WalkResult::advance();
                       })
                       .wasInterrupted();
    }
    statefulMemo[target] = stateful;
    return stateful;
  }

  // Every probe contributes its backward launch nets (flop Qs, top
  // inputs: combinational-cone boundaries) to violatorLaunches. The
  // visited set is global (linear total work); ordering is operand-order
  // DFS. Walk whitelist (safe default): data-transparent ops are crossed;
  // clocks (conservative pins), constants, and anything unlisted stop
  // the walk (under-collection only ever skips buffer candidates, never
  // breaks clocks or logic).
  DenseSet<Value> coneVisited;
  void collectProbeLaunches(hw::HWModuleOp module) {
    for (auto &block : module->getRegion(0))
      for (auto &op : block) {
        auto probe = dyn_cast<synth::TimingProbeOp>(op);
        if (!probe)
          continue;
        SmallVector<Value> worklist{probe.getInput()};
        while (!worklist.empty()) {
          Value current = worklist.pop_back_val();
          if (!coneVisited.insert(current).second)
            continue;
          Operation *defOp = current.getDefiningOp();
          if (!defOp) {
            violatorLaunches.insert(current); // Block arg / PI.
            continue;
          }
          if (isa<hw::ConstantOp>(defOp))
            continue;
          if (auto inner = dyn_cast<synth::TimingProbeOp>(defOp)) {
            worklist.push_back(inner.getInput());
            continue;
          }
          // Sequential boundary (instances and bare ops): launch,
          // never crossed.
          if (isa<FFOp, LatchOp, ClockGateOp>(defOp)) {
            violatorLaunches.insert(current);
            continue;
          }
          if (auto inst = dyn_cast<hw::InstanceOp>(defOp)) {
            auto *target = getOperation().lookupSymbol(inst.getModuleName());
            if (isStatefulTarget(target)) {
              violatorLaunches.insert(current);
              continue;
            }
            for (auto &operand : inst->getOpOperands()) {
              if (isClockPinUse(inst.getOperation(), &operand))
                continue;
              worklist.push_back(operand.get());
            }
            continue;
          }
          if (isa<BooleanLogicOpInterface>(defOp) ||
              isa<comb::MuxOp, comb::ExtractOp, comb::ConcatOp, hw::ArrayGetOp,
                  hw::ArrayCreateOp, hw::ArrayConcatOp, hw::ArraySliceOp>(
                  defOp)) {
            for (auto operand : defOp->getOperands())
              worklist.push_back(operand);
            continue;
          }
          // Unlisted (comb-arith, truth tables, unknown, memory ports):
          // stop (safe under-collection).
        }
      }
  }

  // Forward broadcast-distribution closure from the launch union:
  // data operators (extract/concat/array) and hierarchy ports, stopping
  // at gate pins (counted as loads elsewhere, not crossed), flop inputs
  // (sequential boundary), and clocks. Captures sibling loads sharing a
  // launch driver through parallel branches (fanin cones miss these:
  // the actual broadcast disease). Seeds in file order; membership is a
  // reachability fixpoint hence deterministic. Global visited (linear).
  void collectDistribution(ModuleOp top) {
    SmallVector<Value> worklist;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isLibModule(module))
        continue;
      for (auto &block : module->getRegion(0)) {
        for (auto arg : block.getArguments())
          if (violatorLaunches.contains(arg))
            worklist.push_back(arg);
        for (auto &op : block)
          for (auto result : op.getResults())
            if (violatorLaunches.contains(result))
              worklist.push_back(result);
      }
    }
    // Owner module per value for port crossing (recomputed lazily).
    DenseSet<Value> seen;
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!seen.insert(current).second)
        continue;
      violatorLaunches.insert(current);
      for (auto &use : current.getUses()) {
        Operation *user = use.getOwner();
        if (auto outOp = dyn_cast<hw::OutputOp>(user)) {
          // Through the port: parent instances' corresponding nets.
          unsigned portIdx = use.getOperandNumber();
          auto owner = dyn_cast<hw::HWModuleOp>(outOp->getParentOp());
          if (!owner)
            continue;
          auto instIt = instancesByTarget.find(owner.getModuleNameAttr());
          if (instIt == instancesByTarget.end())
            continue;
          for (auto &entry : instIt->second) {
            hw::InstanceOp inst = entry.second;
            if (portIdx >= inst.getNumResults())
              continue;
            worklist.push_back(inst.getResult(portIdx));
          }
          continue;
        }
        if (isa<hw::InstanceOp>(user))
          continue; // Gate pin: load, not crossed.
        if (user->hasTrait<mlir::OpTrait::ConstantLike>())
          continue;
        if (isa<FFOp, LatchOp, ClockGateOp>(user))
          continue; // Sequential: never crossed.
        if (isa<comb::ExtractOp, comb::ConcatOp, hw::ArrayGetOp,
                hw::ArrayCreateOp, hw::ArrayConcatOp, hw::ArraySliceOp>(user)) {
          for (auto result : user->getResults())
            worklist.push_back(result);
          continue;
        }
        // Comb gates and anything else: terminate (their input pins
        // are the counted loads; outputs are different nets).
      }
    }
  }

  // Output-port indices driven by a value (terminator operands).
  SmallVector<unsigned> drivenPorts(hw::HWModuleOp module, Value value) {
    SmallVector<unsigned> ports;
    auto *body = module.getBodyBlock();
    auto *terminator = body ? body->getTerminator() : nullptr;
    if (!terminator)
      return ports;
    for (unsigned i = 0; i < terminator->getNumOperands(); ++i)
      if (terminator->getOperand(i) == value)
        ports.push_back(i);
    return ports;
  }

  bool bufferModule(hw::HWModuleOp module, hw::HWModuleOp bufferCell) {
    // Collect values first; rewriting mutates use lists.
    SmallVector<Value> values;
    for (auto &block : module->getRegion(0)) {
      for (auto arg : block.getArguments())
        values.push_back(arg);
      for (auto &op : block)
        for (auto result : op.getResults())
          values.push_back(result);
    }
    // Probe launches were collected globally in phase 1 (see
    // runOnOperation); gating below uses the complete set.
    bool changed = false;
    OpBuilder builder(module.getContext());
    for (Value value : values) {
      // Local chunking handles 1-bit nets (buffers are 1-bit cells).
      // Port feeds below handle any integer width bit-wise.
      bool isI1 = value.getType().isInteger(1);
      bool isInt = value.getType().isInteger();
      // Timing-guided mode gates LOCAL chunking by violator launches.
      // Port feeds (below) are always evaluated: ports are rare,
      // high-leverage hierarchy boundaries, and ungated ports keep
      // through-hierarchy broadcasts (concat-mediated drivers) covered.
      bool launchGated = guidedMode && !violatorLaunches.contains(value);
      if (auto *defOp = value.getDefiningOp()) {
        if (defOp->hasTrait<mlir::OpTrait::ConstantLike>())
          continue;
        if (auto inst = dyn_cast<hw::InstanceOp>(defOp))
          if (inst.getNumOperands() == 0)
            continue; // Tie-cell (constant) drivers need no buffers.
      }
      LocalLoads loads;
      if (isI1) {
        loads = collectLoads(value);
        if (loads.hasUnknownTarget)
          continue;
      }
      // Chunk local data loads behind one buffer each (1-bit nets;
      // launch-gated in guided mode).
      size_t chunk = (size_t)std::max<int64_t>(maxFanout, 1);
      if (isI1 && !launchGated && loads.dataLoads.size() > chunk) {
        if (auto *defOp = value.getDefiningOp())
          builder.setInsertionPointAfter(defOp);
        else
          builder.setInsertionPointToStart(&module->getRegion(0).front());
        for (size_t begin = 0; begin < loads.dataLoads.size(); begin += chunk) {
          size_t end = std::min(begin + chunk, loads.dataLoads.size());
          SmallVector<Value> bufInputs{value};
          auto buf = hw::InstanceOp::create(builder, value.getLoc(), bufferCell,
                                            "buffered", bufInputs);
          ++numBuffersInserted;
          for (size_t i = begin; i < end; ++i)
            loads.dataLoads[i]->set(buf.getResult(0));
        }
        ++numNetsBuffered;
        changed = true;
      }
      // Buffers between driver and heavily loaded output ports (any
      // integer width, bit-wise). Port drivers join the actionable set
      // through the broadcast distribution; ungated only in legacy mode.
      if (guidedMode && !violatorLaunches.contains(value))
        continue;
      if (!isInt)
        continue;
      for (unsigned portIdx : drivenPorts(module, value)) {
        uint64_t down = downstreamLoads(module, portIdx);
        bool clk = downstreamClock(module, portIdx);
        if (down <= (uint64_t)maxFanout)
          continue;
        if (clk)
          continue;
        auto *terminator = module.getBodyBlock()->getTerminator();
        builder.setInsertionPoint(terminator);
        unsigned width = value.getType().getIntOrFloatBitWidth();
        if (width == 1) {
          SmallVector<Value> bufInputs{value};
          auto buf = hw::InstanceOp::create(builder, value.getLoc(), bufferCell,
                                            "buffered", bufInputs);
          ++numBuffersInserted;
          terminator->setOperand(portIdx, buf.getResult(0));
        } else {
          // Per-bit buffers recombined through a concat (deterministic
          // bit order).
          SmallVector<Value> bits;
          bits.reserve(width);
          for (unsigned b = 0; b < width; ++b) {
            auto bit =
                comb::ExtractOp::create(builder, value.getLoc(), value, b, 1);
            SmallVector<Value> bufInputs{bit};
            auto buf = hw::InstanceOp::create(
                builder, value.getLoc(), bufferCell, "buffered", bufInputs);
            ++numBuffersInserted;
            bits.push_back(buf.getResult(0));
          }
          auto concat =
              comb::ConcatOp::create(builder, value.getLoc(), ValueRange(bits));
          terminator->setOperand(portIdx, concat.getResult());
        }
        ++numNetsBuffered;
        changed = true;
      }
    }
    // Probes are consumed here: rewire to the raw value and erase so
    // nothing downstream (strip/export) ever observes one.
    for (auto &block : module->getRegion(0)) {
      SmallVector<synth::TimingProbeOp> probes;
      for (auto &op : block)
        if (auto probe = dyn_cast<synth::TimingProbeOp>(op))
          probes.push_back(probe);
      for (auto probe : probes) {
        probe.getResult().replaceAllUsesWith(probe.getInput());
        probe->erase();
        ++numProbesErased;
        changed = true;
      }
    }
      // Uniquify placeholder buffer names deterministically (same identity
    // requirement as TechMapper's mapped_N).
    unsigned bufferedCount = 0;
    module.walk([&](hw::InstanceOp inst) {
      StringRef name = inst.getInstanceName();
      if (name == "buffered" || name.starts_with("buffered_")) {
        inst->setAttr("instanceName",
                      StringAttr::get(inst.getContext(),
                                      "buffered_" + std::to_string(bufferedCount++)));
      }
    });
  return changed;
  }

  // Smallest-area identity (positive-phase) 1-in/1-out liberty cell.
  // PDK-agnostic: bodies are simulated bit-wise instead of name-matched.
  std::optional<hw::HWModuleOp> findBufferCell(ModuleOp top) {
    std::optional<hw::HWModuleOp> best;
    double bestArea = std::numeric_limits<double>::infinity();
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (!isLibModule(module))
        continue;
      if (module->hasAttr("synth.dont_use"))
        continue;
      SmallVector<hw::PortInfo> inputs, outputs;
      splitPorts(module, inputs, outputs);
      if (inputs.size() != 1 || outputs.size() != 1)
        continue;
      if (!inputs[0].type.isInteger(1) || !outputs[0].type.isInteger(1))
        continue;
      bool hasState = module.getBodyBlock()
                          ->walk([](Operation *op) {
                            if (isa<FFOp, LatchOp, ClockGateOp>(op))
                              return WalkResult::interrupt();
                            return WalkResult::advance();
                          })
                          .wasInterrupted();
      if (hasState)
        continue;
      if (!isIdentity(module))
        continue;
      double area = std::numeric_limits<double>::infinity();
      if (auto cost =
              module->getAttrOfType<MappingCostAttr>("synth.mapping_cost"))
        area = cost.getArea().getValue().convertToDouble();
      else if (auto cellDict =
                   module->getAttrOfType<DictionaryAttr>("synth.liberty.cell"))
        if (auto areaAttr = dyn_cast_or_null<FloatAttr>(cellDict.get("area")))
          area = areaAttr.getValue().convertToDouble();
      if (area < bestArea) {
        bestArea = area;
        best = module;
      }
    }
    return best;
  }

  // Bit-wise identity check over a single-block comb body.
  static std::optional<bool> evalBody(hw::HWModuleOp module, bool input) {
    if (!module.getBodyBlock() || !module->getRegion(0).hasOneBlock())
      return std::nullopt;
    DenseMap<Value, bool> env;
    auto args = module.getBodyBlock()->getArguments();
    if (args.size() != 1)
      return std::nullopt;
    env[args[0]] = input;
    std::optional<bool> output;
    for (auto &op : *module.getBodyBlock()) {
      if (isa<hw::OutputOp>(op)) {
        if (op.getNumOperands() != 1)
          return std::nullopt;
        auto found = env.find(op.getOperand(0));
        if (found == env.end())
          return std::nullopt;
        output = found->second;
        continue;
      }
      if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
        if (!constant.getType().isInteger(1))
          return std::nullopt;
        env[constant.getResult()] = !constant.getValue().isZero();
        continue;
      }
      if (auto logic = dyn_cast<BooleanLogicOpInterface>(op)) {
        SmallVector<APInt> bits;
        for (auto operand : logic.getInputs()) {
          auto found = env.find(operand);
          if (found == env.end())
            return std::nullopt;
          bits.push_back(APInt(1, found->second));
        }
        unsigned i = 0;
        APInt result = logic.evaluateBooleanLogic(
            [&](unsigned) -> const APInt & { return bits[i++]; });
        if (result.getBitWidth() != 1)
          return std::nullopt;
        env[logic.getResult()] = result.isOne();
        continue;
      }
      return std::nullopt;
    }
    return output;
  }

  static bool isIdentity(hw::HWModuleOp module) {
    auto out0 = evalBody(module, false);
    auto out1 = evalBody(module, true);
    return out0 && out1 && !*out0 && *out1;
  }
};

} // namespace
