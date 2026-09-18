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
#include "circt/Dialect/Synth/SynthOpInterfaces.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
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
    auto buffer = findBufferCell(top);
    if (!buffer) {
      LLVM_DEBUG(llvm::dbgs() << "no identity buffer cell found\n");
      return markAllAnalysesPreserved();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "buffer cell: " << buffer->getModuleName() << "\n");
    buildChildren();

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
        total += gateUses(net);
        // The net may drive the parent's own output ports: recurse.
        if (auto outputOp = dyn_cast_or_null<hw::OutputOp>(
                parent.getBodyBlock()->getTerminator())) {
          for (unsigned q = 0; q < outputOp->getNumOperands(); ++q) {
            if (outputOp->getOperand(q) == net &&
                q < cacheIt->second.second.size())
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

  // Output-port indices driven by a value (terminator operands).
  SmallVector<unsigned> drivenPorts(hw::HWModuleOp module, Value value) {
    SmallVector<unsigned> ports;
    auto *terminator = module.getBodyBlock()->getTerminator();
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
    bool changed = false;
    OpBuilder builder(module.getContext());
    for (Value value : values) {
      if (!value.getType().isInteger(1))
        continue;
      if (auto *defOp = value.getDefiningOp()) {
        if (defOp->hasTrait<mlir::OpTrait::ConstantLike>())
          continue;
        if (auto inst = dyn_cast<hw::InstanceOp>(defOp))
          if (inst.getNumOperands() == 0)
            continue; // Tie-cell (constant) drivers need no buffers.
      }
      LocalLoads loads = collectLoads(value);
      if (loads.hasUnknownTarget)
        continue;
      // Chunk local data loads behind one buffer each.
      size_t chunk = (size_t)std::max<int64_t>(maxFanout, 1);
      if (loads.dataLoads.size() > chunk) {
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
      // One buffer between driver and heavily loaded output ports.
      for (unsigned portIdx : drivenPorts(module, value)) {
        if (downstreamLoads(module, portIdx) <= (uint64_t)maxFanout)
          continue;
        if (downstreamClock(module, portIdx))
          continue;
        auto *terminator = module.getBodyBlock()->getTerminator();
        builder.setInsertionPoint(terminator);
        SmallVector<Value> bufInputs{value};
        auto buf = hw::InstanceOp::create(builder, value.getLoc(), bufferCell,
                                          "buffered", bufInputs);
        ++numBuffersInserted;
        terminator->setOperand(portIdx, buf.getResult(0));
        ++numNetsBuffered;
        changed = true;
      }
    }
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
