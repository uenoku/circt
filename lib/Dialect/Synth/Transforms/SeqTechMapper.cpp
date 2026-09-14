//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SeqTechMapper pass, which maps sequential
// operations (`seq.firreg`, `seq.compreg`, `seq.compreg.ce`) to instances of
// library cells, the Yosys `dfflibmap` equivalent.
//
// Library cells are `hw.module`s in the same `ModuleOp` whose bodies contain
// a `synth.ff` state element (as produced by the Liberty importer). Only
// rising-edge flops map, matching the `seq` positive-edge convention.
// Synchronous resets and enables are emulated with external `comb.mux`
// logic when the chosen cell lacks the pin; asynchronous resets require a
// matching cell pin. Extra cell inputs (such as scan pins) are tied to zero.
// Among compatible cells, the one with the smallest Liberty `area` wins.
// Unmappable flops are left untouched.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <limits>
#include <tuple>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_SEQTECHMAPPER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

#define DEBUG_TYPE "synth-seq-tech-mapper"

namespace {

bool isI1(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == 1;
  return false;
}

// Peel one `comb.xor`-with-true inversion.
std::pair<Value, bool> peelNot(Value val) {
  auto xorOp = val.getDefiningOp<comb::XorOp>();
  if (!xorOp || xorOp.getInputs().size() != 2)
    return {val, false};
  Value other;
  for (auto operand : xorOp.getInputs()) {
    if (auto constOp = operand.getDefiningOp<hw::ConstantOp>()) {
      auto intTy = dyn_cast<IntegerType>(constOp.getType());
      if (intTy && intTy.getWidth() == 1 && constOp.getValue() == 1)
        continue;
      return {val, false};
    }
    if (other)
      return {val, false};
    other = operand;
  }
  if (!other)
    return {val, false};
  return {other, true};
}

// Trace a value through inversions to a body block argument of `module`.
// Returns the input port name and whether it was inverted.
FailureOr<std::pair<StringAttr, bool>> traceToInputPin(hw::HWModuleOp module,
                                                       Value val) {
  bool inverted = false;
  while (!isa<BlockArgument>(val)) {
    auto [next, flipped] = peelNot(val);
    if (!flipped)
      return failure();
    inverted = !inverted;
    val = next;
  }
  auto arg = cast<BlockArgument>(val);
  for (const auto &port : module.getPortList()) {
    if (!port.isInput())
      continue;
    if (port.argNum == arg.getArgNumber())
      return std::make_pair(port.name, inverted);
  }
  return failure();
}

// A library cell signature extracted from its body.
struct CellSignature {
  hw::HWModuleOp module;
  StringAttr clockPin;
  StringAttr dataPin;
  // Optional controls as (pin, inverted-at-port).
  std::optional<std::pair<StringAttr, bool>> enable;
  std::optional<std::pair<StringAttr, bool>> clear;
  std::optional<std::pair<StringAttr, bool>> preset;
  // Extra input pins (scan and the like) tied to a constant.
  SmallVector<std::pair<StringAttr, bool>> tiedPins;
  // Result index of the state output.
  unsigned outputIdx = 0;
  double area = std::numeric_limits<double>::infinity();
};

// Extract the signature of a library cell: a module with a single
// rising-edge `synth.ff` and no `seq` flops. Fails for anything else.
FailureOr<CellSignature> extractSignature(hw::HWModuleOp module) {
  CellSignature sig;
  sig.module = module;
  auto *body = module.getBodyBlock();
  if (!body)
    return failure();
  synth::FFOp ffOp;
  for (auto op : body->getOps<synth::FFOp>()) {
    if (ffOp)
      return failure();
    ffOp = op;
  }
  if (!ffOp)
    return failure();
  // Rising edge only, matching the `seq` positive-edge convention. The
  // importer encodes clock polarity in the edge, never in logic.
  if (ffOp.getEdge() != synth::EdgeKind::Rise)
    return failure();
  auto clockArg = dyn_cast<BlockArgument>(ffOp.getClk());
  if (!clockArg)
    return failure();
  bool clockFound = false;
  for (const auto &port : module.getPortList()) {
    if (!port.isInput())
      continue;
    if (port.argNum == clockArg.getArgNumber()) {
      sig.clockPin = port.name;
      clockFound = true;
    }
  }
  if (!clockFound)
    return failure();

  llvm::DenseSet<StringAttr> tracedPins;
  tracedPins.insert(sig.clockPin);

  // Data: a plain pin or a scan mux (select, true=scanIn, false=data) as
  // built by the Liberty importer, each leg optionally inverted.
  auto traceOperand = [&](Value v) -> FailureOr<std::pair<StringAttr, bool>> {
    return traceToInputPin(module, v);
  };
  Value data = ffOp.getData();
  if (auto muxOp = data.getDefiningOp<comb::MuxOp>()) {
    auto sel = traceOperand(muxOp.getCond());
    auto tru = traceOperand(muxOp.getTrueValue());
    auto fls = traceOperand(muxOp.getFalseValue());
    if (failed(sel) || failed(tru) || failed(fls))
      return failure();
    // Tie the select to choose the data leg, and the scan leg to zero.
    sig.tiedPins.emplace_back(sel->first, sel->second);
    sig.tiedPins.emplace_back(tru->first, false);
    sig.dataPin = fls->first;
  } else {
    auto base = traceOperand(data);
    if (failed(base))
      return failure();
    sig.dataPin = base->first;
  }
  tracedPins.insert(sig.dataPin);

  auto traceControl =
      [&](Value v,
          std::optional<std::pair<StringAttr, bool>> &slot) -> LogicalResult {
    if (!v)
      return success();
    auto pin = traceOperand(v);
    if (failed(pin))
      return failure();
    slot = *pin;
    return success();
  };
  if (failed(traceControl(ffOp.getEnable(), sig.enable)) ||
      failed(traceControl(ffOp.getClear(), sig.clear)) ||
      failed(traceControl(ffOp.getPreset(), sig.preset)))
    return failure();
  auto noteTraced =
      [&](const std::optional<std::pair<StringAttr, bool>> &slot) {
        if (slot)
          tracedPins.insert(slot->first);
      };
  noteTraced(sig.enable);
  noteTraced(sig.clear);
  noteTraced(sig.preset);
  for (const auto &tied : sig.tiedPins)
    tracedPins.insert(tied.first);

  // Remaining input pins (scan extras and the like) are tied to zero.
  for (const auto &port : module.getPortList()) {
    if (!port.isInput() || tracedPins.count(port.name))
      continue;
    sig.tiedPins.emplace_back(port.name, false);
  }

  // The output port carrying the state: the terminator operand that is the
  // `q` result. Instance results follow output ports in order.
  auto *terminator = body->getTerminator();
  if (!terminator)
    return failure();
  unsigned outputIdx = 0;
  bool foundOutput = false;
  for (const auto &port : module.getPortList()) {
    if (!port.isOutput())
      continue;
    if (outputIdx >= terminator->getNumOperands())
      return failure();
    if (terminator->getOperand(outputIdx) == ffOp.getQ()) {
      sig.outputIdx = outputIdx;
      foundOutput = true;
    }
    ++outputIdx;
  }
  if (!foundOutput)
    return failure();

  if (auto dict = module->getAttrOfType<DictionaryAttr>("synth.liberty.cell")) {
    if (auto area = dyn_cast_or_null<FloatAttr>(dict.get("area")))
      sig.area = area.getValue().convertToDouble();
  }
  return sig;
}

// Wide flops (total bit width > 1) that can be exploded bit by bit:
// reset values, if any, must be constants so each bit keeps one.
// Aggregate types go through a lossless bitcast (bitcast is pure
// bitwise), so arrays of integers explode just like wide integers.
SmallVector<Operation *> collectWideFlops(hw::HWModuleOp module) {
  SmallVector<Operation *> wide;
  module.getBodyBlock()->walk([&](Operation *op) {
    if (op->getNumResults() != 1)
      return;
    if (hw::getBitWidth(op->getResult(0).getType()) <= 1)
      return;
    if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
      if (firreg.getReset() &&
          !firreg.getResetValue().getDefiningOp<hw::ConstantOp>())
        return;
      wide.push_back(op);
    } else if (auto compreg = dyn_cast<seq::CompRegOp>(op)) {
      if (compreg.getReset() &&
          !compreg.getResetValue().getDefiningOp<hw::ConstantOp>())
        return;
      wide.push_back(op);
    } else if (auto ce = dyn_cast<seq::CompRegClockEnabledOp>(op)) {
      if (ce.getReset() && !ce.getResetValue().getDefiningOp<hw::ConstantOp>())
        return;
      wide.push_back(op);
    }
  });
  return wide;
}

// Replace a wide flop with per-bit flops concatenated MSB-first. Returns
// false (leaving the op alone) if reset values are not per-bit constants.
bool explodeFlop(OpBuilder &builder, Operation *op) {
  Location loc = op->getLoc();
  Type resultTy = op->getResult(0).getType();
  int64_t totalBits = hw::getBitWidth(resultTy);
  if (totalBits <= 1)
    return false;
  unsigned width = static_cast<unsigned>(totalBits);
  auto intTy = builder.getIntegerType(width);
  Value next, clock, reset, enable, resetValue;
  bool isAsync = false;
  std::string baseName;
  if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
    next = firreg.getNext();
    clock = firreg.getClk();
    reset = firreg.getReset();
    resetValue = firreg.getResetValue();
    isAsync = firreg.getIsAsync();
    if (!firreg.getName().empty())
      baseName = firreg.getName().str();
  } else if (auto compreg = dyn_cast<seq::CompRegOp>(op)) {
    next = compreg.getInput();
    clock = compreg.getClk();
    reset = compreg.getReset();
    resetValue = compreg.getResetValue();
    if (auto name = compreg.getNameAttr())
      baseName = name.getValue().str();
  } else if (auto ce = dyn_cast<seq::CompRegClockEnabledOp>(op)) {
    next = ce.getInput();
    clock = ce.getClk();
    enable = ce.getClockEnable();
    reset = ce.getReset();
    resetValue = ce.getResetValue();
    if (auto name = ce.getNameAttr())
      baseName = name.getValue().str();
  } else {
    return false;
  }
  bool needsCast = !isa<IntegerType>(resultTy);
  APInt resetBits(width, 0);
  if (reset) {
    // Aggregate reset values have no per-bit constant; only reset-free
    // aggregates explode.
    if (needsCast)
      return false;
    auto constOp = resetValue.getDefiningOp<hw::ConstantOp>();
    if (!constOp)
      return false;
    resetBits = constOp.getValue();
  }
  auto i1 = builder.getI1Type();
  mlir::ImplicitLocOpBuilder ibuilder(loc, builder);
  builder.setInsertionPoint(op);
  if (needsCast)
    next = hw::BitcastOp::create(builder, loc, intTy, next);
  SmallVector<Value> bits;
  for (unsigned i = 0; i < width; ++i) {
    Value dataBit = comb::ExtractOp::create(builder, loc, next, i, 1);
    Value resetBit;
    if (reset) {
      resetBit = hw::ConstantOp::create(builder, loc, i1, resetBits[i] ? 1 : 0);
    }
    StringAttr bitName;
    if (!baseName.empty())
      bitName = builder.getStringAttr(baseName + "_bit" + Twine(i).str());
    Value q;
    if (isa<seq::FirRegOp>(op)) {
      if (reset) {
        q = seq::FirRegOp::create(builder, loc, dataBit, clock, bitName, reset,
                                  resetBit, hw::InnerSymAttr{}, isAsync);
      } else {
        q = seq::FirRegOp::create(builder, loc, dataBit, clock, bitName);
      }
    } else if (isa<seq::CompRegOp>(op)) {
      q = seq::CompRegOp::create(ibuilder, i1, dataBit, clock, bitName, reset,
                                 resetBit, Value(), hw::InnerSymAttr{});
    } else {
      q = seq::CompRegClockEnabledOp::create(ibuilder, i1, dataBit, clock,
                                             enable, bitName, reset, resetBit,
                                             Value(), hw::InnerSymAttr{});
    }
    bits.push_back(q);
  }
  // concat takes MSB first.
  SmallVector<Value> reversed(bits.rbegin(), bits.rend());
  Value concat = comb::ConcatOp::create(builder, loc, reversed);
  Value qval = concat;
  if (needsCast)
    qval = hw::BitcastOp::create(builder, loc, resultTy, concat);
  op->getResult(0).replaceAllUsesWith(qval);
  op->erase();
  return true;
}

// A design flop in canonical terms.
struct FlopSpec {
  Operation *op;
  Value data;
  Value clock;  // ClockType.
  Value enable; // Null if none; active-high synchronous.
  bool hasReset = false;
  bool isAsync = false;
  Value reset; // Active-high.
  bool resetValue = false;
};

// A constant i1 value, or failure.
FailureOr<bool> getConstantI1(Value val) {
  auto constOp = val.getDefiningOp<hw::ConstantOp>();
  if (!constOp || !isI1(constOp.getType()))
    return failure();
  if (constOp.getValue().isZero())
    return false;
  if (constOp.getValue() == 1)
    return true;
  return failure();
}

FailureOr<FlopSpec> extractFlop(Operation *op) {
  FlopSpec spec;
  spec.op = op;
  if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
    if (!isI1(firreg.getData().getType()))
      return failure();
    if (firreg.getPreset().has_value())
      return failure(); // Power-on init has no cell equivalent.
    spec.data = firreg.getNext();
    spec.clock = firreg.getClk();
    if (firreg.getReset()) {
      spec.hasReset = true;
      spec.isAsync = firreg.getIsAsync();
      auto resetValue = getConstantI1(firreg.getResetValue());
      if (failed(resetValue))
        return failure();
      spec.resetValue = *resetValue;
      spec.reset = firreg.getReset();
    }
    return spec;
  }
  if (auto compreg = dyn_cast<seq::CompRegOp>(op)) {
    if (!isI1(compreg.getData().getType()))
      return failure();
    if (compreg.getInitialValue())
      return failure();
    spec.data = compreg.getInput();
    spec.clock = compreg.getClk();
    if (compreg.getReset()) {
      // `seq.compreg` resets are synchronous (see SeqToSV).
      spec.hasReset = true;
      spec.isAsync = false;
      auto resetValue = getConstantI1(compreg.getResetValue());
      if (failed(resetValue))
        return failure();
      spec.resetValue = *resetValue;
      spec.reset = compreg.getReset();
    }
    return spec;
  }
  if (auto ce = dyn_cast<seq::CompRegClockEnabledOp>(op)) {
    if (!isI1(ce.getData().getType()))
      return failure();
    if (ce.getInitialValue())
      return failure();
    spec.data = ce.getInput();
    spec.clock = ce.getClk();
    spec.enable = ce.getClockEnable();
    if (ce.getReset()) {
      spec.hasReset = true;
      spec.isAsync = false;
      auto resetValue = getConstantI1(ce.getResetValue());
      if (failed(resetValue))
        return failure();
      spec.resetValue = *resetValue;
      spec.reset = ce.getReset();
    }
    return spec;
  }
  return failure();
}

struct Match {
  CellSignature *cell;
  bool emulateEnable = false;
  bool emulateReset = false;
};
;

// Find the best compatible cell. Synchronous resets and enables can be
// emulated with external muxes; asynchronous resets need a matching pin.
// Direct feature matches rank before emulations (their cost is not in the
// area model), then smallest area, then name for determinism.
std::optional<Match> matchFlop(const FlopSpec &flop,
                               SmallVectorImpl<CellSignature> &lib) {
  std::optional<Match> best;
  std::optional<std::tuple<bool, double, StringRef>> bestRank;
  for (auto &cell : lib) {
    if (flop.hasReset && flop.isAsync) {
      if (!flop.resetValue && !cell.clear)
        continue;
      if (flop.resetValue && !cell.preset)
        continue;
    }
    Match match;
    match.cell = &cell;
    match.emulateEnable =
        static_cast<bool>(flop.enable) && !cell.enable.has_value();
    match.emulateReset = flop.hasReset && !flop.isAsync;
    auto rank = std::make_tuple(match.emulateEnable, cell.area,
                                cell.module.getModuleName());
    if (!bestRank || rank < *bestRank) {
      best = match;
      bestRank = rank;
    }
  }
  return best;
}

Value clockToI1(OpBuilder &builder, Location loc, Value clock) {
  if (auto toClock = clock.getDefiningOp<seq::ToClockOp>())
    return toClock.getInput();
  return seq::FromClockOp::create(builder, loc, clock);
}

struct SeqTechMapperPass
    : public circt::synth::impl::SeqTechMapperBase<SeqTechMapperPass> {
  using SeqTechMapperBase<SeqTechMapperPass>::SeqTechMapperBase;

  void runOnOperation() override {
    auto top = getOperation();

    auto definesSeqFlop = [](hw::HWModuleOp module) {
      return module.getBodyBlock()
          ->walk([](Operation *op) {
            if (isa<seq::FirRegOp, seq::CompRegOp, seq::CompRegClockEnabledOp>(
                    op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted();
    };

    // Library cells: modules with a decodable `synth.ff` and no `seq` flops.
    SmallVector<CellSignature> lib;
    DenseSet<Operation *> libraryModules;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (definesSeqFlop(module))
        continue;
      auto sig = extractSignature(module);
      if (failed(sig))
        continue;
      libraryModules.insert(module);
      lib.push_back(std::move(*sig));
    }
    if (lib.empty())
      return markAllAnalysesPreserved();

    OpBuilder builder(top.getContext());
    auto i1 = builder.getI1Type();
    auto constVal = [&](Location loc, bool v) {
      return hw::ConstantOp::create(builder, loc, i1, v ? 1 : 0);
    };

    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (libraryModules.contains(module))
        continue;
      // Library cells are single-bit; explode wide integer flops bit by
      // bit first (Yosys `techmap` does the same for `$dff`). Only when
      // reset values (if any) are constants, so each bit keeps the
      // constant reset the mapper requires.
      for (auto *wideOp : collectWideFlops(module))
        if (explodeFlop(builder, wideOp))
          ++numFlopsExploded;
      // Collect first; rewriting mutates the block.
      SmallVector<Operation *> flops;
      module.getBodyBlock()->walk([&](Operation *op) {
        if (isa<seq::FirRegOp, seq::CompRegOp, seq::CompRegClockEnabledOp>(op))
          flops.push_back(op);
      });
      for (auto *flopOp : flops) {
        auto spec = extractFlop(flopOp);
        if (failed(spec)) {
          ++numFlopsSkipped;
          continue;
        }
        auto match = matchFlop(*spec, lib);
        if (!match) {
          ++numFlopsSkipped;
          continue;
        }
        Location loc = flopOp->getLoc();
        builder.setInsertionPoint(flopOp);
        auto &cell = *match->cell;

        // Adjust an active-high design signal to a cell port, inserting an
        // inverter when the port is active-low.
        auto toPort = [&](Value sig,
                          const std::pair<StringAttr, bool> &pin) -> Value {
          if (!pin.second)
            return sig;
          return comb::XorOp::create(builder, loc, sig, constVal(loc, true));
        };

        Value clkI1 = clockToI1(builder, loc, spec->clock);
        Value dataVal = spec->data;
        if (match->emulateReset) {
          dataVal =
              comb::MuxOp::create(builder, loc, spec->reset,
                                  constVal(loc, spec->resetValue), dataVal);
        }

        // Instance inputs in input-port order.
        hw::ModulePortInfo ports(cell.module.getPortList());
        SmallVector<Value> inputs;
        unsigned dataIdx = 0;
        {
          unsigned idx = 0;
          for (const auto &port : ports.getInputs()) {
            Value val;
            if (port.name == cell.clockPin) {
              val = clkI1;
            } else if (port.name == cell.dataPin) {
              val = dataVal;
              dataIdx = idx;
            } else if (cell.enable && port.name == cell.enable->first) {
              if (spec->enable) {
                val = toPort(spec->enable, *cell.enable);
              } else {
                // Tie the enable active.
                val = constVal(loc, cell.enable->second ? false : true);
              }
            } else if (cell.clear && port.name == cell.clear->first) {
              if (spec->hasReset && spec->isAsync && !spec->resetValue) {
                val = toPort(spec->reset, *cell.clear);
              } else {
                val = constVal(loc, cell.clear->second ? true : false);
              }
            } else if (cell.preset && port.name == cell.preset->first) {
              if (spec->hasReset && spec->isAsync && spec->resetValue) {
                val = toPort(spec->reset, *cell.preset);
              } else {
                val = constVal(loc, cell.preset->second ? true : false);
              }
            } else {
              // Scan and other extras tied to zero.
              val = constVal(loc, false);
            }
            inputs.push_back(val);
            ++idx;
          }
        }

        StringAttr instName = builder.getStringAttr("mapped");
        hw::InnerSymAttr innerSym;
        if (auto firreg = dyn_cast<seq::FirRegOp>(flopOp)) {
          if (!firreg.getName().empty())
            instName = builder.getStringAttr(firreg.getName());
          innerSym = firreg.getInnerSymAttr();
        } else if (auto compreg = dyn_cast<seq::CompRegOp>(flopOp)) {
          if (auto name = compreg.getNameAttr();
              name && !name.getValue().empty())
            instName = name;
          innerSym = compreg.getInnerSymAttr();
        } else if (auto ce = dyn_cast<seq::CompRegClockEnabledOp>(flopOp)) {
          if (auto name = ce.getNameAttr(); name && !name.getValue().empty())
            instName = name;
          innerSym = ce.getInnerSymAttr();
        }
        auto inst =
            hw::InstanceOp::create(builder, loc, cell.module, instName, inputs,
                                   builder.getArrayAttr({}), innerSym);
        Value q = inst.getResult(cell.outputIdx);
        if (match->emulateEnable) {
          // Hold-by-feedback: q does not exist until the instance does, so
          // wire the mux after the fact. The cycle is legal for the same
          // reason `seq.firreg` feedback is.
          Value held =
              comb::MuxOp::create(builder, loc, spec->enable, dataVal, q);
          inst->setOperand(dataIdx, held);
        }
        spec->op->getResult(0).replaceAllUsesWith(q);
        spec->op->erase();
        ++numFlopsMapped;
      }
    }
  }
};

} // namespace
