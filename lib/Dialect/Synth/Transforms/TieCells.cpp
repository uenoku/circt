//===- TieCells.cpp - Drive constant inputs from tie cells ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ExportVerilog emits `hw.constant` drivers as Verilog literals, but the
// reference flow ties constant inputs through tie cells (`conb_1` on
// sky130, `TIEHIx1`/`TIELOx1` on ASAP7). This pass reroutes i1-constant
// operands of `hw.instance` ops to shared tie-cell instances, one per
// needed polarity per design module.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_TIECELLS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-tie-cells"

namespace {

// A tie-cell output driving one constant polarity. The target may be a
// plain module (pre-strip) or an extern (post-strip port metadata carries
// over, so discovery works in both cases).
struct TieSource {
  Operation *moduleOp;
  unsigned resultIdx;
  double area;
};

static bool isLibertyCell(hw::HWModuleOp module) {
  return module->hasAttr("synth.liberty.cell") ||
         module->hasAttr("synth.dont_use");
}

/// Consider one module (plain or extern) as a tie-cell candidate: a
/// liberty-sourced cell with no inputs and an i1 output whose pin function
/// is the wanted constant. Plain modules need the liberty marker (so
/// constant-emitting design hierarchy is never used as a tie source);
/// externs lost their marker to stripping, but pin metadata carries over.
/// Smallest area wins.
static void considerTieCandidate(Operation *moduleOp,
                                 ArrayRef<hw::PortInfo> ports, bool eligible,
                                 StringRef want,
                                 std::optional<TieSource> &best) {
  if (!eligible)
    return;
  for (const auto &port : ports)
    if (port.isInput() || port.isInOut())
      return;
  unsigned outputIdx = 0;
  for (const auto &port : ports) {
    if (!port.isOutput()) {
      ++outputIdx;
      continue;
    }
    bool match = false;
    if (port.attrs) {
      if (auto pin = dyn_cast_or_null<DictionaryAttr>(
              port.attrs.get("synth.liberty.pin")))
        if (auto function = dyn_cast_or_null<StringAttr>(pin.get("function")))
          match = function.getValue() == want;
    }
    if (match && port.type.isInteger(1)) {
      double area = std::numeric_limits<double>::infinity();
      if (auto cellDict =
              moduleOp->getAttrOfType<DictionaryAttr>("synth.liberty.cell"))
        if (auto areaAttr = dyn_cast_or_null<FloatAttr>(cellDict.get("area")))
          area = areaAttr.getValueAsDouble();
      if (!best || area < best->area)
        best = TieSource{moduleOp, outputIdx, area};
    }
    ++outputIdx;
  }
}

static std::optional<TieSource> findTieCell(ModuleOp top, bool polarity) {
  std::optional<TieSource> best;
  StringRef want = polarity ? "1" : "0";
  for (auto module : top.getOps<hw::HWModuleOp>())
    considerTieCandidate(module.getOperation(), module.getPortList(),
                         isLibertyCell(module), want, best);
  for (auto externOp : top.getOps<hw::HWModuleExternOp>())
    considerTieCandidate(externOp.getOperation(), externOp.getPortList(),
                         /*eligible=*/true, want, best);
  return best;
}

struct TieCellsPass : public circt::synth::impl::TieCellsBase<TieCellsPass> {
  using TieCellsBase<TieCellsPass>::TieCellsBase;

  void runOnOperation() override {
    ModuleOp top = getOperation();
    auto hiTie = findTieCell(top, /*polarity=*/true);
    auto loTie = findTieCell(top, /*polarity=*/false);
    if (!hiTie && !loTie)
      return markAllAnalysesPreserved();

    bool changed = false;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      // Externs have no body; library cells provide ties, never consume.
      if (isa<hw::HWModuleExternOp>(module) || isLibertyCell(module))
        continue;
      changed |= tieModule(module, hiTie, loTie);
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

  bool tieModule(hw::HWModuleOp module, std::optional<TieSource> hiTie,
                 std::optional<TieSource> loTie) {
    bool changed = false;
    // Lazily created shared tie instances, indexed by polarity.
    hw::InstanceOp tieInst[2];
    auto getTie = [&](bool polarity, Location loc) -> Value {
      if (tieInst[polarity]) {
        TieSource known = polarity ? *hiTie : *loTie;
        return tieInst[polarity].getResult(known.resultIdx);
      }
      TieSource source = polarity ? *hiTie : *loTie;
      OpBuilder builder(module.getBodyBlock(), module.getBodyBlock()->begin());
      auto inst = hw::InstanceOp::create(builder, loc, source.moduleOp,
                                         polarity ? "tied_hi" : "tied_lo",
                                         ArrayRef<Value>{});
      tieInst[polarity] = inst;
      ++numTieInstances;
      return inst.getResult(source.resultIdx);
    };

    // Collect first; rewriting mutates operands.
    SmallVector<std::pair<OpOperand *, bool>> reroutes;
    for (auto &block : module->getRegion(0)) {
      for (auto &op : block) {
        auto inst = dyn_cast<hw::InstanceOp>(op);
        if (!inst || inst->hasAttr("doNotPrint") || inst.getInnerSymAttr())
          continue;
        for (OpOperand &operand : inst->getOpOperands()) {
          Value value = operand.get();
          if (!value.getType().isInteger(1))
            continue;
          auto constant = value.getDefiningOp<hw::ConstantOp>();
          if (!constant)
            continue;
          bool polarity = !constant.getValue().isZero();
          if ((polarity && !hiTie) || (!polarity && !loTie))
            continue;
          reroutes.emplace_back(&operand, polarity);
        }
      }
    }
    for (auto [operand, polarity] : reroutes) {
      Location loc = operand->get().getLoc();
      Value tie = getTie(polarity, loc);
      operand->set(tie);
      ++numRerouted;
      changed = true;
    }
    // Erase newly-dead constants local to this module.
    for (auto &block : module->getRegion(0)) {
      SmallVector<hw::ConstantOp> dead;
      for (auto &op : block)
        if (auto constant = dyn_cast<hw::ConstantOp>(op))
          if (constant->use_empty())
            dead.push_back(constant);
      for (auto constant : dead)
        constant.erase();
    }
    return changed;
  }
};

} // namespace
