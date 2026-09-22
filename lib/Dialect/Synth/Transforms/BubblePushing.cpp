//===- BubblePushing.cpp - Absorb inverters into complements -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Absorb single-fanout inverters into complement cells: when an inverter is
// driven by a cell whose complemented function exists in the library at
// lower combined area (e.g. AND+INV to NAND, MAJ+INV to minority), replace
// both with the complement. Functions match by exact truth table with
// identical port-name sequences (no pinout guessing); costs come from
// mapping_cost. Single pass (each inverter visited once, only removals).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_BUBBLEPUSHING
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-bubble-pushing"

namespace {

// Max intrinsic delay over mapping_cost arcs (0 if none).
int64_t maxIntrinsic(hw::HWModuleOp module) {
  auto cost = module->getAttrOfType<MappingCostAttr>("synth.mapping_cost");
  if (!cost || !cost.getArcs())
    return 0;
  int64_t best = 0;
  for (auto arc : cost.getArcs()) {
    auto arcAttr = dyn_cast<LinearTimingArcAttr>(arc);
    if (arcAttr)
      best = std::max(best, arcAttr.getIntrinsic());
  }
  return best;
}

// True for single-output modules with purely combinational bodies.
bool isCombinationalCell(hw::HWModuleOp module) {
  auto *body = module.getBodyBlock();
  if (!body || body->getOperations().empty())
    return false;
  bool ok = true;
  body->walk([&](Operation *op) {
    if (isa<hw::HWModuleOp>(op))
      return;
    auto *dialect = op->getDialect();
    if (!dialect)
      return;
    StringRef ns = dialect->getNamespace();
    // AIG ops (synth.aig.*) are combinational; other synth/seq ops hold
    // state and have no truth table.
    if (ns == "synth" && op->getName().getStringRef().contains(".aig."))
      return;
    if (ns != "comb" && ns != "arith" && ns != "hw")
      ok = false;
  });
  return ok;
}

struct BubblePushingPass
    : public impl::BubblePushingBase<BubblePushingPass> {
  void runOnOperation() override {
    auto top = getOperation();
    SymbolTable symbols(top);

    // Library cells: modules with mapping costs, keyed by truth table.
    struct CellInfo {
      hw::HWModuleOp module;
      double area;
      SmallVector<StringAttr> ports;
    };
    std::map<std::string, SmallVector<CellInfo>> cellsByTable;
    auto tableKey = [](const BinaryTruthTable &tt) {
      std::string key = std::to_string(tt.numInputs) + ":";
      llvm::raw_string_ostream os(key);
      tt.table.print(os, /*isSigned=*/false);
      return key;
    };
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      auto cost = module->getAttrOfType<MappingCostAttr>("synth.mapping_cost");
      if (!cost || !cost.getArea())
        continue;
      // Combinational bodies only (truth tables don't cover seq/synth).
      if (!isCombinationalCell(module))
        continue;
      auto tt = getTruthTable(module);
      if (failed(tt) || tt->numOutputs != 1)
        continue;
      SmallVector<StringAttr> ports;
      for (const auto &port : module.getPortList()) {
        if (port.isInput())
          ports.push_back(port.name);
      }
      if ((unsigned)ports.size() != tt->numInputs)
        continue;
      cellsByTable[tableKey(*tt)].push_back(
          {module, cost.getArea().getValueAsDouble(), std::move(ports)});
    }
    if (cellsByTable.empty())
      return;

    // NOT truth table (single input).
    auto isNot = [](const BinaryTruthTable &tt) {
      return tt.numInputs == 1 && tt.numOutputs == 1 &&
             tt.table.getBitWidth() == 2 && tt.table[0] && !tt.table[1];
    };


    // Walk modules (skip library bodies themselves; only rewrite design).
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (module->hasAttr("synth.mapping_cost") ||
          module->hasAttr("synth.liberty.cell"))
        continue;
      auto *body = module.getBodyBlock();
      if (!body)
        continue;
      // Collect first (mutation invalidates iteration).
      SmallVector<hw::InstanceOp> invs;
      for (auto inst : body->getOps<hw::InstanceOp>()) {
        auto *target = symbols.lookup(inst.getModuleName());
        auto cellModule = dyn_cast_or_null<hw::HWModuleOp>(target);
        if (!cellModule || !isCombinationalCell(cellModule))
          continue;
        unsigned numOutputs = 0;
        for (const auto &port : cellModule.getPortList()) {
          if (port.isOutput())
            ++numOutputs;
        }
        if (numOutputs != 1)
          continue;
        auto tt = getTruthTable(cellModule);
        if (failed(tt) || !isNot(*tt))
          continue;
        invs.push_back(inst);
      }
      for (auto inv : invs) {
        if (inv->use_empty())
          continue;
        Value driver = inv.getOperand(0);
        if (!driver.hasOneUse())
          continue;
        auto driverOp = driver.getDefiningOp<hw::InstanceOp>();
        if (!driverOp || driverOp.getNumResults() != 1)
          continue;
        auto *driverTarget = symbols.lookup(driverOp.getModuleName());
        auto driverModule = dyn_cast_or_null<hw::HWModuleOp>(driverTarget);
        if (!driverModule || !isCombinationalCell(driverModule))
          continue;
        auto driverTt = getTruthTable(driverModule);
        if (failed(driverTt) || driverTt->numOutputs != 1)
          continue;
        auto driverCost = driverModule->getAttrOfType<MappingCostAttr>(
            "synth.mapping_cost");
        auto invTarget = symbols.lookup(inv.getModuleName());
        auto invModule = dyn_cast_or_null<hw::HWModuleOp>(invTarget);
        auto invCost = invModule
                           ? invModule->getAttrOfType<MappingCostAttr>(
                                 "synth.mapping_cost")
                           : MappingCostAttr();
        if (!driverCost || !driverCost.getArea() || !invCost ||
            !invCost.getArea())
          continue;
        double combined =
            driverCost.getArea().getValueAsDouble() +
            invCost.getArea().getValueAsDouble();
        // Complement function with identical port sequence.
        BinaryTruthTable need = driverTt->applyOutputNegation(1);
        SmallVector<StringAttr> driverPorts;
        for (const auto &port : driverModule.getPortList()) {
          if (port.isInput())
            driverPorts.push_back(port.name);
        }
        hw::HWModuleOp best = nullptr;
        double bestArea = combined;
        auto it = cellsByTable.find(tableKey(need));
        if (it == cellsByTable.end())
          continue;
        int64_t driverDelay = maxIntrinsic(driverModule);
        int64_t invDelay = 0;
        if (invModule) {
          invDelay = maxIntrinsic(invModule);
        }
        for (const auto &cand : it->second) {
          if (cand.ports != driverPorts)
            continue;
          if (cand.area < bestArea &&
              maxIntrinsic(cand.module) <= driverDelay + invDelay) {
            bestArea = cand.area;
            best = cand.module;
          }
        }
        if (!best)
          continue;
        OpBuilder builder(inv);
        SmallVector<Value> inputs(driverOp.getOperands().begin(),
                                  driverOp.getOperands().end());
        auto replacement = hw::InstanceOp::create(
            builder, inv.getLoc(), best.getOperation(),
            StringAttr::get(top.getContext(), inv.getInstanceName()), inputs);
        inv.getResult(0).replaceAllUsesWith(replacement.getResult(0));
        inv.erase();
        if (driverOp->use_empty())
          driverOp.erase();
        ++numAbsorbed;
      }
    }

  }
};

} // namespace
