//===- SizeCritical.cpp - Upsize cells on violated paths ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Upsizes combinational cells on timing-violated paths to stronger drives
// of the same function. Guidance comes from an OpenTimer result JSON with
// path traces (only violated endpoints trace); each path point resolves to
// a cell instance by hierarchical path. Stronger means larger mapping_cost
// area at minimal increase (same function by truth table, identical port
// sequences); sequential cells never resize. Single pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include <limits>
#include <map>
#include <string>
#include "llvm/Support/JSON.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_SIZECRITICAL
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-size-critical"

namespace {

// Same combinational-cell check as bubble pushing (truth tables need it).
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
    if (ns == "synth" && op->getName().getStringRef().contains(".aig."))
      return;
    if (ns != "comb" && ns != "arith" && ns != "hw")
      ok = false;
  });
  return ok;
}

struct SizeCriticalPass : public impl::SizeCriticalBase<SizeCriticalPass> {
  using impl::SizeCriticalBase<SizeCriticalPass>::SizeCriticalBase;
  void runOnOperation() override {
    auto top = getOperation();
    SymbolTable symbols(top);
    auto file = llvm::MemoryBuffer::getFile(staLog.getValue());
    if (!file) {
      top->emitError("cannot read STA report file '")
          << staLog.getValue() << "'";
      return signalPassFailure();
    }
    // Parse report_checks paths: Startpoint blocks with hierarchical pin
    // rows (Delay, Time, Description) and trailing slack. Collects
    // (pins, slackNs) per path; robust to field order (first slack wins).
    struct StaPath {
      SmallVector<std::string> pins;
      double slackNs = 0.0;
      bool hasSlack = false;
    };
    SmallVector<StaPath> staPaths;
    {
      StaPath cur;
      bool inPath = false;
      SmallVector<StringRef> lines;
      StringRef(file.get()->getBuffer()).split(lines, '\n');
      auto push = [&] {
        if (inPath && cur.hasSlack) {
          staPaths.push_back(cur);
          inPath = false;
        }
      };
      for (StringRef line : lines) {
        StringRef t = line.trim();
        if (t.starts_with("Startpoint:")) {
          push();
          cur = StaPath();
          inPath = true;
          continue;
        }
        // Pin row: delay, time, dir?, hier-name, (cell). Heuristic: a token
        // with '/' is a hierarchical pin.
        if (inPath) {
          SmallVector<StringRef> toks;
          t.split(toks, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
          for (StringRef tok : toks) {
            // Strip trailing "(cell)" and direction marks.
            if (tok.contains('/')) {
              std::string pin = tok.str();
              // Remove parenthesized cell suffix if glued? Tokens split
              // by space already separate "(cell)".
              if (!pin.empty() && pin[0] != '(')
                cur.pins.push_back(pin);
            }
          }
          // Slack line: "<num> slack (MET|VIOLATED)".
          auto slackPos = t.find("slack");
          if (slackPos != StringRef::npos) {
            double v = 0.0;
            // Number precedes "slack".
            StringRef before = t.substr(0, slackPos).trim();
            SmallVector<StringRef> nums;
            before.split(nums, ' ', -1, false);
            if (!nums.empty() &&
                !nums.back().getAsDouble(v)) {
              cur.slackNs = v;
              cur.hasSlack = true;
            }
          }
        }
      }
      push();
    }

    // Library cells by truth table (combinational, with costs).
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
      auto cost =
          module->getAttrOfType<MappingCostAttr>("synth.mapping_cost");
      if (!cost || !cost.getArea() || !isCombinationalCell(module))
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

    auto topMod = dyn_cast<hw::HWModuleOp>(
        symbols.lookup(topModuleName.getValue()));
    if (!topMod) {
      top->emitError("top module not found");
      return signalPassFailure();
    }
    // Resolve hierarchical instance path (names below top) to InstanceOp.
    std::function<hw::InstanceOp(hw::HWModuleOp, unsigned)> walk;
    // Iterative walk using explicit stack (paths are short).
    auto findInstance = [&](ArrayRef<mlir::StringAttr> path) {
      hw::HWModuleOp scope = topMod;
      hw::InstanceOp found;
      for (unsigned idx = 0; idx < path.size(); ++idx) {
        auto name = path[idx];
        found = {};
        auto *body = scope.getBodyBlock();
        if (!body)
          return hw::InstanceOp();
        for (auto inst : body->getOps<hw::InstanceOp>()) {
          if (inst.getInstanceName() == name.getValue()) {
            if (found)
              return hw::InstanceOp(); // Ambiguous: skip.
            found = inst;
          }
        }
        if (!found)
          return hw::InstanceOp();
        // Terminal may be a library extern (no body to descend into).
        if (idx + 1 == path.size())
          return found;
        auto *next = symbols.lookup(found.getModuleName());
        auto nextMod = dyn_cast_or_null<hw::HWModuleOp>(next);
        if (!nextMod)
          return hw::InstanceOp();
        scope = nextMod;
      }
      return found;
    };

    // Cells on violated paths (slack below threshold, default violated).
    int64_t threshold =
        slackThresholdPs.getValue() < 0.0
            ? 0
            : llround(slackThresholdPs.getValue());
    DenseSet<hw::InstanceOp> critical;
    for (const auto &path : staPaths) {
      if (!path.hasSlack)
        continue;
      if (llround(path.slackNs * 1000.0) >= threshold)
        continue;
      for (const auto &pin : path.pins) {
        // Hierarchical pin "a/b/C": instances a/b (pin C dropped).
        SmallVector<StringRef> segs;
        StringRef(pin).split(segs, '/');
        if (segs.size() < 2)
          continue;
        SmallVector<mlir::StringAttr> ipath;
        for (unsigned i = 0; i + 1 < segs.size(); ++i)
          ipath.push_back(StringAttr::get(top.getContext(), segs[i]));
        auto inst = findInstance(ipath);
        if (inst)
          critical.insert(inst);
      }
    }
    unsigned resized = 0;
    for (auto inst : critical) {
      auto *target = symbols.lookup(inst.getModuleName());
      auto cellModule = dyn_cast_or_null<hw::HWModuleOp>(target);
      if (!cellModule || !isCombinationalCell(cellModule))
        continue;
      if (inst.getNumResults() != 1)
        continue;
      auto tt = getTruthTable(cellModule);
      if (failed(tt) || tt->numOutputs != 1)
        continue;
      auto cost = cellModule->getAttrOfType<MappingCostAttr>(
          "synth.mapping_cost");
      if (!cost || !cost.getArea())
        continue;
      double curArea = cost.getArea().getValueAsDouble();
      SmallVector<StringAttr> ports;
      for (const auto &port : cellModule.getPortList()) {
        if (port.isInput())
          ports.push_back(port.name);
      }
      // Stronger = larger area at minimal increase (same function, ports).
      hw::HWModuleOp best = nullptr;
      double bestArea = std::numeric_limits<double>::infinity();
      auto it = cellsByTable.find(tableKey(*tt));
      if (it == cellsByTable.end())
        continue;
      for (const auto &cand : it->second) {
        if (cand.ports != ports || cand.area <= curArea)
          continue;
        if (cand.area < bestArea) {
          bestArea = cand.area;
          best = cand.module;
        }
      }
      if (!best)
        continue;
      OpBuilder builder(inst);
      SmallVector<Value> inputs(inst.getOperands().begin(),
                                inst.getOperands().end());
      auto replacement = hw::InstanceOp::create(
          builder, inst.getLoc(), best.getOperation(),
          StringAttr::get(top.getContext(), inst.getInstanceName()), inputs);
      inst.getResult(0).replaceAllUsesWith(replacement.getResult(0));
      inst.erase();
      ++numResized;
    }
  }
};

} // namespace
