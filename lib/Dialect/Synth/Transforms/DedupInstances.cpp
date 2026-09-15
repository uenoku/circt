//===- DedupInstances.cpp - Merge duplicate cell instances -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DedupInstances pass, which merges exactly
// duplicate `hw.instance` ops left behind by technology mapping. Phase-aware
// mapping materializes one inverter cell per negated NPN binding after the
// last CSE, so shared signals fan out to many identical inverters (typically
// ~30% of all inverters, ~5% of all instances). Merging them is a pure win:
// same logic, fewer instances.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include <unordered_map>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_DEDUPINSTANCES
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-dedup-instances"

namespace {

// Key for exact-duplicate detection: same target cell, same SSA operands,
// same parameters.
struct DuplicateKey {
  FlatSymbolRefAttr target;
  SmallVector<Value> operands;
  Attribute parameters;

  bool operator==(const DuplicateKey &other) const {
    return target == other.target && operands == other.operands &&
           parameters == other.parameters;
  }
};

struct DuplicateKeyHash {
  size_t operator()(const DuplicateKey &key) const {
    return llvm::hash_combine(
        key.target, key.parameters,
        llvm::hash_combine_range(key.operands.begin(), key.operands.end()));
  }
};

/// Only pure-combinational signatures are eligible: every operand and
/// result has integer type, with at least one of each. This excludes
/// memories, hierarchical modules with aggregate ports, and physical-only
/// cells (filler/decap/tie with no inputs) whose multiplicity carries
/// physical intent. Sequential and hierarchical targets are excluded
/// separately via targetEligible (mapped clocks are plain i1, so the type
/// check alone cannot see them).
static bool hasCombinationalSignature(hw::InstanceOp inst) {
  if (inst->hasAttr("doNotPrint"))
    return false;
  // May be referenced by innerRef/XMR; merging would break the reference.
  if (inst.getInnerSymAttr())
    return false;
  if (inst.getNumOperands() == 0 || inst.getNumResults() == 0)
    return false;
  for (Value operand : inst.getOperands())
    if (!operand.getType().isInteger())
      return false;
  for (Value result : inst.getResults())
    if (!result.getType().isInteger())
      return false;
  return true;
}

/// Returns true if instances of the target may merge: the target is a known
/// leaf cell that is purely combinational. Specifically:
/// - unknown targets never merge (conservative);
/// - `synth.sram.macro` targets never merge (macro multiplicity is
///   physical);
/// - hierarchical targets (bodies containing `hw.instance`, i.e. design
///   modules and composed supergates) never merge, so hierarchy and flop
///   counts are preserved exactly;
/// - sequential cells never merge: liberty flop bodies hold a stateful
///   element (see SeqTechMapper::extractSignature), or a `clock = "true"`
///   liberty pin attribute marks a clock input. The metadata check survives
///   StripLibToExtern (port lists carry over) so this also works on externs.
///   Flop counts are exact-reconciled downstream, so flops must never merge
///   even when all inputs coincide.
/// Results are cached per target.
static bool targetEligible(hw::InstanceOp inst, ModuleOp top,
                           DenseMap<FlatSymbolRefAttr, bool> &cache) {
  FlatSymbolRefAttr target = inst.getModuleNameAttr();
  auto it = cache.find(target);
  if (it != cache.end())
    return it->second;
  bool eligible = false;
  if (Operation *targetOp = top.lookupSymbol(inst.getModuleName())) {
    eligible = !targetOp->hasAttr("synth.sram.macro");
    if (eligible) {
      bool hasState = false, hasInstance = false;
      targetOp->walk([&](Operation *op) {
        if (isa<synth::FFOp, seq::CompRegOp, seq::FirRegOp, seq::FirMemOp>(op))
          hasState = true;
        if (isa<hw::InstanceOp>(op))
          hasInstance = true;
      });
      eligible = !hasState && !hasInstance;
    }
    if (eligible) {
      auto checkPorts = [&](ArrayRef<hw::PortInfo> ports) {
        for (const auto &port : ports) {
          if (!port.isInput() || !port.attrs)
            continue;
          auto pin = dyn_cast_or_null<DictionaryAttr>(
              port.attrs.get("synth.liberty.pin"));
          if (!pin)
            continue;
          if (auto clock = dyn_cast_or_null<StringAttr>(pin.get("clock")))
            if (clock.getValue() == "true")
              eligible = false;
        }
      };
      if (auto hwMod = dyn_cast<hw::HWModuleOp>(targetOp))
        checkPorts(hwMod.getPortList());
      else if (auto ext = dyn_cast<hw::HWModuleExternOp>(targetOp))
        checkPorts(ext.getPortList());
    }
  }
  cache[target] = eligible;
  return eligible;
}

static bool isMergable(hw::InstanceOp inst, ModuleOp top,
                       DenseMap<FlatSymbolRefAttr, bool> &targetCache) {
  return hasCombinationalSignature(inst) &&
         targetEligible(inst, top, targetCache);
}

struct DedupInstancesPass
    : public circt::synth::impl::DedupInstancesBase<DedupInstancesPass> {
  using DedupInstancesBase<DedupInstancesPass>::DedupInstancesBase;

  void runOnOperation() override {
    bool changed = false;
    DenseMap<FlatSymbolRefAttr, bool> targetCache;
    ModuleOp top = getOperation();
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      // Externs have no body.
      if (isa<hw::HWModuleExternOp>(module.getOperation()))
        continue;
      for (auto &block : module->getRegion(0))
        changed |= dedupBlock(block, top, targetCache);
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

  bool dedupBlock(Block &block, ModuleOp top,
                  DenseMap<FlatSymbolRefAttr, bool> &targetCache) {
    // Operands are block-scoped SSA values, so the map is per block.
    // Instances cannot nest, so direct iteration suffices.
    std::unordered_map<DuplicateKey, hw::InstanceOp, DuplicateKeyHash> seen;
    SmallVector<std::pair<hw::InstanceOp, hw::InstanceOp>> merges;
    for (auto &op : block) {
      auto inst = dyn_cast<hw::InstanceOp>(op);
      if (!inst || !isMergable(inst, top, targetCache))
        continue;
      DuplicateKey key{inst.getModuleNameAttr(),
                       SmallVector<Value>(inst.getOperands()),
                       inst.getParameters()};
      auto [it, inserted] = seen.try_emplace(key, inst);
      if (inserted)
        continue;
      // Same target implies the same signature, but check defensively.
      if (it->second.getNumResults() != inst.getNumResults())
        continue;
      merges.emplace_back(inst, it->second);
    }
    for (auto [dup, first] : merges) {
      dup.replaceAllUsesWith(first.getResults());
      dup.erase();
      ++numMerged;
    }
    return !merges.empty();
  }
};

} // namespace
