//===- ExtractBundles.cpp - Group values for joint mapping ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Enumerates small (2-3 input) cuts over the optimized network, classifies
// AND2/XOR2/MAJ3/XOR3 functions under input permutation, and groups cuts
// sharing a leaf set. XOR3+MAJ3 pairs on a triple (full adders) and XOR2+AND2
// pairs on a pair (half adders) become `synth.create_bundle` ops for joint
// multi-output mapping. Purely additive: discrete logic stays in place.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/BundleUtils.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_EXTRACTBUNDLES
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-extract-bundles"

namespace {

struct ClassifiedCut {
  uint32_t rootIdx;
  Value root;
  // Shared-support values in sorted leaf-index order (matches leafIdxs).
  SmallVector<Value, 4> leaves;
  // Sorted leaf indices: the deterministic group key (pointer order varies
  // per run and must never affect decisions).
  SmallVector<uint32_t, 4> leafIdxs;
  bundle::GateClass gateClass;
};

struct ExtractBundlesPass
    : public circt::synth::impl::ExtractBundlesBase<ExtractBundlesPass> {
  using ExtractBundlesBase<ExtractBundlesPass>::ExtractBundlesBase;

  void runOnOperation() override {
    bool changed = false;
    for (auto module : getOperation().getOps<hw::HWModuleOp>()) {
      // Externs have no body; library cells and mapping patterns are
      // matched against, never extracted from (a bundle inside a pattern
      // could otherwise rewrite to the pattern itself).
      if (isa<hw::HWModuleExternOp>(module) ||
          module->hasAttr("synth.liberty.cell") ||
          module->hasAttr("synth.dont_use") ||
          module->hasAttr("synth.mapping_cost"))
        continue;
      changed |= extractInModule(module);
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

  bool extractInModule(hw::HWModuleOp module) {
    CutRewriterOptions options;
    options.strategy = OptimizationStrategyTiming;
    options.maxCutInputSize = 3;
    options.maxCutSizePerRoot = 64;
    CutEnumerator enumerator(options);
    if (failed(enumerator.enumerateCuts(module)))
      return false;
    const LogicNetwork &network = enumerator.getLogicNetwork();

    // Deterministic node order (cut-set map iteration is unordered).
    SmallVector<uint32_t> indices;
    for (const auto &entry : enumerator.getCutSets())
      indices.push_back(entry.first);
    llvm::sort(indices);

    // Classify small cuts with known truth tables.
    SmallVector<ClassifiedCut> classified;
    for (uint32_t index : indices) {
      Value root = network.getValue(index);
      if (!root || !root.getType().isInteger(1))
        continue;
      auto *set = enumerator.getCutSets().lookup(index);
      if (!set)
        continue;
      for (Cut *cut : set->getCuts()) {
        if (cut->isTrivialCut())
          continue;
        const auto &truthTable = cut->getTruthTable();
        if (!truthTable || truthTable->numInputs < 2 ||
            truthTable->numInputs > 3)
          continue;
        if (truthTable->numOutputs != 1 || truthTable->numInputs < 2 ||
            truthTable->numInputs > 3 || truthTable->table.getBitWidth() > 64)
          continue;
        bundle::GateClass gateClass = bundle::classifyTable(
            truthTable->table.getZExtValue(), truthTable->numInputs);
        if (gateClass == bundle::GateClass::None)
          continue;
        SmallVector<Value> leaves;
        network.getValues(cut->inputs, leaves);
        // Constant/null leaves and non-i1 leaves are out of scope.
        bool clean = true;
        for (Value leaf : leaves)
          if (!leaf || !leaf.getType().isInteger(1))
            clean = false;
        if (!clean)
          continue;
        SmallVector<uint32_t, 4> leafIdxs(cut->inputs.begin(),
                                          cut->inputs.end());
        llvm::sort(leafIdxs);
        // Order values by sorted leaf index so paired cuts share an
        // identical leaf sequence for the bundle record.
        DenseMap<uint32_t, Value> idxToValue;
        for (size_t k = 0; k < cut->inputs.size(); ++k)
          idxToValue[cut->inputs[k]] = leaves[k];
        SmallVector<Value, 4> sortedLeaves;
        for (uint32_t idx : leafIdxs)
          sortedLeaves.push_back(idxToValue[idx]);
        classified.push_back({index, root,
                              {sortedLeaves.begin(), sortedLeaves.end()},
                              leafIdxs, gateClass});
      }
    }

    // Group by leaf set: sort first (cut-set map iteration is unordered).
    // Pointer order is deterministic for a fixed input.
    llvm::sort(classified, [](const ClassifiedCut &a, const ClassifiedCut &b) {
      if (a.leafIdxs.size() != b.leafIdxs.size())
        return a.leafIdxs.size() < b.leafIdxs.size();
      for (size_t i = 0; i < a.leafIdxs.size(); ++i)
        if (a.leafIdxs[i] != b.leafIdxs[i])
          return a.leafIdxs[i] < b.leafIdxs[i];
      return a.rootIdx < b.rootIdx;
    });
    bool changed = false;
    DenseSet<Value> bundled;
    // Two-phase pairing: full adders first across ALL groups, then half
    // adders. Groups are visited smallest-first, so single-pass pairing
    // would let half-adder pairs in small groups steal roots that form
    // full adders in larger groups (e.g. a sum consumed via an XOR2
    // sub-cut). Full adders strictly dominate, so they claim roots first.
    for (int phase = 0; phase < 2; ++phase) {
      for (size_t groupBegin = 0; groupBegin < classified.size();) {
        size_t groupEnd = groupBegin + 1;
        while (groupEnd < classified.size() &&
               classified[groupEnd].leafIdxs == classified[groupBegin].leafIdxs)
          ++groupEnd;
        changed |= pairGroup(module, classified, groupBegin, groupEnd, bundled,
                             /*fullAddersOnly=*/phase == 0);
        groupBegin = groupEnd;
      }
    }
    return changed;
  }

  /// Pair XOR3+MAJ3 (full adders) and XOR2+AND2 (half adders) within one
  /// leaf group. First pair wins per root (deterministic). When
  /// fullAddersOnly is set, half-adder pairing is deferred to a later
  /// phase so full adders claim shared roots first.
  bool pairGroup(hw::HWModuleOp module, ArrayRef<ClassifiedCut> classified,
                 size_t begin, size_t end, DenseSet<Value> &bundled,
                 bool fullAddersOnly) {
    bool changed = false;
    auto tryPair = [&](bundle::GateClass first, bundle::GateClass second,
                       bool fullAdder) {
      for (size_t i = begin; i < end; ++i) {
        if (classified[i].gateClass != first ||
            bundled.contains(classified[i].root))
          continue;
        for (size_t j = begin; j < end; ++j) {
          if (classified[j].gateClass != second ||
              bundled.contains(classified[j].root) ||
              classified[j].root == classified[i].root)
            continue;
          createBundle(module, classified[i].root, classified[j].root,
                       classified[i].leaves, fullAdder);
          bundled.insert(classified[i].root);
          bundled.insert(classified[j].root);
          changed = true;
          break;
        }
      }
    };
    // XOR-first ordering keeps sum before carry in the bundle.
    tryPair(bundle::GateClass::Xor3, bundle::GateClass::Maj3,
            /*fullAdder=*/true);
    // Complemented-carry full adders (XOR3 sum + minority carry) map to
    // cells like ASAP7 FAx1 whose carry output is negated; MapBundles
    // re-derives joint tables, so no polarity bookkeeping is needed here.
    tryPair(bundle::GateClass::Xor3, bundle::GateClass::Minority3,
            /*fullAdder=*/true);
    if (fullAddersOnly)
      return changed;
    tryPair(bundle::GateClass::Xor2, bundle::GateClass::And2,
            /*fullAdder=*/false);
    return changed;
  }

  void createBundle(hw::HWModuleOp module, Value sum, Value carry,
                    ArrayRef<Value> leaves, bool fullAdder) {
    auto *block = module.getBodyBlock();
    OpBuilder builder(block, block->getTerminator()->getIterator());
    // Operands are the two roots followed by the shared support; results
    // pass everything through (leaf results are unused). Recording the
    // support lets mapping simulate the joint function over the paired
    // cut boundary instead of walking transitively past it.
    SmallVector<Value> operands{sum, carry};
    operands.append(leaves.begin(), leaves.end());
    SmallVector<Type> types{sum.getType(), carry.getType()};
    for (Value leaf : leaves)
      types.push_back(leaf.getType());
    synth::CreateBundleOp::create(builder, sum.getLoc(), TypeRange(types),
                                  ValueRange(operands));
    ++numBundles;
    if (fullAdder)
      ++numFullAdders;
    else
      ++numHalfAdders;
  }
};

} // namespace
