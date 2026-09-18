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
        classified.push_back(
            {index, root, {leaves.begin(), leaves.end()}, leafIdxs, gateClass});
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
    for (size_t groupBegin = 0; groupBegin < classified.size();) {
      size_t groupEnd = groupBegin + 1;
      while (groupEnd < classified.size() &&
             classified[groupEnd].leafIdxs == classified[groupBegin].leafIdxs)
        ++groupEnd;
      changed |= pairGroup(module, classified, groupBegin, groupEnd, bundled);
      groupBegin = groupEnd;
    }
    return changed;
  }

  /// Pair XOR3+MAJ3 (full adders) and XOR2+AND2 (half adders) within one
  /// leaf group. First pair wins per root (deterministic).
  bool pairGroup(hw::HWModuleOp module, ArrayRef<ClassifiedCut> classified,
                 size_t begin, size_t end, DenseSet<Value> &bundled) {
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
                       fullAdder);
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
    tryPair(bundle::GateClass::Xor2, bundle::GateClass::And2,
            /*fullAdder=*/false);
    return changed;
  }

  void createBundle(hw::HWModuleOp module, Value sum, Value carry,
                    bool fullAdder) {
    auto *block = module.getBodyBlock();
    OpBuilder builder(block, block->getTerminator()->getIterator());
    synth::CreateBundleOp::create(builder, sum.getLoc(),
                                  TypeRange{sum.getType(), carry.getType()},
                                  ValueRange{sum, carry});
    ++numBundles;
    if (fullAdder)
      ++numFullAdders;
    else
      ++numHalfAdders;
  }
};

} // namespace
