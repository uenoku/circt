//===- RestructureMuxTrees.cpp - Rephase mux trees for 4:1 collapse -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RestructureMuxTrees pass, which rephases
// shared-select mux trees in `synth.aig.and_inv` form so that 4:1 cones
// present clean (non-inverted) mux4 truth tables to technology mapping.
//
// AIG lowering absorbs each mux's output inverter into its consumer, so mux
// trees arrive at mapping with alternating phases: a mux triple whose data
// arms are both consumed inverted implements an inverted mux, and a 4:1 cone
// over two such triples implements an inverted mux4 that costs an extra
// discrete inverter (there is no inverted-output mux4 cell). The mapper
// then fuses only the clean levels (shifted pairing), leaving the widest
// tree level unpacked. Pushing the inversion down through the triple (exact
// local equivalence: redefine the triple, toggle all its use-edges so every
// consumer observes identical values) concentrates inversions at cone
// boundaries, where module-output inverters absorb the top, and leaves
// interior cones clean for bottom-up 4:1 collapse. Pre-map node count is
// unchanged (edge polarities only).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_RESTRUCTUREMUXTREES
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-restructure-mux-trees"

namespace {

/// A 2:1 mux in AIG form: `y = and(~n0, ~n1)` with branch ANDs
/// `n0 = and(sel^a, data0^_)`, `n1 = and(sel^~a, data1^_)` sharing exactly
/// one value (the select) with opposite polarity and distinct data values.
struct MuxTriple {
  aig::AndInverterOp y;
  aig::AndInverterOp n0;
  aig::AndInverterOp n1;
  Value select;
  Value data0;
  Value data1;
  unsigned data0Idx;
  unsigned data1Idx;
};

static void setOperandInverted(aig::AndInverterOp op, unsigned idx, bool inv) {
  SmallVector<bool> flags;
  flags.reserve(op.getNumOperands());
  for (unsigned i = 0, e = op.getNumOperands(); i < e; ++i)
    flags.push_back(i == idx ? inv : op.isInverted(i));
  op.setInverted(flags);
}

static std::optional<MuxTriple> matchMuxTriple(aig::AndInverterOp y) {
  if (!y.getResult().getType().isInteger(1))
    return std::nullopt;
  // Output NAND over exactly two inverted inputs.
  if (y.getNumOperands() != 2 || !y.isInverted(0) || !y.isInverted(1))
    return std::nullopt;
  auto n0 =
      dyn_cast_or_null<aig::AndInverterOp>(y.getOperand(0).getDefiningOp());
  auto n1 =
      dyn_cast_or_null<aig::AndInverterOp>(y.getOperand(1).getDefiningOp());
  if (!n0 || !n1 || n0.getOperation() == n1.getOperation())
    return std::nullopt;
  // Branch values change, so they must be local to this triple.
  if (!n0->hasOneUse() || !n1->hasOneUse())
    return std::nullopt;
  if (n0.getNumOperands() != 2 || n1.getNumOperands() != 2)
    return std::nullopt;
  // Exactly one shared value driven with opposite polarity (the select);
  // the remaining data values must be distinct.
  bool found = false;
  MuxTriple triple;
  triple.y = y;
  triple.n0 = n0;
  triple.n1 = n1;
  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      if (n0.getOperand(i) != n1.getOperand(j))
        continue;
      if (n0.isInverted(i) == n1.isInverted(j))
        continue;
      Value data0 = n0.getOperand(1 - i);
      Value data1 = n1.getOperand(1 - j);
      if (data0 == data1)
        continue;
      if (found)
        return std::nullopt;
      found = true;
      triple.select = n0.getOperand(i);
      triple.data0 = data0;
      triple.data1 = data1;
      triple.data0Idx = 1 - i;
      triple.data1Idx = 1 - j;
    }
  }
  if (!found)
    return std::nullopt;
  return triple;
}

/// Design modules only: never rewrite library patterns, supergate
/// candidates, or stateful cells. Mirrors the TechMapper module filter.
static bool isDesignModule(hw::HWModuleOp module) {
  if (module->hasAttr("synth.mapping_cost") ||
      module->hasAttr("synth.liberty.cell") ||
      module->hasAttr("synth.dont_use") ||
      module->hasAttr("synth.supergate_candidate") ||
      module.getModuleName().starts_with("__supergate_"))
    return false;
  if (module.getBodyBlock()
          ->walk([](Operation *op) {
            if (isa<FFOp, LatchOp, ClockGateOp>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  return true;
}

struct RestructureMuxTreesPass
    : public circt::synth::impl::RestructureMuxTreesBase<
          RestructureMuxTreesPass> {
  using RestructureMuxTreesBase::RestructureMuxTreesBase;

  void runOnOperation() override {
    ModuleOp top = getOperation();
    bool changed = false;
    SmallPtrSet<Operation *, 32> countedTriples;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (!isDesignModule(module))
        continue;
      for (auto &block : module->getRegion(0))
        changed |= normalizeBlock(block, countedTriples);
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

  /// Push inversions down through mux triples whose data arms are both
  /// consumed inverted. Each rewrite is an exact local equivalence (the
  /// triple output is complemented while every use-edge is toggled, so all
  /// consumers observe identical values), hence iteration order affects
  /// only convergence, never correctness. Sweep to fixpoint with a small
  /// cap; straight-line SSA bodies converge in one or two sweeps.
  bool normalizeBlock(Block &block,
                      SmallPtrSetImpl<Operation *> &countedTriples) {
    bool changed = false;
    for (unsigned iter = 0; iter < 8; ++iter) {
      bool iterChanged = false;
      for (auto &op : llvm::make_early_inc_range(block)) {
        auto y = dyn_cast<aig::AndInverterOp>(op);
        if (!y)
          continue;
        auto triple = matchMuxTriple(y);
        if (!triple)
          continue;
        if (countedTriples.insert(y.getOperation()).second)
          ++numMuxTriples;
        // Only rephase triples consuming inverted data on both arms.
        if (!triple->n0.isInverted(triple->data0Idx) ||
            !triple->n1.isInverted(triple->data1Idx))
          continue;
        // Single-use keeps the rewrite forest-local: every consumer edge
        // is toggled, so all consumers observe identical values.
        if (!y->hasOneUse())
          continue;
        Operation *user = *y->user_begin();
        auto consumer = dyn_cast<aig::AndInverterOp>(user);
        if (!consumer)
          continue;
        setOperandInverted(triple->n0, triple->data0Idx, false);
        setOperandInverted(triple->n1, triple->data1Idx, false);
        for (unsigned i = 0, e = consumer.getNumOperands(); i < e; ++i)
          if (consumer.getOperand(i) == y.getResult())
            setOperandInverted(consumer, i, !consumer.isInverted(i));
        ++numRephased;
        iterChanged = true;
      }
      if (!iterChanged)
        break;
      changed = true;
    }
    return changed;
  }
};

} // namespace
