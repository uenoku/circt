//===- StripTimingProbes.cpp - Erase timing probes ------------------------*-
//C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Explicit probe-lifetime terminator: replaces every `synth.timing_probe`
// result with its input and erases the probe, so nothing downstream
// (stripping, export) ever observes one. With no probes present the pass is
// a no-op. All ordering is file/IR order.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/RefinementRegions.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace synth {
#define GEN_PASS_DECL_STRIPTIMINGPROBES
#define GEN_PASS_DEF_STRIPTIMINGPROBES
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

#define DEBUG_TYPE "synth-strip-timing-probes"

namespace {

struct StripTimingProbesPass
    : public circt::synth::impl::StripTimingProbesBase<StripTimingProbesPass> {
  using StripTimingProbesBase<StripTimingProbesPass>::StripTimingProbesBase;

  void runOnOperation() override {
    bool changed = false;
    getOperation()->walk([&](hw::HWModuleOp module) {
      if (!module.getBodyBlock())
        return WalkResult::advance();
      // Collect first; erasure mutates the block.
      SmallVector<synth::TimingProbeOp> probes;
      SmallVector<Operation *> recorded;
      for (auto &op : module.getBodyBlock()->getOperations()) {
        if (auto probe = dyn_cast<synth::TimingProbeOp>(op))
          probes.push_back(probe);
        else if (op.hasAttr(synth::refinePolicyAttrName()))
          recorded.push_back(&op);
      }
      for (auto probe : probes) {
        probe.getResult().replaceAllUsesWith(probe.getInput());
        probe->erase();
        ++numProbesStripped;
        changed = true;
      }
      // Transient refinement records never survive stripping.
      for (Operation *op : recorded) {
        op->removeAttr(synth::refinePolicyAttrName());
        ++numRefineAttrsRemoved;
        changed = true;
      }
      return WalkResult::advance();
    });
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
