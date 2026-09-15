//===- SortTopologically.cpp - Enforce SSA order in modules ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Technology mapping appends cell instances after their users (sequential
// cells are placed before the combinational drivers TechMapper creates
// later), so mapped modules violate SSA dominance order. ExportVerilog
// tolerates any order, but downstream MLIR consumers (notably circt-lec,
// which verifies inputs on load) do not. This pass stably sorts every
// non-extern module body topologically so definitions precede uses.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_SORTTOPOLOGICALLY
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-sort-topologically"

namespace {

struct SortTopologicallyPass
    : public circt::synth::impl::SortTopologicallyBase<SortTopologicallyPass> {
  using SortTopologicallyBase<SortTopologicallyPass>::SortTopologicallyBase;

  void runOnOperation() override {
    bool sawBody = false;
    for (auto module : getOperation().getOps<hw::HWModuleOp>()) {
      // Externs have no body.
      if (isa<hw::HWModuleExternOp>(module.getOperation()))
        continue;
      for (auto &block : module->getRegion(0)) {
        // Stable sort: already-ordered blocks are untouched. Cycles are
        // broken by the sorter (moved to the end in order) and counted.
        sawBody = true;
        if (mlir::sortTopologically(&block))
          ++numSorted;
        else
          ++numUnsortable;
      }
    }
    if (!sawBody)
      markAllAnalysesPreserved();
  }
};

} // namespace
