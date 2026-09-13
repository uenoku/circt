//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the StripLibToExtern pass, which replaces imported
// library `hw.module`s with `hw.module.extern` declarations. Mapping passes
// need the executable bodies; Verilog emission needs only declarations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_STRIPLIBTOEXTERN
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-strip-lib-to-extern"

namespace {

struct StripLibToExternPass
    : public circt::synth::impl::StripLibToExternBase<
          StripLibToExternPass> {
  using StripLibToExternBase<StripLibToExternPass>::StripLibToExternBase;

  void runOnOperation() override {
    auto top = getOperation();
    // Collect first; rewriting mutates the block.
    SmallVector<hw::HWModuleOp> libraries;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      // Design modules never carry these importer markers.
      if (module->hasAttr("synth.liberty.cell") ||
          module->hasAttr("synth.dont_use"))
        libraries.push_back(module);
    }
    if (libraries.empty())
      return markAllAnalysesPreserved();
    OpBuilder builder(top.getContext());
    for (auto module : libraries) {
      builder.setInsertionPoint(module);
      // Ports and parameters carry over; bodies and their megabytes of
      // Liberty metadata do not (mapping already consumed them).
      hw::HWModuleExternOp::create(builder, module.getLoc(),
                                    module.getModuleNameAttr(),
                                    module.getPortList(), StringRef(),
                                    module.getParameters());
      module.erase();
      ++numStripped;
    }
  }
};

} // namespace
