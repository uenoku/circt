//===- LinkLiberty.cpp - Link Liberty files and propagate metadata --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LinkLiberty pass which:
//   1. Parses Liberty files in parallel into nested ModuleOps via
//      `linkLibertyFiles()`.
//   2. Copies `synth.liberty.library` from each nested ModuleOp down to every
//      `hw.module` cell contained within it, providing per-cell provenance.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ImportLiberty.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_LINKLIBERTY
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

struct LinkLibertyPass
    : public circt::synth::impl::LinkLibertyBase<LinkLibertyPass> {
  using LinkLibertyBase::LinkLibertyBase;

  void runOnOperation() override {
    ModuleOp topModule = getOperation();

    if (libertyFiles.empty())
      return;

    // Parse Liberty files in parallel into nested ModuleOps. The function
    // also checks `synth.nldm.time_unit` consistency and sets the merged value
    // on `topModule`.
    SmallVector<StringRef> fileRefs;
    fileRefs.reserve(libertyFiles.size());
    for (const auto &f : libertyFiles)
      fileRefs.push_back(f);

    if (failed(circt::liberty::linkLibertyFiles(fileRefs, &getContext(),
                                                topModule))) {
      signalPassFailure();
      return;
    }

    // Propagate `synth.liberty.library` from each nested ModuleOp to every
    // hw.module cell inside it, providing per-cell provenance.
    for (auto &op : *topModule.getBody()) {
      auto nestedModule = dyn_cast<ModuleOp>(&op);
      if (!nestedModule)
        continue;

      auto libAttr =
          nestedModule->getAttrOfType<DictionaryAttr>("synth.liberty.library");
      if (!libAttr)
        continue;

      nestedModule.walk([&](hw::HWModuleOp cellOp) {
        cellOp->setAttr("synth.liberty.library", libAttr);
      });
    }
  }
};

} // namespace
