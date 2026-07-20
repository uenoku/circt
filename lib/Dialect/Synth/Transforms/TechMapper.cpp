//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TechMapper pass, which performs technology mapping
// by converting logic network representations (AIG operations) into
// technology-specific gate implementations using cut-based rewriting.
//
// The pass uses a cut-based algorithm with priority cuts and NPN canonical
// forms for efficient pattern matching. Declarative `cut_rewrite_pattern`
// operations describe the available implementations and their costs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <atomic>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_TECHMAPPER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-tech-mapper"

namespace {
struct TechMapperPass : public impl::TechMapperBase<TechMapperPass> {
  using TechMapperBase<TechMapperPass>::TechMapperBase;

  LogicalResult initialize(MLIRContext *context) override {
    (void)context;
    npnTable = std::make_shared<const NPNTable>();
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::SymbolTable symbolTable(module);

    SmallVector<std::unique_ptr<CutRewritePattern>> libraryPatterns;
    DenseSet<StringAttr> implementationModules;

    unsigned maxInputSize = 0;
    for (auto patternOp : module.getOps<CutRewritePatternOp>()) {
      patternOp.walk([&](hw::InstanceOp instanceOp) {
        implementationModules.insert(instanceOp.getReferencedModuleNameAttr());
      });
      auto pattern =
          createCutRewritePattern(patternOp, npnTable.get(), &symbolTable);
      if (failed(pattern)) {
        signalPassFailure();
        return;
      }
      maxInputSize =
          std::max(maxInputSize, patternOp.getFunctionType().getNumInputs());
      libraryPatterns.push_back(std::move(*pattern));
    }

    if (libraryPatterns.empty())
      return markAllAnalysesPreserved();

    SmallVector<hw::HWModuleOp> modules;
    for (auto hwModule : module.getOps<hw::HWModuleOp>())
      if (!implementationModules.contains(hwModule.getModuleNameAttr()))
        modules.push_back(hwModule);

    CutRewritePatternSet patternSet(std::move(libraryPatterns));
    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = maxInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.attachDebugTiming = test;
    options.npnTable = npnTable.get();
    std::atomic<uint64_t> numCutsCreatedCount = 0;
    std::atomic<uint64_t> numCutSetsCreatedCount = 0;
    std::atomic<uint64_t> numCutsRewrittenCount = 0;
    auto result = mlir::failableParallelForEach(
        module.getContext(), modules, [&](hw::HWModuleOp hwModule) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Processing module: " << hwModule.getName() << "\n");
          CutRewriter rewriter(options, patternSet);
          if (failed(rewriter.run(hwModule)))
            return failure();
          const auto &stats = rewriter.getStats();
          numCutsCreatedCount.fetch_add(stats.numCutsCreated,
                                        std::memory_order_relaxed);
          numCutSetsCreatedCount.fetch_add(stats.numCutSetsCreated,
                                           std::memory_order_relaxed);
          numCutsRewrittenCount.fetch_add(stats.numCutsRewritten,
                                          std::memory_order_relaxed);
          return success();
        });
    if (failed(result))
      signalPassFailure();
    numCutsCreated += numCutsCreatedCount;
    numCutSetsCreated += numCutSetsCreatedCount;
    numCutsRewritten += numCutsRewrittenCount;
  }

private:
  std::shared_ptr<const NPNTable> npnTable;
};

} // namespace
