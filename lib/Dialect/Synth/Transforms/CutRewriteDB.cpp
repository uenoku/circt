//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the file-backed exact-synthesis cut-rewrite pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "ExactSynthesisImpl.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "synth-cut-rewrite"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_CUTREWRITE
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace {

struct ExactSynthesisPattern : public CutRewritePattern {
  ExactSynthesisPattern(MLIRContext *context,
                        const LoadedExactSynthesisEntry &entry)
      : CutRewritePattern(context), entry(entry) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    const auto &logicNetwork = enumerator.getLogicNetwork();
    if (cut.isTrivialCut() || cut.getOutputSize(logicNetwork) != 1)
      return std::nullopt;

    auto *rootOp = logicNetwork.getGate(cut.getRootIndex()).getOperation();
    if (!rootOp || !rootOp->getResult(0).getType().isInteger(1))
      return std::nullopt;

    for (uint32_t inputIndex : cut.inputs) {
      if (logicNetwork.getGate(inputIndex).getKind() ==
          LogicNetworkGate::Constant)
        continue;
      Value inputValue = logicNetwork.getValue(inputIndex);
      if (!inputValue || !inputValue.getType().isInteger(1))
        return std::nullopt;
    }

    if (!(cut.getNPNClass().truthTable == entry.npnClass.truthTable))
      return std::nullopt;
    return MatchResult(entry.area, entry.delay);
  }

  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(entry.npnClass);
    return true;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    return entry.rewrite(builder, enumerator, cut);
  }

  unsigned getNumOutputs() const override { return 1; }
  StringRef getPatternName() const override { return entry.moduleName; }

private:
  const LoadedExactSynthesisEntry &entry;
};

struct CutRewritePass
    : public circt::synth::impl::CutRewriteBase<CutRewritePass> {
  using circt::synth::impl::CutRewriteBase<CutRewritePass>::CutRewriteBase;

  LogicalResult initialize(MLIRContext *context) override {
    loadedMaxInputSize = maxCutInputSize;
    loadedFileDatabase.reset();
    if (dbFile.empty()) {
      emitError(UnknownLoc::get(context))
          << "synth-cut-rewrite requires 'db-file'";
      return failure();
    }

    auto parsedModule = parseCutRewriteDBFile(dbFile, context);
    if (failed(parsedModule))
      return failure();

    auto database = std::make_shared<LoadedExactSynthesisDatabase>();
    if (failed(loadExactSynthesisDatabaseFromModule(**parsedModule, *database)))
      return failure();
    loadedMaxInputSize = database->maxInputSize;
    loadedFileDatabase = std::move(database);
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns;
    assert(loadedFileDatabase && "file database must be initialized");
    for (const auto &entry : loadedFileDatabase->entries)
      patterns.push_back(std::make_unique<ExactSynthesisPattern>(
          module->getContext(), *entry));

    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize =
        std::min(maxCutInputSize.getValue(), loadedMaxInputSize);
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.allowNoMatch = true;
    options.attachDebugTiming = test;

    CutRewritePatternSet patternSet(std::move(patterns));
    CutRewriter rewriter(options, patternSet);
    if (failed(rewriter.run(module)))
      return signalPassFailure();

    const auto &stats = rewriter.getStats();
    numCutsCreated += stats.numCutsCreated;
    numCutSetsCreated += stats.numCutSetsCreated;
    numCutsRewritten += stats.numCutsRewritten;
  }

private:
  std::shared_ptr<LoadedExactSynthesisDatabase> loadedFileDatabase;
  unsigned loadedMaxInputSize = 0;
};

} // namespace
