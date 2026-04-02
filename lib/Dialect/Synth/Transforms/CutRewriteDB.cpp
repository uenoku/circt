//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the file-backed cut-rewrite pass.
//
//===----------------------------------------------------------------------===//

#include "CutRewriteDBImpl.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

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

FailureOr<std::pair<double, SmallVector<DelayType>>>
circt::synth::getAreaAndDelayFromTechInfo(hw::HWModuleOp module) {
  auto techInfo = module->getAttrOfType<DictionaryAttr>("hw.techlib.info");
  if (!techInfo)
    return module.emitError("cut-rewrite database module missing "
                            "'hw.techlib.info'");

  auto areaAttr = techInfo.getAs<FloatAttr>("area");
  auto delayAttr = techInfo.getAs<ArrayAttr>("delay");
  if (!areaAttr || !delayAttr)
    return module.emitError("cut-rewrite database module must have "
                            "'area'(float) and 'delay'(array) in "
                            "'hw.techlib.info'");

  SmallVector<DelayType> delay;
  for (auto delayValue : delayAttr) {
    auto delayArray = dyn_cast<ArrayAttr>(delayValue);
    if (!delayArray)
      return module.emitError("cut-rewrite database delay entries must be "
                              "arrays");
    for (auto delayElement : delayArray) {
      auto intAttr = dyn_cast<IntegerAttr>(delayElement);
      if (!intAttr)
        return module.emitError("cut-rewrite database delay values must be "
                                "integers");
      delay.push_back(intAttr.getValue().getZExtValue());
    }
  }

  return std::make_pair(areaAttr.getValue().convertToDouble(),
                        std::move(delay));
}

FailureOr<OwningOpRef<mlir::ModuleOp>>
circt::synth::parseCutRewriteDBFile(StringRef dbFile, MLIRContext *context) {
  std::string errorMessage;
  auto input = mlir::openInputFile(dbFile, &errorMessage);
  if (!input) {
    emitError(UnknownLoc::get(context)) << errorMessage;
    return failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  auto parsedModule = parseSourceFile<mlir::ModuleOp>(sourceMgr, context);
  if (!parsedModule)
    return failure();
  return parsedModule;
}

LogicalResult circt::synth::loadCutRewriteDatabaseFromModule(
    mlir::ModuleOp dbModule, LoadedCutRewriteDatabase &database) {
  database.entries.clear();
  database.maxInputSize = 0;

  for (auto hwModule : dbModule.getOps<hw::HWModuleOp>()) {
    auto entry = parseCutRewriteEntry(hwModule);
    if (failed(entry))
      return failure();
    database.maxInputSize = std::max(
        database.maxInputSize,
        static_cast<unsigned>((*entry)->npnClass.truthTable.numInputs));
    database.entries.push_back(std::move(*entry));
  }

  if (database.entries.empty())
    return dbModule.emitError("cut-rewrite database did not contain any "
                              "matching library entries");
  return success();
}

namespace {

struct CutRewriteDatabasePattern : public CutRewritePattern {
  CutRewriteDatabasePattern(MLIRContext *context,
                            const LoadedCutRewriteEntry &entry)
      : CutRewritePattern(context), entry(entry) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    assert(cut.getNPNClass().truthTable == entry.npnClass.truthTable);
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
  const LoadedCutRewriteEntry &entry;
};

struct CutRewritePass
    : public circt::synth::impl::CutRewriteBase<CutRewritePass> {
  using circt::synth::impl::CutRewriteBase<CutRewritePass>::CutRewriteBase;

  LogicalResult initialize(MLIRContext *context) override {
    loadedMaxInputSize = 0;
    loadedDatabase.reset();
    if (dbFile.empty()) {
      emitError(UnknownLoc::get(context))
          << "synth-cut-rewrite requires 'db-file'";
      return failure();
    }

    auto parsedModule = parseCutRewriteDBFile(dbFile, context);
    if (failed(parsedModule))
      return failure();

    auto database = std::make_shared<LoadedCutRewriteDatabase>();
    database->backingModule = std::move(*parsedModule);
    if (failed(loadCutRewriteDatabaseFromModule(*database->backingModule,
                                                *database)))
      return failure();
    loadedMaxInputSize = database->maxInputSize;
    loadedDatabase = std::move(database);
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns;
    assert(loadedDatabase && "file database must be initialized");
    for (const auto &entry : loadedDatabase->entries)
      patterns.push_back(std::make_unique<CutRewriteDatabasePattern>(
          module->getContext(), *entry));

    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = loadedMaxInputSize;
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
  std::shared_ptr<const LoadedCutRewriteDatabase> loadedDatabase;
  unsigned loadedMaxInputSize = 0;
};

} // namespace
