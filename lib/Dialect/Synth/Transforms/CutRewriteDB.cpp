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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/AnalysisManager.h"
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

FailureOr<NPNClass> circt::synth::getNPNClassFromModule(hw::HWModuleOp module) {
  auto inputTypes = module.getInputTypes();
  auto outputTypes = module.getOutputTypes();

  unsigned numInputs = inputTypes.size();
  unsigned numOutputs = outputTypes.size();
  if (numOutputs != 1)
    return module->emitError(
        "modules with multiple outputs are not supported yet");

  for (auto type : inputTypes)
    if (!type.isInteger(1))
      return module->emitError("all input ports must be single bit");
  for (auto type : outputTypes)
    if (!type.isInteger(1))
      return module->emitError("all output ports must be single bit");

  if (numInputs > maxTruthTableInputs)
    return module->emitError("too many inputs for truth table generation");

  SmallVector<Value> results;
  results.reserve(numOutputs);
  auto *bodyBlock = module.getBodyBlock();
  assert(bodyBlock && "module must have a body block");
  for (auto result : bodyBlock->getTerminator()->getOperands())
    results.push_back(result);

  auto truthTable = getTruthTable(results, bodyBlock);
  if (failed(truthTable))
    return failure();
  return NPNClass::computeNPNCanonicalForm(*truthTable);
}

namespace {

static bool isUnsynthesizedTruthTableModule(hw::HWModuleOp module) {
  return llvm::any_of(module.getBodyBlock()->without_terminator(),
                      [](Operation &op) { return isa<comb::TruthTableOp>(op); });
}

static FailureOr<std::string> inferCutRewriteInverterKind(mlir::ModuleOp db) {
  bool sawMIG = false;
  bool sawAIG = false;
  for (auto module : db.getOps<hw::HWModuleOp>()) {
    for (Operation &op : module.getBodyBlock()->without_terminator()) {
      if (isa<synth::mig::MajorityInverterOp>(op))
        sawMIG = true;
      else if (isa<synth::aig::AndInverterOp>(op))
        sawAIG = true;
    }
  }

  // AIG inversion is the conservative fallback for ambiguous databases, and it
  // also covers mixed AIG/MIG bodies because later strashing can flatten them.
  if (sawAIG || !sawMIG)
    return std::string("aig");
  return std::string("mig");
}

static double computeMaterializedModuleArea(hw::HWModuleOp module) {
  double area = 0.0;
  for (Operation &op : module.getBodyBlock()->without_terminator())
    if (isa<synth::aig::AndInverterOp, synth::mig::MajorityInverterOp>(op))
      area += 1.0;
  return area;
}

static FailureOr<SmallVector<DelayType>>
computeMaterializedModuleInputDelays(hw::HWModuleOp module,
                                     LongestPathAnalysis &analysis) {
  auto output = dyn_cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
  if (!output || output.getNumOperands() != 1)
    return module.emitError("generated cut-rewrite database module must "
                            "terminate with a single-output hw.output");

  SmallVector<DataflowPath> paths;
  if (failed(analysis.computeGlobalPaths(output.getOperand(0), /*bitPos=*/0,
                                         paths)))
    return failure();

  SmallVector<DelayType> delays(module.getNumInputPorts(), 0);
  for (const auto &path : paths) {
    auto arg = dyn_cast<BlockArgument>(path.getStartPoint().value);
    if (!arg || arg.getParentBlock() != module.getBodyBlock())
      continue;
    delays[arg.getArgNumber()] =
        std::max(delays[arg.getArgNumber()],
                 static_cast<DelayType>(path.getDelay()));
  }
  return delays;
}

static FailureOr<CutRewriteModuleMetadata>
computeCutRewriteModuleMetadata(hw::HWModuleOp module) {
  CutRewriteModuleMetadata metadata;
  auto npnClass = getNPNClassFromModule(module);
  if (failed(npnClass))
    return failure();
  metadata.npnClass = std::move(*npnClass);

  if (module->hasAttr("hw.techlib.info")) {
    auto areaAndDelay = getAreaAndDelayFromTechInfo(module);
    if (failed(areaAndDelay))
      return failure();
    metadata.area = areaAndDelay->first;
    metadata.delay = std::move(areaAndDelay->second);
    return metadata;
  }

  mlir::ModuleAnalysisManager moduleAnalysisManager(
      module, /*passInstrumentor=*/nullptr);
  mlir::AnalysisManager analysisManager = moduleAnalysisManager;
  LongestPathAnalysis pathAnalysis(
      module, analysisManager,
      LongestPathAnalysisOptions(/*collectDebugInfo=*/false,
                                 /*lazyComputation=*/false,
                                 /*keepOnlyMaxDelayPaths=*/true));

  auto delays = computeMaterializedModuleInputDelays(module, pathAnalysis);
  if (failed(delays))
    return failure();
  metadata.area = computeMaterializedModuleArea(module);
  metadata.delay = std::move(*delays);
  return metadata;
}

} // namespace

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

  auto inferredInverterKind = inferCutRewriteInverterKind(dbModule);
  if (failed(inferredInverterKind))
    return failure();

  SmallVector<hw::HWModuleOp> hwModules(dbModule.getOps<hw::HWModuleOp>());
  std::vector<std::unique_ptr<LoadedCutRewriteEntry>> loadedEntries(
      hwModules.size());
  if (failed(mlir::failableParallelForEachN(
          dbModule.getContext(), 0, hwModules.size(), [&](size_t index) {
            if (isUnsynthesizedTruthTableModule(hwModules[index]))
              return success();
            auto metadata = computeCutRewriteModuleMetadata(hwModules[index]);
            if (failed(metadata))
              return failure();
            metadata->inverterKind = *inferredInverterKind;
            auto entry = parseCutRewriteEntry(hwModules[index], *metadata);
            if (failed(entry))
              return failure();
            loadedEntries[index] = std::move(*entry);
            return success();
          })))
    return failure();

  for (auto &entry : loadedEntries) {
    if (!entry)
      continue;
    database.maxInputSize =
        std::max(database.maxInputSize,
                 static_cast<unsigned>(entry->npnClass.truthTable.numInputs));
    database.entries.push_back(std::move(entry));
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
