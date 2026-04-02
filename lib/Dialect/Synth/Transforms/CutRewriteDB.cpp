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

std::string circt::synth::normalizeCutRewriteDatabaseKind(StringRef kind) {
  std::string normalized = kind.lower();
  for (char &c : normalized)
    if (c == '_')
      c = '-';
  return normalized;
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
  auto kindAttr = dbModule->getAttrOfType<StringAttr>(kCutRewriteDBKindAttr);
  if (!kindAttr)
    return dbModule.emitError("cut-rewrite database missing '")
           << kCutRewriteDBKindAttr << "'";

  database.kind = normalizeCutRewriteDatabaseKind(kindAttr.getValue());

  auto *backend = getCutRewriteDatabaseBackend(database.kind);
  if (!backend)
    return dbModule.emitError("unsupported cut-rewrite database kind '")
           << kindAttr.getValue() << "'";

  database.entries.clear();
  database.maxInputSize = 0;

  for (auto hwModule : dbModule.getOps<hw::HWModuleOp>()) {
    auto entry = backend->parseEntry(hwModule);
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
    loadedMaxInputSize = maxCutInputSize;
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
    if (failed(loadCutRewriteDatabaseFromModule(**parsedModule, *database)))
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
  std::shared_ptr<const LoadedCutRewriteDatabase> loadedDatabase;
  unsigned loadedMaxInputSize = 0;
};

} // namespace
