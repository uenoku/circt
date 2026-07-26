//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements cut rewriting from external implementation databases.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

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

struct DatabaseEntry {
  hw::HWModuleOp module;
  NPNClass npnClass;
  SmallVector<DelayType> delays;
  double area = 0.0;
};

struct CutRewriteDatabase {
  SmallVector<OwningOpRef<ModuleOp>> modules;
  SmallVector<DatabaseEntry> entries;
  Operation *inverterPrototype = nullptr;
  unsigned maxInputSize = 0;
};

static FailureOr<OwningOpRef<ModuleOp>>
parseDatabaseFile(StringRef filename, MLIRContext *context) {
  std::string errorMessage;
  auto input = openInputFile(filename, &errorMessage);
  if (!input) {
    emitError(UnknownLoc::get(context)) << "cannot open cut-rewrite database '"
                                        << filename << "': " << errorMessage;
    return failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!module)
    return failure();
  return module;
}

static LogicalResult validatePortTypes(hw::HWModuleOp module) {
  if (module.getNumInputPorts() == 0)
    return module.emitError("cut-rewrite database entries require an input");
  if (module.getNumOutputPorts() != 1)
    return module.emitError(
        "cut-rewrite database entries require exactly one output");
  if (module.getNumInputPorts() > 8)
    return module.emitError(
        "cut-rewrite database entries support at most 8 inputs");

  for (Type type : module.getInputTypes())
    if (!type.isInteger(1))
      return module.emitError(
          "cut-rewrite database input ports must have type i1");
  if (!module.getOutputTypes().front().isInteger(1))
    return module.emitError(
        "cut-rewrite database output ports must have type i1");
  return success();
}

static FailureOr<DatabaseEntry> loadDatabaseEntry(hw::HWModuleOp module,
                                                  Operation *&inverterPrototype,
                                                  const NPNTable &npnTable) {
  if (failed(validatePortTypes(module)))
    return failure();

  auto *body = module.getBodyBlock();
  auto output = dyn_cast<hw::OutputOp>(body->getTerminator());
  if (!output || output.getNumOperands() != 1)
    return module.emitError(
        "cut-rewrite database entry must end in one hw.output operand");

  unsigned numInputs = module.getNumInputPorts();
  DenseMap<Value, SmallVector<DelayType>> valueDelays;
  for (auto [inputIndex, argument] : llvm::enumerate(body->getArguments())) {
    SmallVector<DelayType> delays(numInputs, -1);
    delays[inputIndex] = 0;
    valueDelays.try_emplace(argument, std::move(delays));
  }

  double area = 0.0;
  for (Operation &op : body->without_terminator()) {
    if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
      valueDelays.try_emplace(constant.getResult(),
                              SmallVector<DelayType>(numInputs, -1));
      continue;
    }

    auto logicOp = dyn_cast<BooleanLogicOpInterface>(op);
    if (!logicOp)
      return op.emitError(
          "cut-rewrite database bodies only support hw.constant and "
          "BooleanLogicOpInterface operations");
    if (logicOp->getNumResults() != 1 ||
        !logicOp.getResult().getType().isInteger(1))
      return op.emitError(
          "cut-rewrite database logic operations must produce one i1 result");

    auto areaCost = logicOp.getLogicAreaCost();
    if (!areaCost)
      return op.emitError("operation does not provide a logic area cost");
    area += *areaCost;

    int64_t depthCost = logicOp.getLogicDepthCost();
    if (depthCost < 0)
      return op.emitError("operation has a negative logic depth cost");

    SmallVector<DelayType> resultDelays(numInputs, -1);
    for (Value operand : logicOp.getInputs()) {
      auto operandIt = valueDelays.find(operand);
      if (operandIt == valueDelays.end())
        return op.emitError(
            "operand is not defined by an earlier supported operation");
      for (unsigned i = 0; i != numInputs; ++i)
        if (operandIt->second[i] >= 0)
          resultDelays[i] =
              std::max(resultDelays[i], operandIt->second[i] + depthCost);
    }
    valueDelays.try_emplace(logicOp.getResult(), std::move(resultDelays));

    if (!inverterPrototype && logicOp.supportsNumInputs(1))
      inverterPrototype = logicOp.getOperation();
  }

  auto outputIt = valueDelays.find(output.getOperand(0));
  if (outputIt == valueDelays.end())
    return output.emitError(
        "output is not defined by a supported database operation");

  auto truthTable = getTruthTable(output.getOperands(), body);
  if (failed(truthTable))
    return failure();
  NPNClass npnClass;
  if (!npnTable.lookup(*truthTable, npnClass))
    npnClass = NPNClass::computeNPNCanonicalForm(*truthTable);
  if (!(npnClass.truthTable == *truthTable))
    return module.emitError(
        "cut-rewrite database entries must implement canonical NPN "
        "representatives");

  DatabaseEntry entry;
  entry.module = module;
  entry.npnClass = std::move(npnClass);
  entry.area = area;
  entry.delays.reserve(numInputs);
  for (DelayType delay : outputIt->second)
    entry.delays.push_back(std::max<DelayType>(delay, 0));
  return entry;
}

static FailureOr<std::shared_ptr<CutRewriteDatabase>>
loadDatabase(ArrayRef<std::string> filenames, MLIRContext *context) {
  auto database = std::make_shared<CutRewriteDatabase>();
  NPNTable npnTable;
  for (const std::string &filename : filenames) {
    auto module = parseDatabaseFile(filename, context);
    if (failed(module))
      return failure();
    database->modules.push_back(std::move(*module));

    for (hw::HWModuleOp hwModule :
         database->modules.back()->getOps<hw::HWModuleOp>()) {
      auto entry =
          loadDatabaseEntry(hwModule, database->inverterPrototype, npnTable);
      if (failed(entry))
        return failure();
      database->maxInputSize =
          std::max(database->maxInputSize,
                   static_cast<unsigned>(hwModule.getNumInputPorts()));
      database->entries.push_back(std::move(*entry));
    }
  }

  if (database->entries.empty()) {
    emitError(UnknownLoc::get(context))
        << "cut-rewrite databases contain no hw.module entries";
    return failure();
  }
  if (!database->inverterPrototype) {
    emitError(UnknownLoc::get(context))
        << "cut-rewrite databases require a BooleanLogicOpInterface "
           "operation supporting one input to materialize NPN phases";
    return failure();
  }
  return database;
}

struct DatabasePattern : CutRewritePattern {
  DatabasePattern(MLIRContext *context, const DatabaseEntry &entry,
                  Operation *inverterPrototype)
      : CutRewritePattern(context), entry(entry),
        inverterPrototype(inverterPrototype) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    (void)enumerator;
    (void)cut;
    return MatchResult(entry.area, entry.delays);
  }

  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(entry.npnClass);
    return true;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    const auto &network = enumerator.getLogicNetwork();
    Operation *root = network.getGate(cut.getRootIndex()).getOperation();
    assert(root && "expected cut root operation");

    const NPNClass &cutNPN = cut.getNPNClass(enumerator.getOptions().npnTable);
    SmallVector<unsigned> inputPermutation;
    cutNPN.getInputPermutation(entry.npnClass, inputPermutation);
    unsigned inputNegation =
        cutNPN.inputNegation ^ entry.npnClass.inputNegation;
    bool outputNegation =
        (cutNPN.outputNegation ^ entry.npnClass.outputNegation) & 1;

    IRMapping mapping;
    auto module = entry.module;
    Block *body = module.getBodyBlock();
    for (auto [index, argument] : llvm::enumerate(body->getArguments())) {
      Value input = network.getValue(cut.inputs[inputPermutation[index]]);
      if ((inputNegation >> index) & 1)
        input = createInverter(builder, root->getLoc(), input);
      mapping.map(argument, input);
    }

    for (Operation &op : body->without_terminator())
      builder.clone(op, mapping);

    auto output = cast<hw::OutputOp>(body->getTerminator());
    Value result = mapping.lookupOrDefault(output.getOperand(0));
    if (outputNegation)
      result = createInverter(builder, root->getLoc(), result);
    if (Operation *resultOp = result.getDefiningOp())
      return resultOp;
    return hw::WireOp::create(builder, root->getLoc(), result).getOperation();
  }

  unsigned getNumOutputs() const override { return 1; }

  StringRef getPatternName() const override {
    auto module = entry.module;
    return module.getModuleName();
  }

  LocationAttr getLoc() const override {
    auto module = entry.module;
    return module.getLoc();
  }

private:
  Value createInverter(OpBuilder &builder, Location location,
                       Value input) const {
    auto logicOp = cast<BooleanLogicOpInterface>(inverterPrototype);
    SmallVector<Value, 1> inputs{input};
    SmallVector<bool, 1> inverted{true};
    return logicOp.createBooleanLogicOp(builder, location, inputs, inverted);
  }

  const DatabaseEntry &entry;
  Operation *inverterPrototype;
};

struct CutRewritePass : circt::synth::impl::CutRewriteBase<CutRewritePass> {
  using circt::synth::impl::CutRewriteBase<CutRewritePass>::CutRewriteBase;

  LogicalResult initialize(MLIRContext *context) override {
    if (dbFiles.empty()) {
      emitError(UnknownLoc::get(context))
          << "synth-cut-rewrite requires at least one 'db-files' entry";
      return failure();
    }
    if (maxCutsPerRoot <= 0) {
      emitError(UnknownLoc::get(context))
          << "'max-cuts-per-root' must be positive";
      return failure();
    }

    SmallVector<std::string> filenames(dbFiles.begin(), dbFiles.end());
    auto loaded = loadDatabase(filenames, context);
    if (failed(loaded))
      return failure();
    database = std::move(*loaded);
    npnTable = std::make_shared<const NPNTable>();
    return success();
  }

  void runOnOperation() override {
    assert(database && "expected initialized cut-rewrite database");

    SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns;
    patterns.reserve(database->entries.size());
    for (const DatabaseEntry &entry : database->entries)
      patterns.push_back(std::make_unique<DatabasePattern>(
          getOperation()->getContext(), entry, database->inverterPrototype));

    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = database->maxInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.allowNoMatch = true;
    options.attachDebugTiming = test;
    options.npnTable = npnTable.get();

    CutRewritePatternSet patternSet(std::move(patterns));
    CutRewriter rewriter(options, patternSet);
    if (failed(rewriter.run(getOperation())))
      return signalPassFailure();

    const CutEnumeratorStats &stats = rewriter.getStats();
    numCutsCreated += stats.numCutsCreated;
    numCutSetsCreated += stats.numCutSetsCreated;
    numCutsRewritten += stats.numCutsRewritten;
  }

  std::shared_ptr<const CutRewriteDatabase> database;
  std::shared_ptr<const NPNTable> npnTable;
};

} // namespace
