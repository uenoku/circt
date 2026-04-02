//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements modular exact-synthesis database generation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/ExactSynthesis.h"
#include "CutRewriteDBImpl.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/SATSolver.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include <array>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#define DEBUG_TYPE "synth-exact-synthesis"

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace {

static constexpr unsigned kMaxExactSynthesisInputs = 4;
static constexpr unsigned kMaxMIGExactSearchArea = 32;
static constexpr unsigned kMaxAIGExactSearchArea = 32;
static constexpr StringLiteral kCutRewriteCanonicalTTAttr =
    "synth.cut_rewrite.canonical_tt";
static constexpr StringLiteral kCutRewriteInverterKindAttr =
    "synth.cut_rewrite.inverter_kind";

enum class CutRewriteInverterKind { mig, aig };

static std::string normalizeExactSynthesisKind(StringRef kind) {
  std::string normalized = kind.lower();
  for (char &c : normalized)
    if (c == '_')
      c = '-';
  return normalized;
}

static FailureOr<CutRewriteInverterKind>
getCutRewriteInverterKind(hw::HWModuleOp module) {
  auto kindAttr =
      module->getAttrOfType<StringAttr>(kCutRewriteInverterKindAttr);
  if (!kindAttr)
    return module.emitError("cut-rewrite database module missing '")
           << kCutRewriteInverterKindAttr << "'";
  if (kindAttr.getValue() == "mig")
    return CutRewriteInverterKind::mig;
  if (kindAttr.getValue() == "aig")
    return CutRewriteInverterKind::aig;
  module.emitError("unsupported cut-rewrite inverter kind '")
      << kindAttr.getValue() << "'";
  return failure();
}

static std::unique_ptr<IncrementalSATSolver>
createExactSATSolver(StringRef backend, int conflictLimit = -1) {
  auto applyConflictLimit = [&](std::unique_ptr<IncrementalSATSolver> solver) {
    if (solver)
      solver->setConflictLimit(conflictLimit);
    return solver;
  };
  if (backend == "auto") {
    if (auto solver = applyConflictLimit(createCadicalSATSolver()))
      return solver;
    return applyConflictLimit(createZ3SATSolver());
  }
  if (backend == "cadical")
    return applyConflictLimit(createCadicalSATSolver());
  if (backend == "z3")
    return applyConflictLimit(createZ3SATSolver());
  return {};
}

struct ExactSignalRef {
  unsigned source = 0;
  bool inverted = false;
};

struct ExactNetworkStep {
  SmallVector<ExactSignalRef, 3> fanins;
};

struct ExactNetwork {
  unsigned numInputs = 0;
  SmallVector<ExactNetworkStep, 4> steps;
  ExactSignalRef output;
};

using ExactCandidate = ExactNetworkStep;

static NPNClass getIdentityNPNClass(const BinaryTruthTable &canonicalTT) {
  SmallVector<unsigned> inputPermutation(canonicalTT.numInputs);
  std::iota(inputPermutation.begin(), inputPermutation.end(), 0);
  return NPNClass(canonicalTT, std::move(inputPermutation), 0, 0);
}

static FailureOr<BinaryTruthTable>
getCanonicalTruthTable(hw::HWModuleOp module) {
  auto canonicalTTAttr =
      module->getAttrOfType<IntegerAttr>(kCutRewriteCanonicalTTAttr);
  if (!canonicalTTAttr)
    return module.emitError("cut-rewrite database module missing '")
           << kCutRewriteCanonicalTTAttr << "'";
  return BinaryTruthTable(module.getNumInputPorts(), 1,
                          canonicalTTAttr.getValue());
}

static void
collectCanonicalTruthTables(unsigned numInputs,
                            SmallVectorImpl<BinaryTruthTable> &truthTables) {
  truthTables.clear();
  assert(numInputs < 6 &&
         "orbit walk requires the function space to fit in uint64_t");

  unsigned numTruthTableBits = 1u << numInputs;
  uint64_t numFunctions = 1ULL << numTruthTableBits;
  llvm::BitVector seen(numFunctions, false);

  SmallVector<unsigned> permutation(numInputs);
  std::iota(permutation.begin(), permutation.end(), 0);
  SmallVector<SmallVector<unsigned>> permutations;
  do {
    permutations.push_back(permutation);
  } while (std::next_permutation(permutation.begin(), permutation.end()));

  for (uint64_t value = 0; value != numFunctions; ++value) {
    if (seen.test(value))
      continue;

    BinaryTruthTable seed(numInputs, 1, APInt(numTruthTableBits, value));
    BinaryTruthTable best = seed;

    for (uint32_t negMask = 0; negMask < (1u << numInputs); ++negMask) {
      BinaryTruthTable negatedTT = seed.applyInputNegation(negMask);
      for (ArrayRef<unsigned> perm : permutations) {
        BinaryTruthTable permutedTT = negatedTT.applyPermutation(perm);
        for (unsigned outputNegMask = 0; outputNegMask != 2; ++outputNegMask) {
          BinaryTruthTable candidateTT =
              permutedTT.applyOutputNegation(outputNegMask);
          seen.set(candidateTT.table.getZExtValue());
          if (candidateTT.isLexicographicallySmaller(best))
            best = std::move(candidateTT);
        }
      }
    }

    truthTables.push_back(std::move(best));
  }
}

static DictionaryAttr createTechInfo(Builder &builder, double area,
                                     ArrayRef<DelayType> delays) {
  SmallVector<Attribute> delayRows;
  delayRows.reserve(delays.size());
  for (DelayType delay : delays)
    delayRows.push_back(
        builder.getArrayAttr({builder.getI64IntegerAttr(delay)}));
  return builder.getDictionaryAttr({
      builder.getNamedAttr("area", builder.getF64FloatAttr(area)),
      builder.getNamedAttr("delay", builder.getArrayAttr(delayRows)),
  });
}

template <typename MaterializeFn>
static void
appendDatabaseEntry(ModuleOp module, OpBuilder &builder, Builder &attrBuilder,
                    StringRef modulePrefix, const BinaryTruthTable &canonicalTT,
                    unsigned variantIndex, StringRef inverterKind,
                    DictionaryAttr techInfo, MaterializeFn materialize) {
  SmallVector<hw::PortInfo> inputs;
  inputs.reserve(canonicalTT.numInputs);
  for (unsigned i = 0; i != canonicalTT.numInputs; ++i) {
    hw::PortInfo port;
    port.name = attrBuilder.getStringAttr(("i" + Twine(i)).str());
    port.type = builder.getI1Type();
    port.dir = hw::ModulePort::Direction::Input;
    port.argNum = i;
    inputs.push_back(port);
  }

  hw::PortInfo output;
  output.name = attrBuilder.getStringAttr("y");
  output.type = builder.getI1Type();
  output.dir = hw::ModulePort::Direction::Output;
  output.argNum = 0;

  SmallString<64> moduleName;
  moduleName += modulePrefix;
  moduleName += "_i";
  moduleName += Twine(canonicalTT.numInputs).str();
  moduleName += "_tt_";
  SmallString<32> ttString;
  canonicalTT.table.toStringUnsigned(ttString, 16);
  moduleName += ttString;
  moduleName += "_v";
  moduleName += Twine(variantIndex).str();

  hw::HWModuleOp hwModule = hw::HWModuleOp::create(
      builder, module.getLoc(), attrBuilder.getStringAttr(moduleName),
      hw::ModulePortInfo(inputs, {output}));
  hwModule->setAttr("hw.techlib.info", techInfo);
  hwModule->setAttr(kCutRewriteInverterKindAttr,
                    attrBuilder.getStringAttr(inverterKind));
  hwModule->setAttr(kCutRewriteCanonicalTTAttr,
                    attrBuilder.getIntegerAttr(
                        builder.getIntegerType(1u << canonicalTT.numInputs),
                        canonicalTT.table));

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(hwModule.getBodyBlock());
  Value result = materialize(builder, hwModule.getLoc(),
                             hwModule.getBodyBlock()->getArguments());
  hwModule.getBodyBlock()->getTerminator()->setOperands({result});
}

class ExactSynthesisBackend {
public:
  virtual ~ExactSynthesisBackend() = default;

  virtual StringRef getKind() const = 0;
  virtual StringRef getInverterKind() const = 0;
  virtual StringRef getModulePrefix() const = 0;
  virtual unsigned getMaxSupportedInputs() const = 0;
  virtual StringRef getFamilyName() const = 0;
  virtual unsigned getArity() const = 0;
  virtual unsigned getMaxSearchArea() const = 0;

  virtual std::pair<BinaryTruthTable, bool>
  normalize(const BinaryTruthTable &tt) const = 0;
  virtual std::optional<ExactNetwork>
  synthesizeDirect(unsigned numInputs, const llvm::APInt &target) const = 0;
  virtual ExactNetwork applyOutputNegation(ExactNetwork network) const = 0;
  virtual void
  enumerateCandidates(unsigned availableSources,
                      SmallVectorImpl<ExactCandidate> &candidates) const = 0;
  virtual void addConditionedSemantics(
      IncrementalSATSolver &solver, int selector, int outLit,
      const ExactCandidate &candidate, unsigned minterm,
      llvm::function_ref<int(unsigned source, unsigned minterm, bool inverted)>
          getSourceLiteral) const = 0;
  virtual Value materializeNetwork(OpBuilder &builder, Location loc,
                                   ValueRange inputs,
                                   const ExactNetwork &network) const = 0;

  virtual LogicalResult
  emitDatabase(ModuleOp module,
               const ExactSynthesisDatabaseGenOptions &options) const = 0;
};

static const ExactSynthesisBackend *getExactSynthesisBackend(StringRef kind);

static SmallVector<DelayType, 6>
computeExactNetworkInputDelays(const ExactNetwork &network) {
  static constexpr DelayType kUnreachable =
      std::numeric_limits<DelayType>::min() / 4;
  SmallVector<DelayType, 6> result(network.numInputs, 0);

  SmallVector<SmallVector<DelayType, 6>, 4> stepDelays;
  stepDelays.reserve(network.steps.size());

  auto getSourceDelays = [&](unsigned source) {
    SmallVector<DelayType, 6> delays(network.numInputs, kUnreachable);
    if (source == 0)
      return delays;
    if (source <= network.numInputs) {
      delays[source - 1] = 0;
      return delays;
    }
    unsigned stepIndex = source - (network.numInputs + 1);
    assert(stepIndex < stepDelays.size() && "step reference out of order");
    return stepDelays[stepIndex];
  };

  for (const auto &step : network.steps) {
    SmallVector<DelayType, 6> delays(network.numInputs, kUnreachable);
    for (ExactSignalRef signal : step.fanins) {
      auto childDelays = getSourceDelays(signal.source);
      for (unsigned i = 0; i != network.numInputs; ++i) {
        if (childDelays[i] == kUnreachable)
          continue;
        delays[i] = std::max(delays[i], childDelays[i] + 1);
      }
    }
    stepDelays.push_back(std::move(delays));
  }

  auto outputDelays = getSourceDelays(network.output.source);
  if (network.output.inverted && network.output.source != 0)
    for (DelayType &delay : outputDelays)
      if (delay != kUnreachable)
        ++delay;

  for (unsigned i = 0; i != network.numInputs; ++i)
    if (outputDelays[i] != kUnreachable)
      result[i] = outputDelays[i];
  return result;
}

static unsigned
computeMaterializedExactNetworkArea(const ExactNetwork &network) {
  unsigned area = network.steps.size();
  if (network.output.inverted && network.output.source != 0)
    ++area;
  return area;
}

class GenericExactSATProblem {
public:
  struct SolveResult {
    IncrementalSATSolver::Result result = IncrementalSATSolver::kUNKNOWN;
    std::optional<ExactNetwork> network;
  };

  GenericExactSATProblem(const ExactSynthesisBackend &backend,
                         IncrementalSATSolver &solver, unsigned numInputs,
                         const llvm::APInt &target, unsigned numSteps)
      : backend(backend), solver(solver), numInputs(numInputs), target(target),
        numSteps(numSteps), arity(backend.getArity()),
        numMinterms(1u << numInputs), totalSources(1 + numInputs + numSteps) {}

  SolveResult solve() {
    buildEncoding();
    auto result = solver.solve();
    if (result != IncrementalSATSolver::kSAT)
      return {.result = result};
    return {.result = result, .network = decodeModel()};
  }

private:
  int newVar() {
    int fresh = ++nextVar;
    solver.reserveVars(fresh);
    return fresh;
  }

  void addAtMostOne(ArrayRef<int> vars) {
    if (vars.size() < 2)
      return;

    SmallVector<int, 8> ladder(vars.size() - 1);
    for (int &var : ladder)
      var = newVar();

    solver.addClause({-vars.front(), ladder.front()});
    for (unsigned i = 1, e = vars.size() - 1; i < e; ++i) {
      solver.addClause({-vars[i], ladder[i]});
      solver.addClause({-ladder[i - 1], ladder[i]});
      solver.addClause({-vars[i], -ladder[i - 1]});
    }
    solver.addClause({-vars.back(), -ladder.back()});
  }

  void addExactlyOne(ArrayRef<int> vars) {
    solver.addClause(vars);
    addAtMostOne(vars);
  }

  int getSourceValueVar(unsigned source, unsigned minterm) const {
    return sourceValueVars[source][minterm];
  }

  int getSourceLiteral(unsigned source, unsigned minterm, bool inverted) const {
    int lit = getSourceValueVar(source, minterm);
    return inverted ? -lit : lit;
  }

  void buildEncoding() {
    sourceValueVars.resize(totalSources);
    for (unsigned source = 0; source != totalSources; ++source) {
      sourceValueVars[source].reserve(numMinterms);
      for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
        sourceValueVars[source].push_back(newVar());
    }

    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({-getSourceValueVar(0, minterm)});

    for (unsigned input = 0; input != numInputs; ++input)
      for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
        solver.addClause({((minterm >> input) & 1)
                              ? getSourceValueVar(1 + input, minterm)
                              : -getSourceValueVar(1 + input, minterm)});

    stepCandidates.resize(numSteps);
    stepSelectionVars.resize(numSteps);
    for (unsigned step = 0; step != numSteps; ++step) {
      unsigned availableSources = 1 + numInputs + step;
      backend.enumerateCandidates(availableSources, stepCandidates[step]);
      auto &selectionVars = stepSelectionVars[step];
      selectionVars.reserve(stepCandidates[step].size());
      for (size_t i = 0, e = stepCandidates[step].size(); i != e; ++i)
        selectionVars.push_back(newVar());
      addExactlyOne(selectionVars);

      unsigned outSource = 1 + numInputs + step;
      for (auto [candidateIndex, candidate] :
           llvm::enumerate(stepCandidates[step])) {
        backend.addConditionedSemantics(
            solver, selectionVars[candidateIndex],
            getSourceValueVar(outSource, 0), candidate, 0,
            [&](unsigned source, unsigned minterm, bool inverted) {
              return getSourceLiteral(source, minterm, inverted);
            });
        for (unsigned minterm = 1; minterm != numMinterms; ++minterm)
          backend.addConditionedSemantics(
              solver, selectionVars[candidateIndex],
              getSourceValueVar(outSource, minterm), candidate, minterm,
              [&](unsigned source, unsigned currentMinterm, bool inverted) {
                return getSourceLiteral(source, currentMinterm, inverted);
              });
      }
    }

    unsigned rootSource = totalSources - 1;
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({target[minterm]
                            ? getSourceValueVar(rootSource, minterm)
                            : -getSourceValueVar(rootSource, minterm)});
  }

  ExactNetwork decodeModel() const {
    ExactNetwork network;
    network.numInputs = numInputs;
    network.steps.reserve(numSteps);

    for (unsigned step = 0; step != numSteps; ++step) {
      const auto &selectionVars = stepSelectionVars[step];
      const auto &candidates = stepCandidates[step];
      for (size_t i = 0, e = selectionVars.size(); i != e; ++i) {
        if (solver.val(selectionVars[i]) != selectionVars[i])
          continue;
        ExactNetworkStep resultStep;
        resultStep.fanins.reserve(arity);
        for (unsigned operand = 0; operand != arity; ++operand)
          resultStep.fanins.push_back(candidates[i].fanins[operand]);
        network.steps.push_back(resultStep);
        break;
      }
    }

    network.output = {1 + numInputs + numSteps - 1, false};
    return network;
  }

  const ExactSynthesisBackend &backend;
  IncrementalSATSolver &solver;
  unsigned numInputs;
  llvm::APInt target;
  unsigned numSteps;
  unsigned arity;
  unsigned numMinterms;
  unsigned totalSources;
  int nextVar = 0;
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  SmallVector<SmallVector<ExactCandidate, 64>, 8> stepCandidates;
  SmallVector<SmallVector<int, 64>, 8> stepSelectionVars;
};

class GenericExactSynthesizer {
public:
  enum class QueryStatus {
    Solved,
    NoSolution,
    ConflictLimitReached,
    Error,
  };

  struct QueryResult {
    QueryStatus status = QueryStatus::Error;
    std::optional<ExactNetwork> network;
  };

  GenericExactSynthesizer(const ExactSynthesisBackend &backend,
                          StringRef satBackend, int conflictLimit)
      : backend(backend), satBackend(satBackend), conflictLimit(conflictLimit) {
  }

  QueryResult synthesizeForArea(const BinaryTruthTable &tt,
                                unsigned area) const {
    auto [normalizedTT, invertOutput] = backend.normalize(tt);
    auto result =
        synthesizeNormalized(normalizedTT.numInputs, normalizedTT.table, area);
    if (result.status != QueryStatus::Solved || !result.network)
      return result;

    ExactNetwork network = *result.network;
    if (invertOutput)
      network = backend.applyOutputNegation(std::move(network));
    return {.status = QueryStatus::Solved,
            .network = std::optional<ExactNetwork>(std::move(network))};
  }

private:
  QueryResult synthesizeNormalized(unsigned numInputs,
                                   const llvm::APInt &target,
                                   unsigned area) const {
    if (area == 0) {
      auto direct = backend.synthesizeDirect(numInputs, target);
      return {.status = direct ? QueryStatus::Solved : QueryStatus::NoSolution,
              .network = std::move(direct)};
    }

    auto solver = createExactSATSolver(satBackend, conflictLimit);
    if (!solver)
      return {.status = QueryStatus::Error};

    GenericExactSATProblem problem(backend, *solver, numInputs, target, area);
    auto solveResult = problem.solve();
    if (solveResult.result == IncrementalSATSolver::kUNKNOWN)
      return {.status = QueryStatus::ConflictLimitReached};
    return {.status = solveResult.network ? QueryStatus::Solved
                                          : QueryStatus::NoSolution,
            .network = std::move(solveResult.network)};
  }

  const ExactSynthesisBackend &backend;
  std::string satBackend;
  int conflictLimit;
};

static LogicalResult emitExactSynthesisDatabaseForBackend(
    const ExactSynthesisBackend &backend, ModuleOp module,
    const ExactSynthesisDatabaseGenOptions &options) {
  if (options.maxInputs > kMaxExactSynthesisInputs) {
    module.emitError() << backend.getFamilyName()
                       << " exact database generation supports at most "
                       << kMaxExactSynthesisInputs << " inputs";
    return failure();
  }
  if (options.conflictLimit < -1) {
    module.emitError()
        << "'conflict-limit' must be greater than or equal to -1";
    return failure();
  }
  if (!createExactSATSolver(options.satSolver, options.conflictLimit)) {
    module.emitError() << "Exact " << backend.getFamilyName()
                       << " database generation requires a SAT solver backend '"
                       << options.satSolver << "'";
    return failure();
  }

  auto *context = module.getContext();
  Builder attrBuilder(context);
  OpBuilder builder(context);
  builder.setInsertionPointToStart(module.getBody());

  for (unsigned numInputs = 1; numInputs <= options.maxInputs; ++numInputs) {
    SmallVector<BinaryTruthTable> canonicalTruthTables;
    collectCanonicalTruthTables(numInputs, canonicalTruthTables);

    std::vector<std::optional<ExactNetwork>> exactNetworks(
        canonicalTruthTables.size());
    if (failed(mlir::failableParallelForEachN(
            context, 0, canonicalTruthTables.size(), [&](size_t index) {
              GenericExactSynthesizer synthesizer(backend, options.satSolver,
                                                  options.conflictLimit);
              SmallString<32> ttString;
              canonicalTruthTables[index].table.toStringUnsigned(ttString, 16);
              bool hitConflictLimit = false;
              for (unsigned area = 0; area <= backend.getMaxSearchArea();
                   ++area) {
                auto result = synthesizer.synthesizeForArea(
                    canonicalTruthTables[index], area);
                if (result.status == GenericExactSynthesizer::QueryStatus::
                                         ConflictLimitReached) {
                  hitConflictLimit = true;
                  LLVM_DEBUG(llvm::dbgs()
                             << "Exact" << backend.getFamilyName()
                             << " dbgen: conflict limit reached for tt="
                             << ttString << " at area=" << area << "\n");
                  break;
                }
                if (result.status ==
                    GenericExactSynthesizer::QueryStatus::Error) {
                  module.emitError()
                      << "failed to synthesize exact "
                      << backend.getFamilyName()
                      << " for canonical truth table " << ttString;
                  return failure();
                }
                if (result.status !=
                        GenericExactSynthesizer::QueryStatus::Solved ||
                    !result.network)
                  continue;

                exactNetworks[index] = std::move(*result.network);
                break;
              }
              if (!exactNetworks[index] && !hitConflictLimit) {
                module.emitError()
                    << "failed to synthesize exact " << backend.getFamilyName()
                    << " for canonical truth table " << ttString;
                return failure();
              }
              if (hitConflictLimit)
                module.emitWarning()
                    << "stopping exact " << backend.getFamilyName()
                    << " search for canonical truth table " << ttString
                    << " due to SAT conflict limit";
              return success();
            })))
      return failure();

    for (auto [index, canonicalTT] : llvm::enumerate(canonicalTruthTables)) {
      if (!exactNetworks[index])
        continue;
      const ExactNetwork &exactNetwork = *exactNetworks[index];
      auto delays = computeExactNetworkInputDelays(exactNetwork);
      appendDatabaseEntry(
          module, builder, attrBuilder, backend.getModulePrefix(), canonicalTT,
          0, backend.getInverterKind(),
          createTechInfo(attrBuilder,
                         computeMaterializedExactNetworkArea(exactNetwork),
                         delays),
          [&](OpBuilder &entryBuilder, Location loc, ValueRange inputs) {
            return backend.materializeNetwork(entryBuilder, loc, inputs,
                                              exactNetwork);
          });
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MIG Exact Synthesis Backend
//===----------------------------------------------------------------------===//
static Value materializeExactMIGNetwork(OpBuilder &builder, Location loc,
                                        ValueRange inputs,
                                        const ExactNetwork &network) {
  Value constZero;
  Value constOne;
  auto getConstant = [&](bool value) -> Value {
    if (!value) {
      if (!constZero)
        constZero = hw::ConstantOp::create(builder, loc, APInt(1, 0));
      return constZero;
    }
    if (!constOne)
      constOne = hw::ConstantOp::create(builder, loc, APInt(1, 1));
    return constOne;
  };

  auto getRawSignal = [&](ExactSignalRef signal,
                          ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(false);
    if (signal.source <= inputs.size())
      return inputs[signal.source - 1];

    unsigned stepIndex = signal.source - (inputs.size() + 1);
    assert(stepIndex < stepValues.size() && "invalid synthesized step index");
    return stepValues[stepIndex];
  };

  auto materializeOutputSignal = [&](ExactSignalRef signal,
                                     ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(signal.inverted);

    Value value = getRawSignal(signal, stepValues);
    if (!signal.inverted) {
      if (value.getDefiningOp())
        return value;
      return hw::WireOp::create(builder, loc, value);
    }

    std::array<Value, 1> operands = {value};
    std::array<bool, 1> inverted = {true};
    return synth::mig::MajorityInverterOp::create(builder, loc, operands,
                                                  inverted);
  };

  SmallVector<Value, 4> stepValues;
  stepValues.reserve(network.steps.size());
  for (const auto &step : network.steps) {
    std::array<Value, 3> operands = {getRawSignal(step.fanins[0], stepValues),
                                     getRawSignal(step.fanins[1], stepValues),
                                     getRawSignal(step.fanins[2], stepValues)};
    std::array<bool, 3> inverted = {step.fanins[0].inverted,
                                    step.fanins[1].inverted,
                                    step.fanins[2].inverted};
    stepValues.push_back(synth::mig::MajorityInverterOp::create(
        builder, loc, operands, inverted));
  }
  return materializeOutputSignal(network.output, stepValues);
}

struct LoadedExactNetworkEntry : public LoadedCutRewriteEntry {
  CutRewriteInverterKind inverterKind;
  Operation *moduleOp;

  LoadedExactNetworkEntry(CutRewriteInverterKind inverterKind,
                          hw::HWModuleOp module)
      : inverterKind(inverterKind), moduleOp(module.getOperation()) {}

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    const auto &logicNetwork = enumerator.getLogicNetwork();
    auto *rootOp = logicNetwork.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");

    auto dbModule = cast<hw::HWModuleOp>(moduleOp);
    auto *bodyBlock = dbModule.getBodyBlock();
    auto output = dyn_cast<hw::OutputOp>(bodyBlock->getTerminator());
    if (!output || output.getNumOperands() != 1) {
      dbModule.emitError("cut-rewrite database module must terminate with a "
                         "single output");
      return failure();
    }

    const auto &cutNPN = cut.getNPNClass();
    assert(cutNPN.inputPermutation.size() == bodyBlock->getNumArguments() &&
           "cut input permutation size mismatch");

    IRMapping mapping;
    for (auto [index, argument] : llvm::enumerate(bodyBlock->getArguments())) {
      Value input =
          logicNetwork.getValue(cut.inputs[cutNPN.inputPermutation[index]]);
      if ((cutNPN.inputNegation >> index) & 1)
        input = materializeInverter(builder, rootOp->getLoc(), input);
      mapping.map(argument, input);
    }

    for (Operation &op : bodyBlock->without_terminator())
      builder.clone(op, mapping);

    Value result = mapping.lookup(output.getOperand(0));
    if (cutNPN.outputNegation & 1)
      result = materializeInverter(builder, rootOp->getLoc(), result);

    if (auto *resultOp = result.getDefiningOp())
      return resultOp;
    return hw::WireOp::create(builder, rootOp->getLoc(), result).getOperation();
  }

private:
  Value materializeInverter(OpBuilder &builder, Location loc,
                            Value input) const {
    std::array<Value, 1> operands = {input};
    std::array<bool, 1> inverted = {true};
    switch (inverterKind) {
    case CutRewriteInverterKind::mig:
      return synth::mig::MajorityInverterOp::create(builder, loc, operands,
                                                    inverted);
    case CutRewriteInverterKind::aig:
      return synth::aig::AndInverterOp::create(builder, loc, operands,
                                               inverted);
    }
    llvm_unreachable("unsupported inverter kind");
  }
};

static FailureOr<std::unique_ptr<LoadedCutRewriteEntry>>
parseExactSynthesisCutRewriteEntry(hw::HWModuleOp module) {
  auto canonicalTT = getCanonicalTruthTable(module);
  if (failed(canonicalTT))
    return failure();
  auto inverterKind = getCutRewriteInverterKind(module);
  if (failed(inverterKind))
    return failure();
  auto areaAndDelay = getAreaAndDelayFromTechInfo(module);
  if (failed(areaAndDelay))
    return failure();

  auto entry = std::make_unique<LoadedExactNetworkEntry>(*inverterKind, module);
  entry->moduleName = module.getModuleName().str();
  entry->npnClass = getIdentityNPNClass(*canonicalTT);
  entry->area = areaAndDelay->first;
  entry->delay = std::move(areaAndDelay->second);
  std::unique_ptr<LoadedCutRewriteEntry> result = std::move(entry);
  return result;
}

class MIGExactSynthesisBackend final : public ExactSynthesisBackend {
public:
  StringRef getKind() const override { return "mig-exact"; }
  StringRef getInverterKind() const override { return "mig"; }
  StringRef getModulePrefix() const override { return "mig_exact"; }
  unsigned getMaxSupportedInputs() const override {
    return kMaxExactSynthesisInputs;
  }
  StringRef getFamilyName() const override { return "MIG"; }
  unsigned getArity() const override { return 3; }
  unsigned getMaxSearchArea() const override { return kMaxMIGExactSearchArea; }

  std::pair<BinaryTruthTable, bool>
  normalize(const BinaryTruthTable &tt) const override {
    if (!tt.table[0])
      return {tt, false};
    BinaryTruthTable normalized = tt;
    normalized.table.flipAllBits();
    return {std::move(normalized), true};
  }

  std::optional<ExactNetwork>
  synthesizeDirect(unsigned numInputs,
                   const llvm::APInt &target) const override {
    ExactNetwork network;
    network.numInputs = numInputs;
    if (target.isZero()) {
      network.output = {0, false};
      return network;
    }
    if (target.isAllOnes()) {
      network.output = {0, true};
      return network;
    }
    for (unsigned input = 0; input != numInputs; ++input) {
      llvm::APInt mask = circt::createVarMask(numInputs, input, true);
      if (target == mask) {
        network.output = {1 + input, false};
        return network;
      }
      llvm::APInt invertedMask = mask;
      invertedMask.flipAllBits();
      if (target == invertedMask) {
        network.output = {1 + input, true};
        return network;
      }
    }
    return std::nullopt;
  }

  ExactNetwork applyOutputNegation(ExactNetwork network) const override {
    if (network.steps.empty()) {
      network.output.inverted = !network.output.inverted;
      return network;
    }
    for (auto &fanin : network.steps.back().fanins)
      fanin.inverted = !fanin.inverted;
    return network;
  }

  void enumerateCandidates(
      unsigned availableSources,
      SmallVectorImpl<ExactCandidate> &candidates) const override {
    candidates.clear();
    for (unsigned a = 0; a + 2 < availableSources; ++a)
      for (unsigned b = a + 1; b + 1 < availableSources; ++b)
        for (unsigned c = b + 1; c < availableSources; ++c)
          for (unsigned invMask = 0; invMask != 8; ++invMask) {
            ExactCandidate candidate;
            candidate.fanins.push_back({a, static_cast<bool>(invMask & 1)});
            candidate.fanins.push_back({b, static_cast<bool>(invMask & 2)});
            candidate.fanins.push_back({c, static_cast<bool>(invMask & 4)});
            candidates.push_back(std::move(candidate));
          }
  }

  void addConditionedSemantics(IncrementalSATSolver &solver, int selector,
                               int outLit, const ExactCandidate &candidate,
                               unsigned minterm,
                               llvm::function_ref<int(unsigned, unsigned, bool)>
                                   getSourceLiteral) const override {
    auto addConditionedClause = [&](std::initializer_list<int> lits) {
      SmallVector<int, 8> clause;
      clause.reserve(lits.size() + 1);
      clause.push_back(-selector);
      clause.append(lits.begin(), lits.end());
      solver.addClause(clause);
    };

    int aLit = getSourceLiteral(candidate.fanins[0].source, minterm,
                                candidate.fanins[0].inverted);
    int bLit = getSourceLiteral(candidate.fanins[1].source, minterm,
                                candidate.fanins[1].inverted);
    int cLit = getSourceLiteral(candidate.fanins[2].source, minterm,
                                candidate.fanins[2].inverted);
    addConditionedClause({-aLit, -bLit, outLit});
    addConditionedClause({-aLit, -cLit, outLit});
    addConditionedClause({-bLit, -cLit, outLit});
    addConditionedClause({aLit, bLit, -outLit});
    addConditionedClause({aLit, cLit, -outLit});
    addConditionedClause({bLit, cLit, -outLit});
  }

  Value materializeNetwork(OpBuilder &builder, Location loc, ValueRange inputs,
                           const ExactNetwork &network) const override {
    return materializeExactMIGNetwork(builder, loc, inputs, network);
  }

  LogicalResult
  emitDatabase(ModuleOp module,
               const ExactSynthesisDatabaseGenOptions &options) const override {
    return emitExactSynthesisDatabaseForBackend(*this, module, options);
  }
};

//===----------------------------------------------------------------------===//
// AIG Exact Synthesis Backend
//===----------------------------------------------------------------------===//
static Value materializeExactAIGNetwork(OpBuilder &builder, Location loc,
                                        ValueRange inputs,
                                        const ExactNetwork &network) {
  Value constZero;
  Value constOne;
  auto getConstant = [&](bool value) -> Value {
    if (!value) {
      if (!constZero)
        constZero = hw::ConstantOp::create(builder, loc, APInt(1, 0));
      return constZero;
    }
    if (!constOne)
      constOne = hw::ConstantOp::create(builder, loc, APInt(1, 1));
    return constOne;
  };

  auto getRawSignal = [&](ExactSignalRef signal,
                          ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(false);
    if (signal.source <= inputs.size())
      return inputs[signal.source - 1];

    unsigned stepIndex = signal.source - (inputs.size() + 1);
    assert(stepIndex < stepValues.size() && "invalid synthesized step index");
    return stepValues[stepIndex];
  };

  auto materializeOutputSignal = [&](ExactSignalRef signal,
                                     ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(signal.inverted);

    Value value = getRawSignal(signal, stepValues);
    if (!signal.inverted) {
      if (value.getDefiningOp())
        return value;
      return hw::WireOp::create(builder, loc, value);
    }

    std::array<Value, 1> operands = {value};
    std::array<bool, 1> inverted = {true};
    return synth::aig::AndInverterOp::create(builder, loc, operands, inverted);
  };

  SmallVector<Value, 4> stepValues;
  stepValues.reserve(network.steps.size());
  for (const auto &step : network.steps) {
    std::array<Value, 2> operands = {getRawSignal(step.fanins[0], stepValues),
                                     getRawSignal(step.fanins[1], stepValues)};
    std::array<bool, 2> inverted = {step.fanins[0].inverted,
                                    step.fanins[1].inverted};
    stepValues.push_back(
        synth::aig::AndInverterOp::create(builder, loc, operands, inverted));
  }
  return materializeOutputSignal(network.output, stepValues);
}

class AIGExactSynthesisBackend final : public ExactSynthesisBackend {
public:
  StringRef getKind() const override { return "aig-exact"; }
  StringRef getInverterKind() const override { return "aig"; }
  StringRef getModulePrefix() const override { return "aig_exact"; }
  unsigned getMaxSupportedInputs() const override {
    return kMaxExactSynthesisInputs;
  }
  StringRef getFamilyName() const override { return "AIG"; }
  unsigned getArity() const override { return 2; }
  unsigned getMaxSearchArea() const override { return kMaxAIGExactSearchArea; }

  std::pair<BinaryTruthTable, bool>
  normalize(const BinaryTruthTable &tt) const override {
    if (!tt.table[0])
      return {tt, false};
    BinaryTruthTable normalized = tt;
    normalized.table.flipAllBits();
    return {std::move(normalized), true};
  }

  std::optional<ExactNetwork>
  synthesizeDirect(unsigned numInputs,
                   const llvm::APInt &target) const override {
    ExactNetwork network;
    network.numInputs = numInputs;
    if (target.isZero()) {
      network.output = {0, false};
      return network;
    }
    if (target.isAllOnes()) {
      network.output = {0, true};
      return network;
    }
    for (unsigned input = 0; input != numInputs; ++input) {
      llvm::APInt mask = circt::createVarMask(numInputs, input, true);
      if (target == mask) {
        network.output = {1 + input, false};
        return network;
      }
    }
    return std::nullopt;
  }

  ExactNetwork applyOutputNegation(ExactNetwork network) const override {
    network.output.inverted = !network.output.inverted;
    return network;
  }

  void enumerateCandidates(
      unsigned availableSources,
      SmallVectorImpl<ExactCandidate> &candidates) const override {
    candidates.clear();
    for (unsigned a = 0; a + 1 < availableSources; ++a)
      for (unsigned b = a + 1; b < availableSources; ++b)
        for (unsigned invMask = 0; invMask != 4; ++invMask) {
          ExactCandidate candidate;
          candidate.fanins.push_back({a, static_cast<bool>(invMask & 1)});
          candidate.fanins.push_back({b, static_cast<bool>(invMask & 2)});
          candidates.push_back(std::move(candidate));
        }
  }

  void addConditionedSemantics(IncrementalSATSolver &solver, int selector,
                               int outLit, const ExactCandidate &candidate,
                               unsigned minterm,
                               llvm::function_ref<int(unsigned, unsigned, bool)>
                                   getSourceLiteral) const override {
    auto addConditionedClause = [&](std::initializer_list<int> lits) {
      SmallVector<int, 8> clause;
      clause.reserve(lits.size() + 1);
      clause.push_back(-selector);
      clause.append(lits.begin(), lits.end());
      solver.addClause(clause);
    };

    int aLit = getSourceLiteral(candidate.fanins[0].source, minterm,
                                candidate.fanins[0].inverted);
    int bLit = getSourceLiteral(candidate.fanins[1].source, minterm,
                                candidate.fanins[1].inverted);
    addConditionedClause({-aLit, -bLit, outLit});
    addConditionedClause({aLit, -outLit});
    addConditionedClause({bLit, -outLit});
  }

  Value materializeNetwork(OpBuilder &builder, Location loc, ValueRange inputs,
                           const ExactNetwork &network) const override {
    return materializeExactAIGNetwork(builder, loc, inputs, network);
  }

  LogicalResult
  emitDatabase(ModuleOp module,
               const ExactSynthesisDatabaseGenOptions &options) const override {
    return emitExactSynthesisDatabaseForBackend(*this, module, options);
  }
};

static const ExactSynthesisBackend *getExactSynthesisBackend(StringRef kind) {
  static const MIGExactSynthesisBackend migBackend;
  static const AIGExactSynthesisBackend aigBackend;

  std::string normalized = normalizeExactSynthesisKind(kind);
  if (normalized == migBackend.getKind())
    return &migBackend;
  if (normalized == aigBackend.getKind())
    return &aigBackend;
  return nullptr;
}

} // namespace

FailureOr<std::unique_ptr<LoadedCutRewriteEntry>>
circt::synth::parseCutRewriteEntry(hw::HWModuleOp module) {
  return parseExactSynthesisCutRewriteEntry(module);
}

LogicalResult circt::synth::emitExactSynthesisDatabase(
    mlir::ModuleOp module, StringRef kind,
    const ExactSynthesisDatabaseGenOptions &options) {
  std::string normalizedKind = normalizeExactSynthesisKind(kind);
  auto *backend = getExactSynthesisBackend(normalizedKind);
  if (!backend) {
    module.emitError() << "unsupported database kind '" << kind << "'";
    return failure();
  }

  return backend->emitDatabase(module, options);
}
