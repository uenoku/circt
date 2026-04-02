//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a modular exact-synthesis database generator and a
// generic file-backed cut-rewrite pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/ExactSynthesis.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/SATSolver.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include <array>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

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

static constexpr unsigned kMaxExactSynthesisInputs = 4;
static constexpr unsigned kMaxMIGExactSearchArea = 32;
static constexpr unsigned kMaxAIGExactSearchArea = 32;
static constexpr StringLiteral kCutRewriteCanonicalTTAttr =
    "synth.cut_rewrite.canonical_tt";
static constexpr StringLiteral kCutRewriteDBKindAttr =
    "synth.cut_rewrite.db_kind";

static std::string normalizeDatabaseKind(StringRef kind) {
  std::string normalized = kind.lower();
  for (char &c : normalized)
    if (c == '_')
      c = '-';
  return normalized;
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

static NPNClass getIdentityNPNClass(const BinaryTruthTable &canonicalTT) {
  SmallVector<unsigned> inputPermutation(canonicalTT.numInputs);
  std::iota(inputPermutation.begin(), inputPermutation.end(), 0);
  return NPNClass(canonicalTT, std::move(inputPermutation), 0, 0);
}

static FailureOr<BinaryTruthTable> getCanonicalTruthTable(hw::HWModuleOp module) {
  auto canonicalTTAttr =
      module->getAttrOfType<IntegerAttr>(kCutRewriteCanonicalTTAttr);
  if (!canonicalTTAttr)
    return module.emitError("cut-rewrite database module missing '")
           << kCutRewriteCanonicalTTAttr << "'";
  return BinaryTruthTable(module.getNumInputPorts(), 1, canonicalTTAttr.getValue());
}

static FailureOr<ExactSignalRef>
lookupExactSignal(Value value, DenseMap<Value, ExactSignalRef> &valueToSignal,
                  Operation *user, StringRef familyName) {
  auto it = valueToSignal.find(value);
  if (it == valueToSignal.end()) {
    auto diag = user->emitError("cut-rewrite ")
                << familyName << " database module used an unsupported operand";
    diag.attachNote() << "operand: " << value;
    return failure();
  }
  return it->second;
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
static void appendDatabaseEntry(ModuleOp module, OpBuilder &builder,
                                Builder &attrBuilder, StringRef modulePrefix,
                                const BinaryTruthTable &canonicalTT,
                                unsigned variantIndex,
                                DictionaryAttr techInfo,
                                MaterializeFn materialize) {
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
  hwModule->setAttr(
      kCutRewriteCanonicalTTAttr,
      attrBuilder.getIntegerAttr(builder.getIntegerType(1u << canonicalTT.numInputs),
                                 canonicalTT.table));

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(hwModule.getBodyBlock());
  Value result = materialize(builder, hwModule.getLoc(),
                             hwModule.getBodyBlock()->getArguments());
  hwModule.getBodyBlock()->getTerminator()->setOperands({result});
}

struct LoadedExactSynthesisEntry {
  virtual ~LoadedExactSynthesisEntry() = default;

  std::string moduleName;
  NPNClass npnClass;
  double area = 0.0;
  SmallVector<DelayType> delay;

  virtual FailureOr<Operation *> rewrite(OpBuilder &builder,
                                         CutEnumerator &enumerator,
                                         const Cut &cut) const = 0;
};

struct LoadedExactSynthesisDatabase {
  std::string kind;
  std::vector<std::unique_ptr<LoadedExactSynthesisEntry>> entries;
  unsigned maxInputSize = 0;
};

class ExactSynthesisBackend {
public:
  virtual ~ExactSynthesisBackend() = default;

  virtual StringRef getKind() const = 0;
  virtual StringRef getModulePrefix() const = 0;
  virtual unsigned getMaxSupportedInputs() const = 0;
  virtual StringRef getFamilyName() const = 0;

  virtual LogicalResult
  emitDatabase(ModuleOp module,
               const ExactSynthesisDatabaseGenOptions &options) const = 0;
  virtual FailureOr<std::unique_ptr<LoadedExactSynthesisEntry>>
  parseEntry(hw::HWModuleOp module) const = 0;
};

static const ExactSynthesisBackend *getExactSynthesisBackend(StringRef kind);

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

//===----------------------------------------------------------------------===//
// MIG Exact Synthesis Backend
//===----------------------------------------------------------------------===//

struct ExactMIGStep {
  std::array<ExactSignalRef, 3> fanins;
};

struct ExactMIGNetwork {
  unsigned numInputs = 0;
  SmallVector<ExactMIGStep, 4> steps;
  ExactSignalRef output;
};

static ExactMIGNetwork invertExactMIGOutput(ExactMIGNetwork network) {
  if (network.steps.empty()) {
    network.output.inverted = !network.output.inverted;
    return network;
  }

  for (auto &fanin : network.steps.back().fanins)
    fanin.inverted = !fanin.inverted;
  return network;
}

struct ExactMIGCandidate {
  std::array<unsigned, 3> fanins;
  std::array<bool, 3> inverted;
};

static void
enumerateExactMIGCandidates(unsigned availableSources,
                            SmallVectorImpl<ExactMIGCandidate> &candidates) {
  candidates.clear();
  for (unsigned a = 0; a + 2 < availableSources; ++a)
    for (unsigned b = a + 1; b + 1 < availableSources; ++b)
      for (unsigned c = b + 1; c < availableSources; ++c)
        for (unsigned invMask = 0; invMask != 8; ++invMask)
          candidates.push_back(
              {{{a, b, c}},
               {{static_cast<bool>(invMask & 1), static_cast<bool>(invMask & 2),
                 static_cast<bool>(invMask & 4)}}});
}

class ExactMIGSATProblem {
public:
  struct SolveResult {
    IncrementalSATSolver::Result result = IncrementalSATSolver::kUNKNOWN;
    std::optional<ExactMIGNetwork> network;
  };

  ExactMIGSATProblem(IncrementalSATSolver &solver, unsigned numInputs,
                     const llvm::APInt &target, unsigned numSteps)
      : solver(solver), numInputs(numInputs), target(target),
        numSteps(numSteps), numMinterms(1u << numInputs),
        totalSources(1 + numInputs + numSteps) {}

  SolveResult solve() {
    buildEncoding();
    LLVM_DEBUG(llvm::dbgs() << "ExactMIG SAT solve: numInputs=" << numInputs
                            << " numSteps=" << numSteps << " numMinterms="
                            << numMinterms << " vars=" << nextVar << "\n");
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

  void addConditionedClause(int selector, std::initializer_list<int> lits) {
    SmallVector<int, 8> clause;
    clause.reserve(lits.size() + 1);
    clause.push_back(-selector);
    clause.append(lits.begin(), lits.end());
    solver.addClause(clause);
  }

  void addConditionedMajority(int selector, int outLit, int aLit, int bLit,
                              int cLit) {
    addConditionedClause(selector, {-aLit, -bLit, outLit});
    addConditionedClause(selector, {-aLit, -cLit, outLit});
    addConditionedClause(selector, {-bLit, -cLit, outLit});
    addConditionedClause(selector, {aLit, bLit, -outLit});
    addConditionedClause(selector, {aLit, cLit, -outLit});
    addConditionedClause(selector, {bLit, cLit, -outLit});
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
      enumerateExactMIGCandidates(availableSources, stepCandidates[step]);
      auto &selectionVars = stepSelectionVars[step];
      selectionVars.reserve(stepCandidates[step].size());
      for (size_t i = 0, e = stepCandidates[step].size(); i != e; ++i)
        selectionVars.push_back(newVar());
      addExactlyOne(selectionVars);

      unsigned outSource = 1 + numInputs + step;
      for (auto [candidateIndex, candidate] :
           llvm::enumerate(stepCandidates[step])) {
        int selector = selectionVars[candidateIndex];
        for (unsigned minterm = 0; minterm != numMinterms; ++minterm) {
          addConditionedMajority(
              selector, getSourceValueVar(outSource, minterm),
              getSourceLiteral(candidate.fanins[0], minterm,
                               candidate.inverted[0]),
              getSourceLiteral(candidate.fanins[1], minterm,
                               candidate.inverted[1]),
              getSourceLiteral(candidate.fanins[2], minterm,
                               candidate.inverted[2]));
        }
      }
    }

    unsigned rootSource = totalSources - 1;
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({target[minterm] ? getSourceValueVar(rootSource, minterm)
                                        : -getSourceValueVar(rootSource, minterm)});
  }

  ExactMIGNetwork decodeModel() const {
    ExactMIGNetwork network;
    network.numInputs = numInputs;
    network.steps.reserve(numSteps);

    for (unsigned step = 0; step != numSteps; ++step) {
      const auto &selectionVars = stepSelectionVars[step];
      const auto &candidates = stepCandidates[step];
      for (size_t i = 0, e = selectionVars.size(); i != e; ++i) {
        if (solver.val(selectionVars[i]) != selectionVars[i])
          continue;
        ExactMIGStep resultStep;
        for (unsigned operand = 0; operand != 3; ++operand)
          resultStep.fanins[operand] = {candidates[i].fanins[operand],
                                        candidates[i].inverted[operand]};
        network.steps.push_back(resultStep);
        break;
      }
    }

    network.output = {1 + numInputs + numSteps - 1, false};
    return network;
  }

  IncrementalSATSolver &solver;
  unsigned numInputs;
  llvm::APInt target;
  unsigned numSteps;
  unsigned numMinterms;
  unsigned totalSources;
  int nextVar = 0;
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  SmallVector<SmallVector<ExactMIGCandidate, 64>, 8> stepCandidates;
  SmallVector<SmallVector<int, 64>, 8> stepSelectionVars;
};

class ExactMIGSynthesizer {
public:
  enum class QueryStatus {
    Solved,
    NoSolution,
    ConflictLimitReached,
    Error,
  };

  struct QueryResult {
    QueryStatus status = QueryStatus::Error;
    std::optional<ExactMIGNetwork> network;
  };

  explicit ExactMIGSynthesizer(StringRef backend, int conflictLimit)
      : backend(backend), conflictLimit(conflictLimit) {}

  QueryResult synthesizeForArea(const BinaryTruthTable &tt, unsigned area) const {
    auto [normalizedTT, invertOutput] = normalize(tt);
    auto result =
        synthesizeNormalized(normalizedTT.numInputs, normalizedTT.table, area);
    if (result.status != QueryStatus::Solved || !result.network)
      return result;

    ExactMIGNetwork network = *result.network;
    if (invertOutput)
      network = invertExactMIGOutput(std::move(network));
    return {.status = QueryStatus::Solved,
            .network = std::optional<ExactMIGNetwork>(std::move(network))};
  }

private:
  static std::pair<BinaryTruthTable, bool>
  normalize(const BinaryTruthTable &tt) {
    if (!tt.table[0])
      return {tt, false};
    BinaryTruthTable normalized = tt;
    normalized.table.flipAllBits();
    return {std::move(normalized), true};
  }

  std::optional<ExactMIGNetwork>
  synthesizeDirect(unsigned numInputs, const llvm::APInt &target) const {
    ExactMIGNetwork network;
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

  QueryResult synthesizeNormalized(unsigned numInputs, const llvm::APInt &target,
                                   unsigned area) const {
    if (area == 0) {
      auto direct = synthesizeDirect(numInputs, target);
      return {.status = direct ? QueryStatus::Solved : QueryStatus::NoSolution,
              .network = std::move(direct)};
    }

    auto solver = createExactSATSolver(backend, conflictLimit);
    if (!solver)
      return {.status = QueryStatus::Error};

    ExactMIGSATProblem problem(*solver, numInputs, target, area);
    auto solveResult = problem.solve();
    if (solveResult.result == IncrementalSATSolver::kUNKNOWN)
      return {.status = QueryStatus::ConflictLimitReached};
    return {.status = solveResult.network ? QueryStatus::Solved
                                          : QueryStatus::NoSolution,
            .network = std::move(solveResult.network)};
  }

  std::string backend;
  int conflictLimit;
};

static ExactMIGNetwork remapExactMIGToCutInputs(ExactMIGNetwork network,
                                                const NPNClass &npnClass) {
  assert(network.numInputs == npnClass.truthTable.numInputs &&
         "network and NPN class input size mismatch");

  auto remapSignal = [&](ExactSignalRef &signal) {
    if (signal.source == 0 || signal.source > network.numInputs)
      return;
    unsigned canonicalInput = signal.source - 1;
    signal.source = npnClass.inputPermutation[canonicalInput] + 1;
    if ((npnClass.inputNegation >> canonicalInput) & 1)
      signal.inverted = !signal.inverted;
  };

  for (auto &step : network.steps)
    for (auto &fanin : step.fanins)
      remapSignal(fanin);
  remapSignal(network.output);

  if (npnClass.outputNegation & 1)
    network = invertExactMIGOutput(std::move(network));
  return network;
}

static SmallVector<DelayType, 6>
computeExactMIGInputDelays(const ExactMIGNetwork &network) {
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

static unsigned computeMaterializedExactMIGArea(const ExactMIGNetwork &network) {
  unsigned area = network.steps.size();
  if (network.output.inverted && network.output.source != 0)
    ++area;
  return area;
}

static Value materializeExactMIGNetwork(OpBuilder &builder, Location loc,
                                        ValueRange inputs,
                                        const ExactMIGNetwork &network) {
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

static FailureOr<ExactMIGNetwork>
parseExactMIGNetworkFromModule(hw::HWModuleOp module) {
  auto inputTypes = module.getInputTypes();
  auto outputTypes = module.getOutputTypes();
  if (outputTypes.size() != 1)
    return module.emitError(
        "cut-rewrite MIG database modules must have a single output");
  for (Type type : inputTypes)
    if (!type.isInteger(1))
      return module.emitError(
          "cut-rewrite MIG database module inputs must be i1");
  for (Type type : outputTypes)
    if (!type.isInteger(1))
      return module.emitError(
          "cut-rewrite MIG database module outputs must be i1");

  ExactMIGNetwork network;
  network.numInputs = module.getNumInputPorts();
  auto *bodyBlock = module.getBodyBlock();
  DenseMap<Value, ExactSignalRef> valueToSignal;
  for (auto [index, argument] : llvm::enumerate(bodyBlock->getArguments()))
    valueToSignal[argument] = {static_cast<unsigned>(index + 1), false};

  for (Operation &op : bodyBlock->without_terminator()) {
    if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
      if (!constant.getType().isInteger(1))
        return constant.emitError(
            "cut-rewrite MIG database constants must be i1");
      valueToSignal[constant.getResult()] = {0,
                                             constant.getValueAttr().getValue()[0]};
      continue;
    }

    if (auto wire = dyn_cast<hw::WireOp>(op)) {
      auto signal = lookupExactSignal(wire.getInput(), valueToSignal, wire, "MIG");
      if (failed(signal))
        return failure();
      valueToSignal[wire.getResult()] = *signal;
      continue;
    }

    if (auto majority = dyn_cast<synth::mig::MajorityInverterOp>(op)) {
      SmallVector<ExactSignalRef> operands;
      operands.reserve(majority.getInputs().size());
      for (auto [index, operand] : llvm::enumerate(majority.getInputs())) {
        auto signal =
            lookupExactSignal(operand, valueToSignal, majority, "MIG");
        if (failed(signal))
          return failure();
        if (majority.isInverted(index))
          signal->inverted = !signal->inverted;
        operands.push_back(*signal);
      }

      if (operands.size() == 1) {
        valueToSignal[majority.getResult()] = operands.front();
        continue;
      }
      if (operands.size() != 3)
        return majority.emitError(
            "cut-rewrite MIG database currently supports only 1-input "
            "inverters and 3-input majorities");

      network.steps.push_back({{operands[0], operands[1], operands[2]}});
      valueToSignal[majority.getResult()] = {
          static_cast<unsigned>(network.numInputs + network.steps.size()),
          false};
      continue;
    }

    return op.emitError("unsupported operation in cut-rewrite MIG database");
  }

  auto output = dyn_cast<hw::OutputOp>(bodyBlock->getTerminator());
  if (!output || output.getNumOperands() != 1)
    return module.emitError(
        "cut-rewrite MIG database module must terminate with a single-output "
        "hw.output");

  auto outputSignal =
      lookupExactSignal(output.getOperand(0), valueToSignal, output, "MIG");
  if (failed(outputSignal))
    return failure();
  network.output = *outputSignal;
  return network;
}

struct LoadedExactMIGEntry : public LoadedExactSynthesisEntry {
  ExactMIGNetwork network;

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    const auto &logicNetwork = enumerator.getLogicNetwork();
    ExactMIGNetwork exactNetwork = remapExactMIGToCutInputs(network, cut.getNPNClass());
    auto *rootOp = logicNetwork.getGate(cut.getRootIndex()).getOperation();
    SmallVector<Value> inputs;
    inputs.reserve(cut.getInputSize());
    for (uint32_t inputIndex : cut.inputs)
      inputs.push_back(logicNetwork.getValue(inputIndex));
    return materializeExactMIGNetwork(builder, rootOp->getLoc(), inputs,
                                      exactNetwork)
        .getDefiningOp();
  }
};

class MIGExactSynthesisBackend final : public ExactSynthesisBackend {
public:
  StringRef getKind() const override { return "mig-exact"; }
  StringRef getModulePrefix() const override { return "mig_exact"; }
  unsigned getMaxSupportedInputs() const override {
    return kMaxExactSynthesisInputs;
  }
  StringRef getFamilyName() const override { return "MIG"; }

  LogicalResult
  emitDatabase(ModuleOp module,
               const ExactSynthesisDatabaseGenOptions &options) const override {
    if (options.maxInputs > getMaxSupportedInputs()) {
      module.emitError() << "MIG exact database generation supports at most "
                         << getMaxSupportedInputs() << " inputs";
      return failure();
    }
    if (options.conflictLimit < -1) {
      module.emitError()
          << "'conflict-limit' must be greater than or equal to -1";
      return failure();
    }
    if (!createExactSATSolver(options.satSolver, options.conflictLimit)) {
      module.emitError()
          << "Exact MIG database generation requires a SAT solver backend '"
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

      std::vector<std::optional<ExactMIGNetwork>> exactNetworks(
          canonicalTruthTables.size());
      if (failed(mlir::failableParallelForEachN(
              context, 0, canonicalTruthTables.size(), [&](size_t index) {
                ExactMIGSynthesizer synthesizer(options.satSolver,
                                                options.conflictLimit);
                SmallString<32> ttString;
                canonicalTruthTables[index].table.toStringUnsigned(ttString, 16);
                bool hitConflictLimit = false;
                for (unsigned area = 0; area <= kMaxMIGExactSearchArea; ++area) {
                  auto result =
                      synthesizer.synthesizeForArea(canonicalTruthTables[index],
                                                    area);
                  if (result.status ==
                      ExactMIGSynthesizer::QueryStatus::ConflictLimitReached) {
                    hitConflictLimit = true;
                    LLVM_DEBUG(llvm::dbgs()
                               << "ExactMIG dbgen: conflict limit reached for tt="
                               << ttString << " at area=" << area << "\n");
                    break;
                  }
                  if (result.status ==
                      ExactMIGSynthesizer::QueryStatus::Error) {
                    module.emitError()
                        << "failed to synthesize exact MIG for canonical truth "
                           "table "
                        << ttString;
                    return failure();
                  }
                  if (result.status !=
                          ExactMIGSynthesizer::QueryStatus::Solved ||
                      !result.network)
                    continue;

                  exactNetworks[index] = std::move(*result.network);
                  break;
                }
                if (!exactNetworks[index] && !hitConflictLimit) {
                  module.emitError()
                      << "failed to synthesize exact MIG for canonical truth "
                         "table "
                      << ttString;
                  return failure();
                }
                if (hitConflictLimit)
                  module.emitWarning()
                      << "stopping exact MIG search for canonical truth table "
                      << ttString << " due to SAT conflict limit";
                return success();
              })))
        return failure();

      for (auto [index, canonicalTT] : llvm::enumerate(canonicalTruthTables)) {
        if (!exactNetworks[index])
          continue;
        const ExactMIGNetwork &exactNetwork = *exactNetworks[index];
        auto delays = computeExactMIGInputDelays(exactNetwork);
        appendDatabaseEntry(
            module, builder, attrBuilder, getModulePrefix(), canonicalTT, 0,
            createTechInfo(attrBuilder,
                           computeMaterializedExactMIGArea(exactNetwork), delays),
            [&](OpBuilder &entryBuilder, Location loc, ValueRange inputs) {
              return materializeExactMIGNetwork(entryBuilder, loc, inputs,
                                               exactNetwork);
            });
      }
    }

    return success();
  }

  FailureOr<std::unique_ptr<LoadedExactSynthesisEntry>>
  parseEntry(hw::HWModuleOp module) const override {
    auto canonicalTT = getCanonicalTruthTable(module);
    if (failed(canonicalTT))
      return failure();
    auto network = parseExactMIGNetworkFromModule(module);
    if (failed(network))
      return failure();

    auto entry = std::make_unique<LoadedExactMIGEntry>();
    entry->moduleName = module.getModuleName().str();
    entry->network = std::move(*network);
    entry->npnClass = getIdentityNPNClass(*canonicalTT);
    entry->delay = computeExactMIGInputDelays(entry->network);
    entry->area = computeMaterializedExactMIGArea(entry->network);
    std::unique_ptr<LoadedExactSynthesisEntry> result = std::move(entry);
    return result;
  }
};

//===----------------------------------------------------------------------===//
// AIG Exact Synthesis Backend
//===----------------------------------------------------------------------===//

struct ExactAIGStep {
  std::array<ExactSignalRef, 2> fanins;
};

struct ExactAIGNetwork {
  unsigned numInputs = 0;
  SmallVector<ExactAIGStep, 4> steps;
  ExactSignalRef output;
};

struct ExactAIGCandidate {
  std::array<unsigned, 2> fanins;
  std::array<bool, 2> inverted;
};

static void
enumerateExactAIGCandidates(unsigned availableSources,
                            SmallVectorImpl<ExactAIGCandidate> &candidates) {
  candidates.clear();
  for (unsigned a = 0; a + 1 < availableSources; ++a)
    for (unsigned b = a + 1; b < availableSources; ++b)
      for (unsigned invMask = 0; invMask != 4; ++invMask)
        candidates.push_back(
            {{{a, b}},
             {{static_cast<bool>(invMask & 1),
               static_cast<bool>(invMask & 2)}}});
}

class ExactAIGSATProblem {
public:
  struct SolveResult {
    IncrementalSATSolver::Result result = IncrementalSATSolver::kUNKNOWN;
    std::optional<ExactAIGNetwork> network;
  };

  ExactAIGSATProblem(IncrementalSATSolver &solver, unsigned numInputs,
                     const llvm::APInt &target, unsigned numSteps)
      : solver(solver), numInputs(numInputs), target(target),
        numSteps(numSteps), numMinterms(1u << numInputs),
        totalSources(1 + numInputs + numSteps) {}

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

  void addConditionedClause(int selector, std::initializer_list<int> lits) {
    SmallVector<int, 8> clause;
    clause.reserve(lits.size() + 1);
    clause.push_back(-selector);
    clause.append(lits.begin(), lits.end());
    solver.addClause(clause);
  }

  void addConditionedAnd(int selector, int outLit, int aLit, int bLit) {
    addConditionedClause(selector, {-aLit, -bLit, outLit});
    addConditionedClause(selector, {aLit, -outLit});
    addConditionedClause(selector, {bLit, -outLit});
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
      enumerateExactAIGCandidates(availableSources, stepCandidates[step]);
      auto &selectionVars = stepSelectionVars[step];
      selectionVars.reserve(stepCandidates[step].size());
      for (size_t i = 0, e = stepCandidates[step].size(); i != e; ++i)
        selectionVars.push_back(newVar());
      addExactlyOne(selectionVars);

      unsigned outSource = 1 + numInputs + step;
      for (auto [candidateIndex, candidate] :
           llvm::enumerate(stepCandidates[step])) {
        int selector = selectionVars[candidateIndex];
        for (unsigned minterm = 0; minterm != numMinterms; ++minterm) {
          addConditionedAnd(selector, getSourceValueVar(outSource, minterm),
                            getSourceLiteral(candidate.fanins[0], minterm,
                                             candidate.inverted[0]),
                            getSourceLiteral(candidate.fanins[1], minterm,
                                             candidate.inverted[1]));
        }
      }
    }

    unsigned rootSource = totalSources - 1;
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({target[minterm] ? getSourceValueVar(rootSource, minterm)
                                        : -getSourceValueVar(rootSource, minterm)});
  }

  ExactAIGNetwork decodeModel() const {
    ExactAIGNetwork network;
    network.numInputs = numInputs;
    network.steps.reserve(numSteps);

    for (unsigned step = 0; step != numSteps; ++step) {
      const auto &selectionVars = stepSelectionVars[step];
      const auto &candidates = stepCandidates[step];
      for (size_t i = 0, e = selectionVars.size(); i != e; ++i) {
        if (solver.val(selectionVars[i]) != selectionVars[i])
          continue;
        ExactAIGStep resultStep;
        for (unsigned operand = 0; operand != 2; ++operand)
          resultStep.fanins[operand] = {candidates[i].fanins[operand],
                                        candidates[i].inverted[operand]};
        network.steps.push_back(resultStep);
        break;
      }
    }

    network.output = {1 + numInputs + numSteps - 1, false};
    return network;
  }

  IncrementalSATSolver &solver;
  unsigned numInputs;
  llvm::APInt target;
  unsigned numSteps;
  unsigned numMinterms;
  unsigned totalSources;
  int nextVar = 0;
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  SmallVector<SmallVector<ExactAIGCandidate, 64>, 8> stepCandidates;
  SmallVector<SmallVector<int, 64>, 8> stepSelectionVars;
};

class ExactAIGSynthesizer {
public:
  enum class QueryStatus {
    Solved,
    NoSolution,
    ConflictLimitReached,
    Error,
  };

  struct QueryResult {
    QueryStatus status = QueryStatus::Error;
    std::optional<ExactAIGNetwork> network;
  };

  explicit ExactAIGSynthesizer(StringRef backend, int conflictLimit)
      : backend(backend), conflictLimit(conflictLimit) {}

  QueryResult synthesizeForArea(const BinaryTruthTable &tt, unsigned area) const {
    if (area == 0) {
      auto direct = synthesizeDirect(tt.numInputs, tt.table);
      return {.status = direct ? QueryStatus::Solved : QueryStatus::NoSolution,
              .network = std::move(direct)};
    }

    auto solver = createExactSATSolver(backend, conflictLimit);
    if (!solver)
      return {.status = QueryStatus::Error};

    ExactAIGSATProblem problem(*solver, tt.numInputs, tt.table, area);
    auto solveResult = problem.solve();
    if (solveResult.result == IncrementalSATSolver::kUNKNOWN)
      return {.status = QueryStatus::ConflictLimitReached};
    return {.status = solveResult.network ? QueryStatus::Solved
                                          : QueryStatus::NoSolution,
            .network = std::move(solveResult.network)};
  }

private:
  std::optional<ExactAIGNetwork>
  synthesizeDirect(unsigned numInputs, const llvm::APInt &target) const {
    ExactAIGNetwork network;
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

  std::string backend;
  int conflictLimit;
};

static ExactAIGNetwork remapExactAIGToCutInputs(ExactAIGNetwork network,
                                                const NPNClass &npnClass) {
  assert(network.numInputs == npnClass.truthTable.numInputs &&
         "network and NPN class input size mismatch");

  auto remapSignal = [&](ExactSignalRef &signal) {
    if (signal.source == 0 || signal.source > network.numInputs)
      return;
    unsigned canonicalInput = signal.source - 1;
    signal.source = npnClass.inputPermutation[canonicalInput] + 1;
    if ((npnClass.inputNegation >> canonicalInput) & 1)
      signal.inverted = !signal.inverted;
  };

  for (auto &step : network.steps)
    for (auto &fanin : step.fanins)
      remapSignal(fanin);
  remapSignal(network.output);
  if (npnClass.outputNegation & 1)
    network.output.inverted = !network.output.inverted;
  return network;
}

static SmallVector<DelayType, 6>
computeExactAIGInputDelays(const ExactAIGNetwork &network) {
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

static unsigned computeMaterializedExactAIGArea(const ExactAIGNetwork &network) {
  unsigned area = network.steps.size();
  if (network.output.inverted && network.output.source != 0)
    ++area;
  return area;
}

static Value materializeExactAIGNetwork(OpBuilder &builder, Location loc,
                                        ValueRange inputs,
                                        const ExactAIGNetwork &network) {
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

static FailureOr<ExactAIGNetwork>
parseExactAIGNetworkFromModule(hw::HWModuleOp module) {
  auto inputTypes = module.getInputTypes();
  auto outputTypes = module.getOutputTypes();
  if (outputTypes.size() != 1)
    return module.emitError(
        "cut-rewrite AIG database modules must have a single output");
  for (Type type : inputTypes)
    if (!type.isInteger(1))
      return module.emitError(
          "cut-rewrite AIG database module inputs must be i1");
  for (Type type : outputTypes)
    if (!type.isInteger(1))
      return module.emitError(
          "cut-rewrite AIG database module outputs must be i1");

  ExactAIGNetwork network;
  network.numInputs = module.getNumInputPorts();
  auto *bodyBlock = module.getBodyBlock();
  DenseMap<Value, ExactSignalRef> valueToSignal;
  for (auto [index, argument] : llvm::enumerate(bodyBlock->getArguments()))
    valueToSignal[argument] = {static_cast<unsigned>(index + 1), false};

  for (Operation &op : bodyBlock->without_terminator()) {
    if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
      if (!constant.getType().isInteger(1))
        return constant.emitError(
            "cut-rewrite AIG database constants must be i1");
      valueToSignal[constant.getResult()] = {0,
                                             constant.getValueAttr().getValue()[0]};
      continue;
    }

    if (auto wire = dyn_cast<hw::WireOp>(op)) {
      auto signal = lookupExactSignal(wire.getInput(), valueToSignal, wire, "AIG");
      if (failed(signal))
        return failure();
      valueToSignal[wire.getResult()] = *signal;
      continue;
    }

    if (auto andOp = dyn_cast<synth::aig::AndInverterOp>(op)) {
      SmallVector<ExactSignalRef> operands;
      operands.reserve(andOp.getInputs().size());
      for (auto [index, operand] : llvm::enumerate(andOp.getInputs())) {
        auto signal = lookupExactSignal(operand, valueToSignal, andOp, "AIG");
        if (failed(signal))
          return failure();
        if (andOp.isInverted(index))
          signal->inverted = !signal->inverted;
        operands.push_back(*signal);
      }

      if (operands.size() == 1) {
        valueToSignal[andOp.getResult()] = operands.front();
        continue;
      }
      if (operands.size() != 2)
        return andOp.emitError(
            "cut-rewrite AIG database currently supports only 1-input "
            "inverters and 2-input AND nodes");

      network.steps.push_back({{operands[0], operands[1]}});
      valueToSignal[andOp.getResult()] = {
          static_cast<unsigned>(network.numInputs + network.steps.size()),
          false};
      continue;
    }

    return op.emitError("unsupported operation in cut-rewrite AIG database");
  }

  auto output = dyn_cast<hw::OutputOp>(bodyBlock->getTerminator());
  if (!output || output.getNumOperands() != 1)
    return module.emitError(
        "cut-rewrite AIG database module must terminate with a single-output "
        "hw.output");

  auto outputSignal =
      lookupExactSignal(output.getOperand(0), valueToSignal, output, "AIG");
  if (failed(outputSignal))
    return failure();
  network.output = *outputSignal;
  return network;
}

struct LoadedExactAIGEntry : public LoadedExactSynthesisEntry {
  ExactAIGNetwork network;

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    const auto &logicNetwork = enumerator.getLogicNetwork();
    ExactAIGNetwork exactNetwork = remapExactAIGToCutInputs(network, cut.getNPNClass());
    auto *rootOp = logicNetwork.getGate(cut.getRootIndex()).getOperation();
    SmallVector<Value> inputs;
    inputs.reserve(cut.getInputSize());
    for (uint32_t inputIndex : cut.inputs)
      inputs.push_back(logicNetwork.getValue(inputIndex));
    return materializeExactAIGNetwork(builder, rootOp->getLoc(), inputs,
                                      exactNetwork)
        .getDefiningOp();
  }
};

class AIGExactSynthesisBackend final : public ExactSynthesisBackend {
public:
  StringRef getKind() const override { return "aig-exact"; }
  StringRef getModulePrefix() const override { return "aig_exact"; }
  unsigned getMaxSupportedInputs() const override {
    return kMaxExactSynthesisInputs;
  }
  StringRef getFamilyName() const override { return "AIG"; }

  LogicalResult
  emitDatabase(ModuleOp module,
               const ExactSynthesisDatabaseGenOptions &options) const override {
    if (options.maxInputs > getMaxSupportedInputs()) {
      module.emitError() << "AIG exact database generation supports at most "
                         << getMaxSupportedInputs() << " inputs";
      return failure();
    }
    if (options.conflictLimit < -1) {
      module.emitError()
          << "'conflict-limit' must be greater than or equal to -1";
      return failure();
    }
    if (!createExactSATSolver(options.satSolver, options.conflictLimit)) {
      module.emitError()
          << "Exact AIG database generation requires a SAT solver backend '"
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

      std::vector<std::optional<ExactAIGNetwork>> exactNetworks(
          canonicalTruthTables.size());
      if (failed(mlir::failableParallelForEachN(
              context, 0, canonicalTruthTables.size(), [&](size_t index) {
                ExactAIGSynthesizer synthesizer(options.satSolver,
                                                options.conflictLimit);
                SmallString<32> ttString;
                canonicalTruthTables[index].table.toStringUnsigned(ttString, 16);
                bool hitConflictLimit = false;
                for (unsigned area = 0; area <= kMaxAIGExactSearchArea; ++area) {
                  auto result =
                      synthesizer.synthesizeForArea(canonicalTruthTables[index],
                                                    area);
                  if (result.status ==
                      ExactAIGSynthesizer::QueryStatus::ConflictLimitReached) {
                    hitConflictLimit = true;
                    LLVM_DEBUG(llvm::dbgs()
                               << "ExactAIG dbgen: conflict limit reached for tt="
                               << ttString << " at area=" << area << "\n");
                    break;
                  }
                  if (result.status ==
                      ExactAIGSynthesizer::QueryStatus::Error) {
                    module.emitError()
                        << "failed to synthesize exact AIG for canonical truth "
                           "table "
                        << ttString;
                    return failure();
                  }
                  if (result.status !=
                          ExactAIGSynthesizer::QueryStatus::Solved ||
                      !result.network)
                    continue;

                  exactNetworks[index] = std::move(*result.network);
                  break;
                }
                if (!exactNetworks[index] && !hitConflictLimit) {
                  module.emitError()
                      << "failed to synthesize exact AIG for canonical truth "
                         "table "
                      << ttString;
                  return failure();
                }
                if (hitConflictLimit)
                  module.emitWarning()
                      << "stopping exact AIG search for canonical truth table "
                      << ttString << " due to SAT conflict limit";
                return success();
              })))
        return failure();

      for (auto [index, canonicalTT] : llvm::enumerate(canonicalTruthTables)) {
        if (!exactNetworks[index])
          continue;
        const ExactAIGNetwork &exactNetwork = *exactNetworks[index];
        auto delays = computeExactAIGInputDelays(exactNetwork);
        appendDatabaseEntry(
            module, builder, attrBuilder, getModulePrefix(), canonicalTT, 0,
            createTechInfo(attrBuilder,
                           computeMaterializedExactAIGArea(exactNetwork), delays),
            [&](OpBuilder &entryBuilder, Location loc, ValueRange inputs) {
              return materializeExactAIGNetwork(entryBuilder, loc, inputs,
                                               exactNetwork);
            });
      }
    }

    return success();
  }

  FailureOr<std::unique_ptr<LoadedExactSynthesisEntry>>
  parseEntry(hw::HWModuleOp module) const override {
    auto canonicalTT = getCanonicalTruthTable(module);
    if (failed(canonicalTT))
      return failure();
    auto network = parseExactAIGNetworkFromModule(module);
    if (failed(network))
      return failure();

    auto entry = std::make_unique<LoadedExactAIGEntry>();
    entry->moduleName = module.getModuleName().str();
    entry->network = std::move(*network);
    entry->npnClass = getIdentityNPNClass(*canonicalTT);
    entry->delay = computeExactAIGInputDelays(entry->network);
    entry->area = computeMaterializedExactAIGArea(entry->network);
    std::unique_ptr<LoadedExactSynthesisEntry> result = std::move(entry);
    return result;
  }
};

static const ExactSynthesisBackend *getExactSynthesisBackend(StringRef kind) {
  static const MIGExactSynthesisBackend migBackend;
  static const AIGExactSynthesisBackend aigBackend;

  std::string normalized = normalizeDatabaseKind(kind);
  if (normalized == migBackend.getKind())
    return &migBackend;
  if (normalized == aigBackend.getKind())
    return &aigBackend;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Generic DB Loading and Cut Rewrite Pass
//===----------------------------------------------------------------------===//

static FailureOr<OwningOpRef<mlir::ModuleOp>>
parseCutRewriteDBFile(StringRef dbFile, MLIRContext *context) {
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

static LogicalResult
loadExactSynthesisDatabaseFromModule(mlir::ModuleOp dbModule,
                                     LoadedExactSynthesisDatabase &database) {
  auto kindAttr = dbModule->getAttrOfType<StringAttr>(kCutRewriteDBKindAttr);
  if (!kindAttr)
    return dbModule.emitError("cut-rewrite database missing '")
           << kCutRewriteDBKindAttr << "'";

  std::string normalizedKind = normalizeDatabaseKind(kindAttr.getValue());
  auto *backend = getExactSynthesisBackend(normalizedKind);
  if (!backend)
    return dbModule.emitError("unsupported exact-synthesis database kind '")
           << kindAttr.getValue() << "'";

  database.kind = normalizedKind;
  database.entries.clear();
  database.maxInputSize = 0;

  for (auto hwModule : dbModule.getOps<hw::HWModuleOp>()) {
    auto entry = backend->parseEntry(hwModule);
    if (failed(entry))
      return failure();
    database.maxInputSize = std::max(
        database.maxInputSize, static_cast<unsigned>((*entry)->npnClass.truthTable.numInputs));
    database.entries.push_back(std::move(*entry));
  }

  if (database.entries.empty())
    return dbModule.emitError("cut-rewrite database did not contain any "
                              "matching library entries");
  return success();
}

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
      patterns.push_back(
          std::make_unique<ExactSynthesisPattern>(module->getContext(), *entry));

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

LogicalResult circt::synth::emitExactSynthesisDatabase(
    mlir::ModuleOp module, StringRef kind,
    const ExactSynthesisDatabaseGenOptions &options) {
  std::string normalizedKind = normalizeDatabaseKind(kind);
  auto *backend = getExactSynthesisBackend(normalizedKind);
  if (!backend) {
    module.emitError() << "unsupported database kind '" << kind << "'";
    return failure();
  }

  module->setAttr(kCutRewriteDBKindAttr,
                  StringAttr::get(module.getContext(), backend->getKind()));
  return backend->emitDatabase(module, options);
}
