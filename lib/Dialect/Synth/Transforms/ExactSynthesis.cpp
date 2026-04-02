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

struct ExactSignalRef {
  // `source == 0` denotes the constant-false source. All other sources are
  // numbered as 1-based primary inputs followed by synthesized steps.
  unsigned source = 0;
  bool inverted = false;
};

struct ExactNetworkStep {
  // One backend-specific primitive node in the synthesized network.
  SmallVector<ExactSignalRef, 3> fanins;
};

struct ExactNetwork {
  // Compact backend-independent representation used during SAT search and DB
  // generation. The concrete IR form is produced later by `materializeNetwork`.
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

    // Walk the full NPN orbit of one seed function, mark every equivalent
    // representative as seen, and keep the lexicographically smallest truth
    // table as the canonical class representative.
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
  // Exact-synthesis databases are serialized as one `hw.module` per canonical
  // truth table. The loaded cut-rewrite path later clones these bodies
  // directly instead of reconstructing them from an intermediate graph.
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

  // User-visible identifier accepted by `--kind`.
  virtual StringRef getKind() const = 0;
  // Inverter family stored in each serialized DB entry so the generic loader
  // knows which single-input inverter op to materialize for NPN corrections.
  virtual StringRef getInverterKind() const = 0;
  // Prefix used for generated `hw.module` names in the on-disk database.
  virtual StringRef getModulePrefix() const = 0;
  // Hard limit for this backend family, independent of CLI options.
  virtual unsigned getMaxSupportedInputs() const = 0;
  // Short family label used in diagnostics and debug output.
  virtual StringRef getFamilyName() const = 0;
  // Primitive node fanin count used by the SAT encoding.
  virtual unsigned getArity() const = 0;
  // Maximum area budget explored during exact search.
  virtual unsigned getMaxSearchArea() const = 0;

  // Normalize a truth table into the subset of NPN space this backend chooses
  // to synthesize directly. The returned flag tells the caller whether the
  // synthesized network must be complemented again afterward.
  virtual std::pair<BinaryTruthTable, bool>
  normalize(const BinaryTruthTable &tt) const = 0;
  // Recognize trivial zero-area solutions such as constants and projections.
  // Returning `nullopt` tells the generic solver to build a SAT instance.
  virtual std::optional<ExactNetwork>
  synthesizeDirect(unsigned numInputs, const llvm::APInt &target) const = 0;
  // Apply a deferred output inversion in the backend's native representation.
  // MIG can often absorb this into the last majority; AIG typically cannot.
  virtual ExactNetwork applyOutputNegation(ExactNetwork network) const = 0;
  // Enumerate all primitive nodes that may appear at the next synthesized
  // step, using the currently available constant/input/step sources.
  virtual void
  enumerateCandidates(unsigned availableSources,
                      SmallVectorImpl<ExactCandidate> &candidates) const = 0;
  // Emit CNF clauses for one candidate node at one minterm. `selector` gates
  // the clauses so only the chosen candidate constrains the step output.
  virtual void addConditionedSemantics(
      IncrementalSATSolver &solver, int selector, int outLit,
      const ExactCandidate &candidate, unsigned minterm,
      llvm::function_ref<int(unsigned source, unsigned minterm, bool inverted)>
          getSourceLiteral) const = 0;
  // Lower the abstract `ExactNetwork` into concrete dialect IR when writing
  // database entries.
  virtual Value materializeNetwork(OpBuilder &builder, Location loc,
                                   ValueRange inputs,
                                   const ExactNetwork &network) const = 0;

  // Populate `module` with all canonical database entries supported by this
  // backend.
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
    // Delay is tracked per primary input. Each node contributes unit delay on
    // top of the longest reachable input-to-fanin path.
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

    // Encode pairwise exclusion with a sequential ladder. This keeps the
    // auxiliary-variable cost linear instead of quadratic in `vars.size()`.
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
    // One long positive clause enforces "at least one", and the ladder
    // encoding above enforces "at most one".
    solver.addClause(vars);
    addAtMostOne(vars);
  }

  int getSourceValueVar(unsigned source, unsigned minterm) const {
    return sourceValueVars[source][minterm];
  }

  int getSourceLiteral(unsigned source, unsigned minterm, bool inverted) const {
    // Candidates refer to sources in network numbering, not SAT numbering.
    // This helper converts one source/minterm pair into the corresponding CNF
    // literal, optionally complemented for inverted fanins.
    int lit = getSourceValueVar(source, minterm);
    return inverted ? -lit : lit;
  }

  void buildEncoding() {
    // Every source gets one SAT variable per minterm so the backend can
    // constrain each candidate purely in terms of truth-table semantics.
    sourceValueVars.resize(totalSources);
    for (unsigned source = 0; source != totalSources; ++source) {
      sourceValueVars[source].reserve(numMinterms);
      for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
        sourceValueVars[source].push_back(newVar());
    }

    // Source 0 is the dedicated constant-false source for every backend.
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({-getSourceValueVar(0, minterm)});

    // Primary inputs are fully determined by the minterm index itself: bit
    // `input` of the minterm number is the truth-table value of that input.
    for (unsigned input = 0; input != numInputs; ++input)
      for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
        solver.addClause({((minterm >> input) & 1)
                              ? getSourceValueVar(1 + input, minterm)
                              : -getSourceValueVar(1 + input, minterm)});

    stepCandidates.resize(numSteps);
    stepSelectionVars.resize(numSteps);
    for (unsigned step = 0; step != numSteps; ++step) {
      // A step may use the constant source, any primary input, and any
      // previously synthesized step, but never future steps.
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
        // The selector literal gates the semantics of one candidate. Exactly
        // one selector is true, so the SAT model picks one concrete node
        // implementation per synthesized step.
        //
        // Repeating this for every minterm is what turns structural synthesis
        // into pure Boolean satisfiability: the solver must assign values to
        // all internal sources so that the selected primitive computes the
        // right truth-table row for each minterm independently.
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
    // The last synthesized step is the network root. Constraining its value on
    // every minterm to match `target` forces the entire selected network to
    // implement the requested truth table.
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
        // `addExactlyOne` guarantees a unique selected candidate for each
        // step, so decoding only has to find the one positive selector.
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
  // Total sources in the abstract network numbering:
  //   0                  -> constant false
  //   [1, numInputs]     -> primary inputs
  //   remaining sources  -> synthesized steps in topological order
  unsigned totalSources;
  int nextVar = 0;
  // `sourceValueVars[source][minterm]` is the SAT variable that represents the
  // Boolean value of one source on one truth-table row.
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  // Structural choices available for each synthesized step.
  SmallVector<SmallVector<ExactCandidate, 64>, 8> stepCandidates;
  // One-hot selector variables choosing which candidate realizes each step.
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
      // Area-zero solutions are only constants or projections, so let the
      // backend answer those directly without building a SAT instance.
      auto direct = backend.synthesizeDirect(numInputs, target);
      return {.status = direct ? QueryStatus::Solved : QueryStatus::NoSolution,
              .network = std::move(direct)};
    }

    auto solver = createIncrementalSATSolver(satBackend);
    if (!solver)
      return {.status = QueryStatus::Error};
    solver->setConflictLimit(conflictLimit);

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
  if (!createIncrementalSATSolver(options.satSolver)) {
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
      // Stored database bodies are already in canonical input order. Rebuild
      // the original cut by applying the inverse NPN input permutation and the
      // required input negations while mapping block arguments.
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
    // For MIGs, negating a majority can be absorbed by inverting all three
    // fanins of the final majority node instead of materializing a separate
    // output inverter.
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
    // AIG output negation is represented explicitly because the final node is
    // binary AND rather than a self-dual majority.
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
