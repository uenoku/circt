//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements exact synthesis of truth-table database entries.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/ExactSynthesis.h"
#include "CutRewriteDBImpl.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/SATSolver.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
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

namespace circt {
namespace synth {
#define GEN_PASS_DEF_EXACTSYNTHESIS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

namespace {

static constexpr unsigned kMaxExactSynthesisInputs = 4;
static constexpr unsigned kMaxMIGExactSearchArea = 8;
static constexpr unsigned kMaxAIGExactSearchArea = 10;

enum class CutRewriteInverterKind { mig, aig };

static std::string formatTruthTable(const llvm::APInt &table) {
  SmallString<32> ttString;
  table.toStringUnsigned(ttString, 16);
  return std::string(ttString);
}

static std::string formatTruthTable(const BinaryTruthTable &truthTable) {
  return formatTruthTable(truthTable.table);
}

static std::string normalizeExactSynthesisKind(StringRef kind) {
  std::string normalized = kind.lower();
  for (char &c : normalized)
    if (c == '_')
      c = '-';
  return normalized;
}

static FailureOr<CadicalSATSolverOptions::CadicalSolverConfig>
parseCadicalConfig(Operation *op, StringRef config) {
  if (config == "default")
    return CadicalSATSolverOptions::CadicalSolverConfig::Default;
  if (config == "plain")
    return CadicalSATSolverOptions::CadicalSolverConfig::Plain;
  if (config == "sat")
    return CadicalSATSolverOptions::CadicalSolverConfig::Sat;
  if (config == "unsat")
    return CadicalSATSolverOptions::CadicalSolverConfig::Unsat;
  op->emitError() << "unsupported CaDiCaL configuration '" << config << "'";
  return failure();
}

static FailureOr<CutRewriteInverterKind>
parseCutRewriteInverterKind(Operation *op, StringRef kind) {
  if (kind == "mig")
    return CutRewriteInverterKind::mig;
  if (kind == "aig")
    return CutRewriteInverterKind::aig;
  op->emitError("unsupported cut-rewrite inverter kind '") << kind << "'";
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

class ExactSynthesisBackend {
public:
  virtual ~ExactSynthesisBackend() = default;

  // User-visible identifier accepted by `--kind`.
  virtual StringRef getKind() const = 0;
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
  // Return true if a candidate step computes only a constant or a projection.
  // Such steps can always be removed from an optimum solution.
  virtual bool isTrivialCandidate(const ExactCandidate &candidate) const = 0;
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
};

static const ExactSynthesisBackend *getExactSynthesisBackend(StringRef kind);

class GenericExactSATProblem {
public:
  struct SolveResult {
    IncrementalSATSolver::Result result = IncrementalSATSolver::kUNKNOWN;
    std::optional<ExactNetwork> network;
  };

  GenericExactSATProblem(const ExactSynthesisBackend &backend,
                         IncrementalSATSolver &solver, unsigned numInputs,
                         const llvm::APInt &target)
      : backend(backend), solver(solver), numInputs(numInputs), target(target),
        arity(backend.getArity()), numMinterms(1u << numInputs) {
    initializeBaseEncoding();
  }

  unsigned getNumSteps() const { return stepCandidates.size(); }

  void appendStep() {
    unsigned step = stepCandidates.size();
    appendSourceValueVars();

    stepCandidates.emplace_back();
    stepSelectionVars.emplace_back();

    unsigned availableSources = 1 + numInputs + step;
    backend.enumerateCandidates(availableSources, stepCandidates.back());
    llvm::erase_if(stepCandidates.back(), [&](const ExactCandidate &candidate) {
      return backend.isTrivialCandidate(candidate);
    });

    auto &selectionVars = stepSelectionVars.back();
    selectionVars.reserve(stepCandidates.back().size());
    for (size_t i = 0, e = stepCandidates.back().size(); i != e; ++i)
      selectionVars.push_back(newVar());
    addExactlyOne(selectionVars);

    LLVM_DEBUG(llvm::dbgs()
               << "  step " << step << ": availableSources="
               << availableSources
               << " candidates=" << stepCandidates.back().size() << "\n");

    if (step != 0)
      addAdjacentStepOrdering(step - 1, step);
    addCandidateSemanticsForStep(step);
  }

  SolveResult solveCurrentArea() {
    unsigned numSteps = getNumSteps();
    assert(numSteps != 0 && "area-zero queries do not use SAT");

    int areaAssumption = createAreaAssumption(numSteps);
    LLVM_DEBUG(llvm::dbgs()
               << "Exact SAT solve: family=" << backend.getFamilyName()
               << " inputs=" << numInputs << " steps=" << numSteps
               << " minterms=" << numMinterms << " vars=" << nextVar << "\n");
    auto result = solver.solve({areaAssumption});
    LLVM_DEBUG(llvm::dbgs()
               << "Exact SAT result: family=" << backend.getFamilyName()
               << " inputs=" << numInputs << " steps=" << numSteps
               << " result=" << static_cast<int>(result) << "\n");
    if (result != IncrementalSATSolver::kSAT)
      return {.result = result};
    return {.result = result, .network = decodeModel(numSteps)};
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

  static bool haveSameFaninSources(const ExactCandidate &lhs,
                                   const ExactCandidate &rhs) {
    assert(lhs.fanins.size() == rhs.fanins.size() && "arity mismatch");
    for (auto [lhsFanin, rhsFanin] : llvm::zip_equal(lhs.fanins, rhs.fanins))
      if (lhsFanin.source != rhsFanin.source)
        return false;
    return true;
  }

  static bool isColexLess(const ExactCandidate &lhs,
                          const ExactCandidate &rhs) {
    assert(lhs.fanins.size() == rhs.fanins.size() && "arity mismatch");
    for (int i = static_cast<int>(lhs.fanins.size()) - 1; i >= 0; --i) {
      if (lhs.fanins[i].source < rhs.fanins[i].source)
        return true;
      if (lhs.fanins[i].source > rhs.fanins[i].source)
        return false;
    }
    return false;
  }

  static unsigned getOperatorOrderKey(const ExactCandidate &candidate) {
    unsigned key = 0;
    for (auto [index, fanin] : llvm::enumerate(candidate.fanins))
      if (fanin.inverted)
        key |= 1u << index;
    return key;
  }

  void initializeBaseEncoding() {
    // Every source gets one SAT variable per minterm so the backend can
    // constrain each candidate purely in terms of truth-table semantics.
    sourceValueVars.resize(1 + numInputs);
    for (auto &vars : sourceValueVars) {
      vars.reserve(numMinterms);
      for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
        vars.push_back(newVar());
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
  }

  void appendSourceValueVars() {
    sourceValueVars.emplace_back();
    auto &vars = sourceValueVars.back();
    vars.reserve(numMinterms);
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      vars.push_back(newVar());
  }

  void addAdjacentStepOrdering(unsigned prevStep, unsigned nextStep) {
    const auto &prevCandidates = stepCandidates[prevStep];
    const auto &nextCandidates = stepCandidates[nextStep];
    const auto &prevSelectionVars = stepSelectionVars[prevStep];
    const auto &nextSelectionVars = stepSelectionVars[nextStep];

    for (auto [prevIndex, prevCandidate] : llvm::enumerate(prevCandidates)) {
      SmallVector<int, 64> allowedNextSelections;
      for (auto [nextIndex, nextCandidate] : llvm::enumerate(nextCandidates))
        if (!isColexLess(nextCandidate, prevCandidate))
          allowedNextSelections.push_back(nextSelectionVars[nextIndex]);

      assert(!allowedNextSelections.empty() &&
             "colex symmetry break must leave some valid next-step choices");
      SmallVector<int, 65> clause;
      clause.reserve(allowedNextSelections.size() + 1);
      clause.push_back(-prevSelectionVars[prevIndex]);
      clause.append(allowedNextSelections.begin(), allowedNextSelections.end());
      solver.addClause(clause);
    }

    for (auto [prevIndex, prevCandidate] : llvm::enumerate(prevCandidates))
      for (auto [nextIndex, nextCandidate] : llvm::enumerate(nextCandidates))
        if (haveSameFaninSources(prevCandidate, nextCandidate) &&
            getOperatorOrderKey(nextCandidate) <
                getOperatorOrderKey(prevCandidate))
          solver.addClause(
              {-prevSelectionVars[prevIndex], -nextSelectionVars[nextIndex]});
  }

  void addCandidateSemanticsForStep(unsigned step) {
    unsigned outSource = 1 + numInputs + step;
    const auto &selectionVars = stepSelectionVars[step];
    for (auto [candidateIndex, candidate] : llvm::enumerate(stepCandidates[step])) {
      // The selector literal gates the semantics of one candidate. Exactly
      // one selector is true, so the SAT model picks one concrete node
      // implementation per synthesized step.
      //
      // Repeating this for every minterm is what turns structural synthesis
      // into pure Boolean satisfiability: the solver must assign values to
      // all internal sources so that the selected primitive computes the
      // right truth-table row for each minterm independently.
      backend.addConditionedSemantics(
          solver, selectionVars[candidateIndex], getSourceValueVar(outSource, 0),
          candidate, 0,
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

  int createAreaAssumption(unsigned numSteps) {
    assert(numSteps != 0 && "area-zero queries do not use SAT");
    int assumption = newVar();
    areaAssumptions.push_back(assumption);

    addRootConstraintsForArea(assumption, numSteps);
    addUseAllStepsConstraintsForArea(assumption, numSteps);
    return assumption;
  }

  void addRootConstraintsForArea(int assumption, unsigned numSteps) {
    unsigned rootSource = 1 + numInputs + numSteps - 1;
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({-assumption, target[minterm]
                                         ? getSourceValueVar(rootSource, minterm)
                                         : -getSourceValueVar(rootSource,
                                                              minterm)});
  }

  void addUseAllStepsConstraintsForArea(int assumption, unsigned numSteps) {
    for (unsigned step = 0; step + 1 < numSteps; ++step) {
      unsigned source = 1 + numInputs + step;
      SmallVector<int, 32> users;
      for (unsigned userStep = step + 1; userStep != numSteps; ++userStep)
        for (auto [candidateIndex, candidate] :
             llvm::enumerate(stepCandidates[userStep]))
          if (llvm::any_of(candidate.fanins, [&](const ExactSignalRef &fanin) {
                return fanin.source == source;
              }))
            users.push_back(stepSelectionVars[userStep][candidateIndex]);
      LLVM_DEBUG(llvm::dbgs() << "  use-all-steps: step " << step << " source="
                              << source << " users=" << users.size() << "\n");
      SmallVector<int, 33> clause;
      clause.reserve(users.size() + 1);
      clause.push_back(-assumption);
      clause.append(users.begin(), users.end());
      solver.addClause(clause);
    }
  }

  ExactNetwork decodeModel(unsigned numSteps) const {
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
  unsigned arity;
  unsigned numMinterms;
  int nextVar = 0;
  // `sourceValueVars[source][minterm]` is the SAT variable that represents the
  // Boolean value of one source on one truth-table row.
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  // Structural choices available for each synthesized step.
  SmallVector<SmallVector<ExactCandidate, 64>, 8> stepCandidates;
  // One-hot selector variables choosing which candidate realizes each step.
  SmallVector<SmallVector<int, 64>, 8> stepSelectionVars;
  // One assumption variable per area query.
  SmallVector<int, 8> areaAssumptions;
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
                          StringRef satBackend,
                          CadicalSATSolverOptions cadicalOptions,
                          int conflictLimit)
      : backend(backend), satBackend(satBackend),
        cadicalOptions(cadicalOptions), conflictLimit(conflictLimit) {}

  QueryResult synthesizeForArea(const BinaryTruthTable &tt,
                                unsigned area) const {
    auto [normalizedTT, invertOutput] = backend.normalize(tt);
    LLVM_DEBUG(llvm::dbgs()
               << "Exact synthesis query: family=" << backend.getFamilyName()
               << " area=" << area << " original-tt=" << formatTruthTable(tt)
               << " normalized-tt=" << formatTruthTable(normalizedTT)
               << " invert-output=" << invertOutput << "\n");
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
  unsigned computeMaxReservedVars(unsigned numInputs) const {
    unsigned maxArea = backend.getMaxSearchArea();
    unsigned numMinterms = 1u << numInputs;
    unsigned totalVars = (1 + numInputs + maxArea) * numMinterms;

    for (unsigned step = 0; step != maxArea; ++step) {
      SmallVector<ExactCandidate, 64> candidates;
      unsigned availableSources = 1 + numInputs + step;
      backend.enumerateCandidates(availableSources, candidates);
      llvm::erase_if(candidates, [&](const ExactCandidate &candidate) {
        return backend.isTrivialCandidate(candidate);
      });
      totalVars += candidates.size();
      if (candidates.size() >= 2)
        totalVars += candidates.size() - 1;
    }

    // One assumption literal is created for each positive-area query.
    totalVars += maxArea;
    return totalVars;
  }

  QueryResult synthesizeNormalized(unsigned numInputs,
                                   const llvm::APInt &target,
                                   unsigned area) const {
    if (area == 0) {
      // Area-zero solutions are only constants or projections, so let the
      // backend answer those directly without building a SAT instance.
      auto direct = backend.synthesizeDirect(numInputs, target);
      LLVM_DEBUG(llvm::dbgs()
                 << "Exact synthesis direct query: family="
                 << backend.getFamilyName() << " inputs=" << numInputs
                 << " tt=" << formatTruthTable(target)
                 << " solved=" << static_cast<bool>(direct) << "\n");
      return {.status = direct ? QueryStatus::Solved : QueryStatus::NoSolution,
              .network = std::move(direct)};
    }

    if (!solver) {
      solver = createIncrementalSATSolver(satBackend, cadicalOptions);
      if (!solver)
        return {.status = QueryStatus::Error};
      solver->reserveVars(computeMaxReservedVars(numInputs));
      solver->setConflictLimit(conflictLimit);
    }
    if (!problem)
      problem =
          std::make_unique<GenericExactSATProblem>(backend, *solver, numInputs,
                                                  target);

    while (problem->getNumSteps() < area)
      problem->appendStep();

    auto solveResult = problem->solveCurrentArea();
    if (solveResult.result == IncrementalSATSolver::kUNKNOWN)
      return {.status = QueryStatus::ConflictLimitReached};
    return {.status = solveResult.network ? QueryStatus::Solved
                                          : QueryStatus::NoSolution,
            .network = std::move(solveResult.network)};
  }

  const ExactSynthesisBackend &backend;
  std::string satBackend;
  CadicalSATSolverOptions cadicalOptions;
  int conflictLimit;
  mutable std::unique_ptr<IncrementalSATSolver> solver;
  mutable std::unique_ptr<GenericExactSATProblem> problem;
};

static FailureOr<BinaryTruthTable>
parseTruthTableTargetFromModule(hw::HWModuleOp module) {
  auto *bodyBlock = module.getBodyBlock();
  auto output = dyn_cast<hw::OutputOp>(bodyBlock->getTerminator());
  if (!output || output.getNumOperands() != 1)
    return module.emitError("exact synthesis requires a single-output "
                            "hw.module");

  for (Type type : module.getInputTypes())
    if (!type.isInteger(1))
      return module.emitError(
          "exact synthesis only supports single-bit inputs");
  for (Type type : module.getOutputTypes())
    if (!type.isInteger(1))
      return module.emitError(
          "exact synthesis only supports single-bit outputs");

  auto truthTableOp = output.getOperand(0).getDefiningOp<comb::TruthTableOp>();
  if (!truthTableOp)
    return module.emitError("exact synthesis expects hw.output to be driven by "
                            "a comb.truth_table");
  if (truthTableOp->getBlock() != bodyBlock)
    return module.emitError("exact synthesis expects the comb.truth_table to "
                            "live in the module body");
  if (truthTableOp.getInputs().size() != bodyBlock->getNumArguments())
    return truthTableOp.emitError("truth table input count must match module "
                                  "input count");

  // The predefined generator emits operands in reverse port order so module
  // input 0 remains the LSB in `BinaryTruthTable`.
  for (auto [index, operand] : llvm::enumerate(truthTableOp.getInputs()))
    if (operand !=
        bodyBlock->getArgument(bodyBlock->getNumArguments() - 1 - index))
      return truthTableOp.emitError("truth table operands must list module "
                                    "inputs in reverse order");

  auto tableAttr = truthTableOp.getLookupTable();
  llvm::APInt table(tableAttr.size(), 0);
  for (auto [index, bit] : llvm::enumerate(tableAttr))
    table.setBitVal(index, bit);
  LLVM_DEBUG(llvm::dbgs() << "Parsed truth-table module "
                          << module.getModuleName() << ": "
                          << "inputs=" << bodyBlock->getNumArguments()
                          << " tt=" << formatTruthTable(table) << "\n");
  return BinaryTruthTable(bodyBlock->getNumArguments(), 1, table);
}

static FailureOr<std::optional<ExactNetwork>>
exactSynthesizeTruthTableForBackend(const ExactSynthesisBackend &backend,
                                    const BinaryTruthTable &target,
                                    const ExactSynthesisRunOptions &options,
                                    Operation *op) {
  auto cadicalConfig = parseCadicalConfig(op, options.cadicalConfig);
  if (failed(cadicalConfig))
    return failure();
  CadicalSATSolverOptions cadicalOptions;
  cadicalOptions.config = *cadicalConfig;
  GenericExactSynthesizer synthesizer(backend, options.satSolver,
                                      cadicalOptions, options.conflictLimit);
  SmallString<32> ttString;
  target.table.toStringUnsigned(ttString, 16);
  for (unsigned area = 0; area <= backend.getMaxSearchArea(); ++area) {
    auto result = synthesizer.synthesizeForArea(target, area);
    if (result.status ==
        GenericExactSynthesizer::QueryStatus::ConflictLimitReached) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Exact synthesis stopped at conflict limit: family="
                 << backend.getFamilyName() << " tt=" << ttString
                 << " area=" << area << "\n");
      return std::optional<ExactNetwork>();
    }
    if (result.status == GenericExactSynthesizer::QueryStatus::Error) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Exact synthesis error: family=" << backend.getFamilyName()
                 << " tt=" << ttString << " area=" << area << "\n");
      op->emitError() << "failed to synthesize exact "
                      << backend.getFamilyName() << " for truth table "
                      << ttString;
      return failure();
    }
    if (result.status == GenericExactSynthesizer::QueryStatus::Solved &&
        result.network) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Exact synthesis solved: family=" << backend.getFamilyName()
                 << " tt=" << ttString << " area=" << area
                 << " steps=" << result.network->steps.size() << "\n");
      return std::optional<ExactNetwork>(std::move(*result.network));
    }
    LLVM_DEBUG(llvm::dbgs() << "Exact synthesis no solution at area: family="
                            << backend.getFamilyName() << " tt=" << ttString
                            << " area=" << area << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Exact synthesis exhausted search area: family="
                          << backend.getFamilyName() << " tt=" << ttString
                          << " max-area=" << backend.getMaxSearchArea()
                          << "\n");
  return std::optional<ExactNetwork>();
}

static FailureOr<std::optional<ExactNetwork>>
exactSynthesizeModuleForBackend(const ExactSynthesisBackend &backend,
                                hw::HWModuleOp module,
                                const ExactSynthesisRunOptions &options) {
  auto target = parseTruthTableTargetFromModule(module);
  if (failed(target))
    return failure();
  return exactSynthesizeTruthTableForBackend(backend, *target, options,
                                             module.getOperation());
}

static void rewriteModuleWithExactNetwork(const ExactSynthesisBackend &backend,
                                          hw::HWModuleOp module,
                                          const ExactNetwork &network) {
  auto *bodyBlock = module.getBodyBlock();
  auto output = cast<hw::OutputOp>(bodyBlock->getTerminator());

  SmallVector<Operation *> toErase;
  for (Operation &op : bodyBlock->without_terminator())
    toErase.push_back(&op);

  OpBuilder builder(module);
  builder.setInsertionPointToStart(bodyBlock);
  LLVM_DEBUG(llvm::dbgs() << "Rewriting module " << module.getModuleName()
                          << " with " << backend.getFamilyName()
                          << " network steps=" << network.steps.size()
                          << " output-inverted=" << network.output.inverted
                          << "\n");
  Value result = backend.materializeNetwork(builder, module.getLoc(),
                                            bodyBlock->getArguments(), network);
  output->setOperands({result});
  for (Operation *op : llvm::reverse(toErase))
    op->erase();
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
parseExactSynthesisCutRewriteEntry(hw::HWModuleOp module,
                                   const CutRewriteModuleMetadata &metadata) {
  auto inverterKind =
      parseCutRewriteInverterKind(module, metadata.inverterKind);
  if (failed(inverterKind))
    return failure();

  auto entry = std::make_unique<LoadedExactNetworkEntry>(*inverterKind, module);
  entry->moduleName = module.getModuleName().str();
  entry->npnClass = getIdentityNPNClass(metadata.npnClass.truthTable);
  entry->area = metadata.area;
  entry->delay = metadata.delay;
  std::unique_ptr<LoadedCutRewriteEntry> result = std::move(entry);
  return result;
}

class MIGExactSynthesisBackend final : public ExactSynthesisBackend {
public:
  StringRef getKind() const override { return "mig-exact"; }
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

  bool isTrivialCandidate(const ExactCandidate &) const override {
    return false;
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

  bool isTrivialCandidate(const ExactCandidate &candidate) const override {
    return llvm::any_of(candidate.fanins, [](const ExactSignalRef &fanin) {
      return fanin.source == 0;
    });
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
};

static const ExactSynthesisBackend *getExactSynthesisBackend(StringRef kind) {
  static const MIGExactSynthesisBackend migBackend;
  static const AIGExactSynthesisBackend aigBackend;

  std::string normalized = normalizeExactSynthesisKind(kind);
  if (normalized == migBackend.getKind() || normalized == "mig")
    return &migBackend;
  if (normalized == aigBackend.getKind() || normalized == "aig")
    return &aigBackend;
  return nullptr;
}

} // namespace

FailureOr<std::unique_ptr<LoadedCutRewriteEntry>>
circt::synth::parseCutRewriteEntry(hw::HWModuleOp module,
                                   const CutRewriteModuleMetadata &metadata) {
  return parseExactSynthesisCutRewriteEntry(module, metadata);
}

LogicalResult circt::synth::exactSynthesizeTruthTable(
    hw::HWModuleOp module, StringRef kind,
    const ExactSynthesisRunOptions &options) {
  auto *backend = getExactSynthesisBackend(kind);
  if (!backend) {
    module.emitError() << "unsupported exact synthesis kind '" << kind << "'";
    return failure();
  }
  if (options.conflictLimit < -1) {
    module.emitError()
        << "'conflict-limit' must be greater than or equal to -1";
    return failure();
  }
  auto cadicalConfig = parseCadicalConfig(module, options.cadicalConfig);
  if (failed(cadicalConfig))
    return failure();
  CadicalSATSolverOptions cadicalOptions;
  cadicalOptions.config = *cadicalConfig;
  if (!createIncrementalSATSolver(options.satSolver, cadicalOptions)) {
    module.emitError() << "Exact " << backend->getFamilyName()
                       << " synthesis requires a SAT solver backend '"
                       << options.satSolver << "'";
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Running exact synthesis on module "
                          << module.getModuleName() << ": kind=" << kind
                          << " solver=" << options.satSolver
                          << " cadical-config=" << options.cadicalConfig
                          << " conflict-limit=" << options.conflictLimit
                          << "\n");
  auto network = exactSynthesizeModuleForBackend(*backend, module, options);
  if (failed(network))
    return failure();
  if (*network) {
    rewriteModuleWithExactNetwork(*backend, module, **network);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "Leaving module " << module.getModuleName()
               << " as comb.truth_table after unsuccessful exact synthesis\n");
  }
  return success();
}

LogicalResult circt::synth::emitExactSynthesisDatabase(
    mlir::ModuleOp module, StringRef kind,
    const ExactSynthesisDatabaseGenOptions &options) {
  auto *backend = getExactSynthesisBackend(kind);
  if (!backend) {
    module.emitError() << "unsupported database kind '" << kind << "'";
    return failure();
  }

  if (options.maxInputs > backend->getMaxSupportedInputs()) {
    module.emitError() << backend->getFamilyName()
                       << " exact database generation supports at most "
                       << backend->getMaxSupportedInputs() << " inputs";
    return failure();
  }

  PassManager pm(module.getContext());
  GenPredefinedOptions predefinedOptions;
  predefinedOptions.kind = "npn";
  predefinedOptions.maxInputs = options.maxInputs;
  pm.addPass(createGenPredefined(std::move(predefinedOptions)));
  if (failed(pm.run(module)))
    return failure();

  ExactSynthesisRunOptions exactOptions;
  exactOptions.satSolver = options.satSolver;
  exactOptions.cadicalConfig = options.cadicalConfig;
  exactOptions.conflictLimit = options.conflictLimit;
  for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "Exact database entry synthesis: module="
                            << hwModule.getModuleName()
                            << " backend=" << backend->getFamilyName() << "\n");
    if (failed(exactSynthesizeTruthTable(hwModule, kind, exactOptions)))
      return failure();
  }
  return success();
}

namespace {

struct ExactSynthesisPass
    : public circt::synth::impl::ExactSynthesisBase<ExactSynthesisPass> {
  using circt::synth::impl::ExactSynthesisBase<
      ExactSynthesisPass>::ExactSynthesisBase;

  void runOnOperation() override {
    ExactSynthesisRunOptions options;
    options.satSolver = satSolver;
    options.cadicalConfig = cadicalConfig;
    options.conflictLimit = conflictLimit;
    if (failed(exactSynthesizeTruthTable(getOperation(), kind, options)))
      signalPassFailure();
  }
};

} // namespace
