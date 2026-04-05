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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <array>
#include <memory>
#include <mutex>
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
static constexpr unsigned kMaxMIGExactSearchArea = 32;
static constexpr unsigned kMaxDIGExactSearchArea = 16;

enum class CutRewriteInverterKind { mig, aig, dig };
enum class ExactSynthesisObjective { area, depthSize };

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

static FailureOr<ExactSynthesisObjective>
parseExactSynthesisObjective(Operation *op, StringRef objective) {
  if (objective == "area")
    return ExactSynthesisObjective::area;
  if (objective == "depth-size")
    return ExactSynthesisObjective::depthSize;
  op->emitError() << "unsupported exact synthesis objective '" << objective
                  << "'";
  return failure();
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
  if (kind == "dig")
    return CutRewriteInverterKind::dig;
  op->emitError("unsupported cut-rewrite inverter kind '") << kind << "'";
  return failure();
}

struct ExactSignalRef {
  // `source == 0` denotes the constant-false source. All other sources are
  // numbered as 1-based primary inputs followed by synthesized steps.
  unsigned source = 0;
  bool inverted = false;
};

enum class ExactNodeKind {
  Maj3,
  Xor2,
  Dot3,
};

struct ExactNetworkStep {
  // One backend-specific primitive node in the synthesized network.
  ExactNodeKind kind = ExactNodeKind::Maj3;
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

static unsigned getCandidateInversionMask(const ExactCandidate &candidate) {
  unsigned key = 0;
  for (auto [index, fanin] : llvm::enumerate(candidate.fanins))
    if (fanin.inverted)
      key |= 1u << index;
  return key;
}

static bool isCandidateLess(const ExactCandidate &lhs,
                            const ExactCandidate &rhs) {
  if (lhs.fanins.size() != rhs.fanins.size())
    return lhs.fanins.size() < rhs.fanins.size();
  for (size_t i = 0, e = lhs.fanins.size(); i != e; ++i) {
    if (lhs.fanins[i].source < rhs.fanins[i].source)
      return true;
    if (lhs.fanins[i].source > rhs.fanins[i].source)
      return false;
  }
  if (lhs.kind != rhs.kind)
    return static_cast<unsigned>(lhs.kind) < static_cast<unsigned>(rhs.kind);
  return getCandidateInversionMask(lhs) < getCandidateInversionMask(rhs);
}

static unsigned computeDepthAreaUpperBound(unsigned arity, unsigned depth,
                                           unsigned maxArea) {
  uint64_t total = 0;
  uint64_t nodesAtLevel = 1;
  for (unsigned currentDepth = 0; currentDepth != depth; ++currentDepth) {
    total += nodesAtLevel;
    if (total >= maxArea)
      return maxArea;
    if (nodesAtLevel > maxArea / arity)
      nodesAtLevel = maxArea;
    else
      nodesAtLevel *= arity;
  }
  return static_cast<unsigned>(std::min<uint64_t>(total, maxArea));
}

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
  // Some families can absorb this into the final node instead of materializing
  // a separate inverter.
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
                         const llvm::APInt &target, unsigned numSteps)
      : backend(backend), solver(solver), numInputs(numInputs), target(target),
        numSteps(numSteps), numMinterms(1u << numInputs),
        totalSources(1 + numInputs + numSteps) {}

  SolveResult solve() {
    buildEncoding();
    LLVM_DEBUG(llvm::dbgs()
               << "Exact SAT solve: family=" << backend.getFamilyName()
               << " inputs=" << numInputs << " steps=" << numSteps
               << " minterms=" << numMinterms << " vars=" << nextVar << "\n");
    auto result = solver.solve();
    LLVM_DEBUG(llvm::dbgs()
               << "Exact SAT result: family=" << backend.getFamilyName()
               << " inputs=" << numInputs << " steps=" << numSteps
               << " result=" << static_cast<int>(result) << "\n");
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

  void addExactlyOne(ArrayRef<int> vars) {
    solver.addExactlyOne(vars, [&] { return newVar(); });
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
      llvm::erase_if(stepCandidates[step],
                     [&](const ExactCandidate &candidate) {
                       return backend.isTrivialCandidate(candidate);
                     });
      auto &selectionVars = stepSelectionVars[step];
      selectionVars.reserve(stepCandidates[step].size());
      for (size_t i = 0, e = stepCandidates[step].size(); i != e; ++i)
        selectionVars.push_back(newVar());
      addExactlyOne(selectionVars);
      LLVM_DEBUG(llvm::dbgs()
                 << "  step " << step
                 << ": availableSources=" << availableSources
                 << " candidates=" << stepCandidates[step].size() << "\n");
    }

    addAdjacentStepSymmetryBreakingConstraints();
    addCandidateSemanticsConstraints();
    addUseAllStepsConstraints();

    unsigned rootSource = totalSources - 1;
    // The last synthesized step is the network root. Constraining its value on
    // every minterm to match `target` forces the entire selected network to
    // implement the requested truth table.
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({target[minterm]
                            ? getSourceValueVar(rootSource, minterm)
                            : -getSourceValueVar(rootSource, minterm)});
  }

  void addAdjacentStepSymmetryBreakingConstraints() {
    for (unsigned step = 0; step + 1 < numSteps; ++step)
      addAdjacentStepOrdering(step, step + 1);
  }

  void addAdjacentStepOrdering(unsigned prevStep, unsigned nextStep) {
    const auto &prevCandidates = stepCandidates[prevStep];
    const auto &nextCandidates = stepCandidates[nextStep];
    const auto &prevSelectionVars = stepSelectionVars[prevStep];
    const auto &nextSelectionVars = stepSelectionVars[nextStep];

    for (auto [prevIndex, prevCandidate] : llvm::enumerate(prevCandidates)) {
      SmallVector<int, 64> allowedNextSelections;
      for (auto [nextIndex, nextCandidate] : llvm::enumerate(nextCandidates))
        if (!isCandidateLess(nextCandidate, prevCandidate))
          allowedNextSelections.push_back(nextSelectionVars[nextIndex]);

      assert(!allowedNextSelections.empty() &&
             "colex symmetry break must leave some valid next-step choices");
      SmallVector<int, 65> clause;
      clause.reserve(allowedNextSelections.size() + 1);
      clause.push_back(-prevSelectionVars[prevIndex]);
      clause.append(allowedNextSelections.begin(), allowedNextSelections.end());
      solver.addClause(clause);
    }

  }

  void addCandidateSemanticsConstraints() {
    for (unsigned step = 0; step != numSteps; ++step) {
      unsigned outSource = 1 + numInputs + step;
      const auto &selectionVars = stepSelectionVars[step];
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
  }

  void addUseAllStepsConstraints() {
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
      solver.addClause(users);
    }
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
        network.steps.push_back(candidates[i]);
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

class GenericDepthExactSATProblem {
public:
  struct SolveResult {
    IncrementalSATSolver::Result result = IncrementalSATSolver::kUNKNOWN;
    std::optional<ExactNetwork> network;
  };

  GenericDepthExactSATProblem(const ExactSynthesisBackend &backend,
                              IncrementalSATSolver &solver, unsigned numInputs,
                              const llvm::APInt &target, unsigned numSteps,
                              unsigned targetDepth)
      : backend(backend), solver(solver), numInputs(numInputs), target(target),
        numSteps(numSteps), targetDepth(targetDepth),
        numMinterms(1u << numInputs), totalSources(1 + numInputs + numSteps) {}

  SolveResult solve() {
    buildEncoding();
    LLVM_DEBUG(llvm::dbgs()
               << "Exact SAT solve: family=" << backend.getFamilyName()
               << " inputs=" << numInputs << " steps=" << numSteps
               << " depth=" << targetDepth << " minterms=" << numMinterms
               << " vars=" << nextVar << "\n");
    auto result = solver.solve();
    LLVM_DEBUG(llvm::dbgs()
               << "Exact SAT result: family=" << backend.getFamilyName()
               << " inputs=" << numInputs << " steps=" << numSteps
               << " depth=" << targetDepth
               << " result=" << static_cast<int>(result) << "\n");
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

  void addExactlyOne(ArrayRef<int> vars) {
    solver.addExactlyOne(vars, [&] { return newVar(); });
  }

  int getSourceValueVar(unsigned source, unsigned minterm) const {
    return sourceValueVars[source][minterm];
  }

  int getSourceLiteral(unsigned source, unsigned minterm, bool inverted) const {
    int lit = getSourceValueVar(source, minterm);
    return inverted ? -lit : lit;
  }

  unsigned getStepIndexForSource(unsigned source) const {
    assert(source > numInputs && "source must reference a synthesized step");
    return source - (1 + numInputs);
  }

  int getStepDepthVar(unsigned step, unsigned depth) const {
    assert(depth >= 1 && depth <= targetDepth && "invalid depth");
    return stepDepthVars[step][depth - 1];
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
    stepDepthVars.resize(numSteps);
    for (unsigned step = 0; step != numSteps; ++step) {
      unsigned availableSources = 1 + numInputs + step;
      backend.enumerateCandidates(availableSources, stepCandidates[step]);
      llvm::erase_if(stepCandidates[step],
                     [&](const ExactCandidate &candidate) {
                       return backend.isTrivialCandidate(candidate);
                     });

      auto &selectionVars = stepSelectionVars[step];
      selectionVars.reserve(stepCandidates[step].size());
      for (size_t i = 0, e = stepCandidates[step].size(); i != e; ++i)
        selectionVars.push_back(newVar());
      addExactlyOne(selectionVars);

      auto &depthVars = stepDepthVars[step];
      depthVars.reserve(targetDepth);
      for (unsigned depth = 1; depth <= targetDepth; ++depth)
        depthVars.push_back(newVar());
      addExactlyOne(depthVars);

      LLVM_DEBUG(llvm::dbgs() << "  step " << step
                              << ": availableSources=" << availableSources
                              << " candidates=" << stepCandidates[step].size()
                              << " depths=" << targetDepth << "\n");
    }

    addCandidateSemanticsConstraints();
    addDepthConstraints();
    addUseAllStepsConstraints();
    addAdjacentDepthOrderingConstraints();
    addAdjacentSameLayerSymmetryBreakingConstraints();
    addRootConstraints();
  }

  void addCandidateSemanticsConstraints() {
    for (unsigned step = 0; step != numSteps; ++step) {
      unsigned outSource = 1 + numInputs + step;
      const auto &selectionVars = stepSelectionVars[step];
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
  }

  void addDepthConstraints() {
    for (unsigned step = 0; step != numSteps; ++step) {
      const auto &selectionVars = stepSelectionVars[step];
      for (auto [candidateIndex, candidate] :
           llvm::enumerate(stepCandidates[step])) {
        for (unsigned depth = 1; depth <= targetDepth; ++depth) {
          SmallVector<int, 4> supportingFanins;
          for (const auto &fanin : candidate.fanins) {
            if (fanin.source <= numInputs)
              continue;
            unsigned producerStep = getStepIndexForSource(fanin.source);
            for (unsigned producerDepth = depth; producerDepth <= targetDepth;
                 ++producerDepth)
              solver.addClause({-selectionVars[candidateIndex],
                                -getStepDepthVar(step, depth),
                                -getStepDepthVar(producerStep, producerDepth)});
            if (depth > 1)
              supportingFanins.push_back(
                  getStepDepthVar(producerStep, depth - 1));
          }

          if (depth == 1)
            continue;

          SmallVector<int, 8> clause;
          clause.reserve(supportingFanins.size() + 2);
          clause.push_back(-selectionVars[candidateIndex]);
          clause.push_back(-getStepDepthVar(step, depth));
          clause.append(supportingFanins.begin(), supportingFanins.end());
          solver.addClause(clause);
        }
      }
    }
  }

  void addUseAllStepsConstraints() {
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
      solver.addClause(users);
    }
  }

  void addAdjacentDepthOrderingConstraints() {
    for (unsigned step = 0; step + 1 < numSteps; ++step)
      for (unsigned prevDepth = 1; prevDepth <= targetDepth; ++prevDepth)
        for (unsigned nextDepth = 1; nextDepth < prevDepth; ++nextDepth)
          solver.addClause({-getStepDepthVar(step, prevDepth),
                            -getStepDepthVar(step + 1, nextDepth)});
  }

  void addAdjacentSameLayerSymmetryBreakingConstraints() {
    for (unsigned step = 0; step + 1 < numSteps; ++step)
      addAdjacentSameLayerOrdering(step, step + 1);
  }

  void addAdjacentSameLayerOrdering(unsigned prevStep, unsigned nextStep) {
    const auto &prevCandidates = stepCandidates[prevStep];
    const auto &nextCandidates = stepCandidates[nextStep];
    const auto &prevSelectionVars = stepSelectionVars[prevStep];
    const auto &nextSelectionVars = stepSelectionVars[nextStep];

    for (auto [prevIndex, prevCandidate] : llvm::enumerate(prevCandidates))
      for (auto [nextIndex, nextCandidate] : llvm::enumerate(nextCandidates)) {
        if (!isCandidateLess(nextCandidate, prevCandidate))
          continue;

        for (unsigned depth = 1; depth <= targetDepth; ++depth)
          solver.addClause({-prevSelectionVars[prevIndex],
                            -nextSelectionVars[nextIndex],
                            -getStepDepthVar(prevStep, depth),
                            -getStepDepthVar(nextStep, depth)});
      }
  }

  void addRootConstraints() {
    solver.addClause({getStepDepthVar(numSteps - 1, targetDepth)});

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
        network.steps.push_back(candidates[i]);
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
  unsigned targetDepth;
  unsigned numMinterms;
  unsigned totalSources;
  int nextVar = 0;
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  SmallVector<SmallVector<ExactCandidate, 64>, 8> stepCandidates;
  SmallVector<SmallVector<int, 64>, 8> stepSelectionVars;
  SmallVector<SmallVector<int, 8>, 8> stepDepthVars;
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
                          ExactSynthesisObjective objective,
                          StringRef satBackend,
                          CadicalSATSolverOptions cadicalOptions,
                          int conflictLimit)
      : backend(backend), objective(objective), satBackend(satBackend),
        cadicalOptions(cadicalOptions), conflictLimit(conflictLimit) {}

  QueryResult synthesize(const BinaryTruthTable &tt) const {
    auto [normalizedTT, invertOutput] = backend.normalize(tt);
    LLVM_DEBUG(llvm::dbgs()
               << "Exact synthesis query: family=" << backend.getFamilyName()
               << " objective="
               << (objective == ExactSynthesisObjective::area ? "area"
                                                              : "depth-size")
               << " original-tt=" << formatTruthTable(tt)
               << " normalized-tt=" << formatTruthTable(normalizedTT)
               << " invert-output=" << invertOutput << "\n");

    auto result = objective == ExactSynthesisObjective::area
                      ? synthesizeForMinimumArea(normalizedTT)
                      : synthesizeForMinimumDepthThenArea(normalizedTT);
    if (result.status != QueryStatus::Solved || !result.network)
      return result;

    ExactNetwork network = *result.network;
    if (invertOutput)
      network = backend.applyOutputNegation(std::move(network));
    return {.status = QueryStatus::Solved,
            .network = std::optional<ExactNetwork>(std::move(network))};
  }

private:
  QueryResult synthesizeNormalizedForArea(unsigned numInputs,
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

    auto solver = createIncrementalSATSolver(satBackend, cadicalOptions);
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

  QueryResult synthesizeNormalizedForDepth(unsigned numInputs,
                                           const llvm::APInt &target,
                                           unsigned area,
                                           unsigned depth) const {
    auto solver = createIncrementalSATSolver(satBackend, cadicalOptions);
    if (!solver)
      return {.status = QueryStatus::Error};
    solver->setConflictLimit(conflictLimit);

    GenericDepthExactSATProblem problem(backend, *solver, numInputs, target,
                                        area, depth);
    auto solveResult = problem.solve();
    if (solveResult.result == IncrementalSATSolver::kUNKNOWN)
      return {.status = QueryStatus::ConflictLimitReached};
    return {.status = solveResult.network ? QueryStatus::Solved
                                          : QueryStatus::NoSolution,
            .network = std::move(solveResult.network)};
  }

  QueryResult synthesizeForMinimumArea(const BinaryTruthTable &tt) const {
    for (unsigned area = 0; area <= backend.getMaxSearchArea(); ++area) {
      auto result = synthesizeNormalizedForArea(tt.numInputs, tt.table, area);
      if (result.status != QueryStatus::NoSolution)
        return result;
      LLVM_DEBUG(llvm::dbgs()
                 << "Exact synthesis no solution at area: family="
                 << backend.getFamilyName() << " tt=" << formatTruthTable(tt)
                 << " area=" << area << "\n");
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Exact synthesis exhausted search area: family="
               << backend.getFamilyName() << " tt=" << formatTruthTable(tt)
               << " max-area=" << backend.getMaxSearchArea() << "\n");
    return {.status = QueryStatus::NoSolution};
  }

  QueryResult
  synthesizeForMinimumDepthThenArea(const BinaryTruthTable &tt) const {
    auto direct = synthesizeNormalizedForArea(tt.numInputs, tt.table, 0);
    if (direct.status != QueryStatus::NoSolution)
      return direct;

    unsigned maxArea = backend.getMaxSearchArea();
    for (unsigned depth = 1; depth <= maxArea; ++depth) {
      unsigned areaUpperBound =
          computeDepthAreaUpperBound(backend.getArity(), depth, maxArea);
      for (unsigned area = 1; area <= areaUpperBound; ++area) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Exact synthesis depth query: family="
                   << backend.getFamilyName() << " tt=" << formatTruthTable(tt)
                   << " depth=" << depth << " area=" << area << "\n");
        auto result =
            synthesizeNormalizedForDepth(tt.numInputs, tt.table, area, depth);
        if (result.status != QueryStatus::NoSolution)
          return result;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Exact synthesis no solution at depth: family="
                 << backend.getFamilyName() << " tt=" << formatTruthTable(tt)
                 << " depth=" << depth << "\n");
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Exact synthesis exhausted search depth: family="
               << backend.getFamilyName() << " tt=" << formatTruthTable(tt)
               << " max-depth=" << backend.getMaxSearchArea() << "\n");
    return {.status = QueryStatus::NoSolution};
  }

  const ExactSynthesisBackend &backend;
  ExactSynthesisObjective objective;
  std::string satBackend;
  CadicalSATSolverOptions cadicalOptions;
  int conflictLimit;
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
  auto objective = parseExactSynthesisObjective(op, options.objective);
  if (failed(objective))
    return failure();
  auto cadicalConfig = parseCadicalConfig(op, options.cadicalConfig);
  if (failed(cadicalConfig))
    return failure();
  CadicalSATSolverOptions cadicalOptions;
  cadicalOptions.config = *cadicalConfig;
  GenericExactSynthesizer synthesizer(backend, *objective, options.satSolver,
                                      cadicalOptions, options.conflictLimit);
  SmallString<32> ttString;
  target.table.toStringUnsigned(ttString, 16);
  auto result = synthesizer.synthesize(target);
  if (result.status ==
      GenericExactSynthesizer::QueryStatus::ConflictLimitReached) {
    LLVM_DEBUG(llvm::dbgs()
               << "Exact synthesis stopped at conflict limit: family="
               << backend.getFamilyName() << " tt=" << ttString
               << " objective=" << options.objective << "\n");
    return std::optional<ExactNetwork>();
  }
  if (result.status == GenericExactSynthesizer::QueryStatus::Error) {
    LLVM_DEBUG(llvm::dbgs() << "Exact synthesis error: family="
                            << backend.getFamilyName() << " tt=" << ttString
                            << " objective=" << options.objective << "\n");
    op->emitError() << "failed to synthesize exact " << backend.getFamilyName()
                    << " for truth table " << ttString;
    return failure();
  }
  if (result.network)
    return std::optional<ExactNetwork>(std::move(*result.network));
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

  auto materializeInverter = [&](Value value) -> Value {
    std::array<Value, 1> operands = {value};
    std::array<bool, 1> inverted = {true};
    return synth::mig::MajorityInverterOp::create(builder, loc, operands,
                                                  inverted);
  };

  auto materializeSignal = [&](ExactSignalRef signal,
                               ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(signal.inverted);
    Value value = getRawSignal(signal, stepValues);
    if (!signal.inverted)
      return value;
    return materializeInverter(value);
  };

  auto materializeOutputSignal = [&](ExactSignalRef signal,
                                     ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(signal.inverted);

    Value value = signal.inverted ? materializeSignal(signal, stepValues)
                                  : getRawSignal(signal, stepValues);
    if (!signal.inverted) {
      if (value.getDefiningOp())
        return value;
      return hw::WireOp::create(builder, loc, value);
    }
    return value;
  };

  SmallVector<Value, 4> stepValues;
  stepValues.reserve(network.steps.size());
  for (const auto &step : network.steps) {
    switch (step.kind) {
    case ExactNodeKind::Maj3: {
      std::array<Value, 3> operands = {
          getRawSignal(step.fanins[0], stepValues),
          getRawSignal(step.fanins[1], stepValues),
          getRawSignal(step.fanins[2], stepValues)};
      std::array<bool, 3> inverted = {step.fanins[0].inverted,
                                      step.fanins[1].inverted,
                                      step.fanins[2].inverted};
      stepValues.push_back(synth::mig::MajorityInverterOp::create(
          builder, loc, operands, inverted));
      break;
    }
    case ExactNodeKind::Xor2: {
      Value lhs = materializeSignal(step.fanins[0], stepValues);
      Value rhs = materializeSignal(step.fanins[1], stepValues);
      stepValues.push_back(comb::XorOp::create(builder, loc, lhs, rhs));
      break;
    }
    case ExactNodeKind::Dot3:
      llvm_unreachable("DIG node in MIG network materialization");
    }
  }
  return materializeOutputSignal(network.output, stepValues);
}

static Value materializeExactDIGNetwork(OpBuilder &builder, Location loc,
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

  auto materializeInverter = [&](Value value) -> Value {
    std::array<Value, 1> operands = {value};
    std::array<bool, 1> inverted = {true};
    return synth::dig::DotInverterOp::create(builder, loc, operands, inverted);
  };

  auto materializeSignal = [&](ExactSignalRef signal,
                               ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(signal.inverted);
    Value value = getRawSignal(signal, stepValues);
    if (!signal.inverted)
      return value;
    return materializeInverter(value);
  };

  auto materializeOutputSignal = [&](ExactSignalRef signal,
                                     ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(signal.inverted);

    Value value = signal.inverted ? materializeSignal(signal, stepValues)
                                  : getRawSignal(signal, stepValues);
    if (!signal.inverted) {
      if (value.getDefiningOp())
        return value;
      return hw::WireOp::create(builder, loc, value);
    }
    return value;
  };

  SmallVector<Value, 4> stepValues;
  stepValues.reserve(network.steps.size());
  for (const auto &step : network.steps) {
    switch (step.kind) {
    case ExactNodeKind::Dot3: {
      std::array<Value, 3> operands = {
          getRawSignal(step.fanins[0], stepValues),
          getRawSignal(step.fanins[1], stepValues),
          getRawSignal(step.fanins[2], stepValues)};
      std::array<bool, 3> inverted = {step.fanins[0].inverted,
                                      step.fanins[1].inverted,
                                      step.fanins[2].inverted};
      stepValues.push_back(
          synth::dig::DotInverterOp::create(builder, loc, operands, inverted));
      break;
    }
    case ExactNodeKind::Maj3:
    case ExactNodeKind::Xor2:
      llvm_unreachable("non-DIG node in DIG network materialization");
    }
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

    const auto &cutNPN = cut.getNPNClass(enumerator.getOptions());
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
    case CutRewriteInverterKind::dig:
      return synth::dig::DotInverterOp::create(builder, loc, operands,
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

struct MIGBackendConfig {
  StringLiteral kind;
  StringLiteral familyName;
  bool enableXor;
};

class ConfigurableMIGExactSynthesisBackend final : public ExactSynthesisBackend {
public:
  explicit ConfigurableMIGExactSynthesisBackend(MIGBackendConfig config)
      : config(config) {}

  StringRef getKind() const override { return config.kind; }
  unsigned getMaxSupportedInputs() const override {
    return kMaxExactSynthesisInputs;
  }
  StringRef getFamilyName() const override { return config.familyName; }
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
    auto &lastStep = network.steps.back();
    switch (lastStep.kind) {
    case ExactNodeKind::Maj3:
      for (auto &fanin : lastStep.fanins)
        fanin.inverted = !fanin.inverted;
      break;
    case ExactNodeKind::Xor2:
      lastStep.fanins[0].inverted = !lastStep.fanins[0].inverted;
      break;
    case ExactNodeKind::Dot3:
      llvm_unreachable("DIG node in MIG output-negation handling");
    }
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
            candidate.kind = ExactNodeKind::Maj3;
            candidate.fanins.push_back({a, static_cast<bool>(invMask & 1)});
            candidate.fanins.push_back({b, static_cast<bool>(invMask & 2)});
            candidate.fanins.push_back({c, static_cast<bool>(invMask & 4)});
            candidates.push_back(std::move(candidate));
          }

    if (!config.enableXor)
      return;

    for (unsigned a = 1; a + 1 < availableSources; ++a)
      for (unsigned b = a + 1; b < availableSources; ++b) {
        ExactCandidate candidate;
        candidate.kind = ExactNodeKind::Xor2;
        candidate.fanins.push_back({a, false});
        candidate.fanins.push_back({b, false});
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

    switch (candidate.kind) {
    case ExactNodeKind::Maj3: {
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
      break;
    }
    case ExactNodeKind::Xor2: {
      int aLit = getSourceLiteral(candidate.fanins[0].source, minterm,
                                  candidate.fanins[0].inverted);
      int bLit = getSourceLiteral(candidate.fanins[1].source, minterm,
                                  candidate.fanins[1].inverted);
      addConditionedClause({aLit, bLit, -outLit});
      addConditionedClause({-aLit, -bLit, -outLit});
      addConditionedClause({aLit, -bLit, outLit});
      addConditionedClause({-aLit, bLit, outLit});
      break;
    }
    case ExactNodeKind::Dot3:
      llvm_unreachable("DIG node in MIG semantics encoding");
    }
  }

  Value materializeNetwork(OpBuilder &builder, Location loc, ValueRange inputs,
                           const ExactNetwork &network) const override {
    return materializeExactMIGNetwork(builder, loc, inputs, network);
  }

private:
  MIGBackendConfig config;
};

class DIGExactSynthesisBackend final : public ExactSynthesisBackend {
public:
  StringRef getKind() const override { return "dig"; }
  unsigned getMaxSupportedInputs() const override {
    return kMaxExactSynthesisInputs;
  }
  StringRef getFamilyName() const override { return "DIG"; }
  unsigned getArity() const override { return 3; }
  unsigned getMaxSearchArea() const override { return kMaxDIGExactSearchArea; }

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
    network.output.inverted = !network.output.inverted;
    return network;
  }

  void enumerateCandidates(
      unsigned availableSources,
      SmallVectorImpl<ExactCandidate> &candidates) const override {
    candidates.clear();
    for (unsigned x = 1; x < availableSources; ++x)
      for (unsigned y = 1; y < availableSources; ++y)
        for (unsigned z = 1; z < availableSources; ++z)
          for (unsigned invMask = 0; invMask != 8; ++invMask) {
            ExactCandidate candidate;
            candidate.kind = ExactNodeKind::Dot3;
            candidate.fanins.push_back({x, static_cast<bool>(invMask & 1)});
            candidate.fanins.push_back({y, static_cast<bool>(invMask & 2)});
            candidate.fanins.push_back({z, static_cast<bool>(invMask & 4)});
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
    int xLit = getSourceLiteral(candidate.fanins[0].source, minterm,
                                candidate.fanins[0].inverted);
    int yLit = getSourceLiteral(candidate.fanins[1].source, minterm,
                                candidate.fanins[1].inverted);
    int zLit = getSourceLiteral(candidate.fanins[2].source, minterm,
                                candidate.fanins[2].inverted);

    for (unsigned assignment = 0; assignment != 8; ++assignment) {
      bool x = assignment & 1;
      bool y = assignment & 2;
      bool z = assignment & 4;
      bool value = x ^ (z || (x && y));

      SmallVector<int, 4> clause = {-selector, x ? -xLit : xLit,
                                    y ? -yLit : yLit, z ? -zLit : zLit};
      clause.push_back(value ? outLit : -outLit);
      solver.addClause(clause);
    }
  }

  Value materializeNetwork(OpBuilder &builder, Location loc, ValueRange inputs,
                           const ExactNetwork &network) const override {
    return materializeExactDIGNetwork(builder, loc, inputs, network);
  }
};

static const ExactSynthesisBackend *getExactSynthesisBackend(StringRef kind) {
  static const ConfigurableMIGExactSynthesisBackend migBackend(
      {StringLiteral("mig"), StringLiteral("MIG"), false});
  static const ConfigurableMIGExactSynthesisBackend migXorBackend(
      {StringLiteral("mig-xor"), StringLiteral("MIG-XOR"), true});
  static const DIGExactSynthesisBackend digBackend;

  std::string normalized = normalizeExactSynthesisKind(kind);
  if (normalized == migBackend.getKind())
    return &migBackend;
  if (normalized == migXorBackend.getKind())
    return &migXorBackend;
  if (normalized == digBackend.getKind())
    return &digBackend;
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
  auto objective = parseExactSynthesisObjective(module, options.objective);
  if (failed(objective))
    return failure();
  if (backend->getKind() == "dig" &&
      *objective != ExactSynthesisObjective::area) {
    module.emitError()
        << "exact synthesis backend 'dig' only supports objective 'area'";
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
                          << " objective=" << options.objective
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

namespace {

static LogicalResult exactSynthesizeTruthTables(ModuleOp module, StringRef kind,
                                                const ExactSynthesisRunOptions &options) {
  SmallVector<hw::HWModuleOp> modules;
  for (auto hwModule : module.getOps<hw::HWModuleOp>())
    modules.push_back(hwModule);

  std::atomic<unsigned> completedCount = 0;
  std::mutex progressMutex;
  return mlir::failableParallelForEach(
      module.getContext(), modules, [&](hw::HWModuleOp hwModule) {
        if (failed(exactSynthesizeTruthTable(hwModule, kind, options)))
          return failure();

        if (modules.size() > 1) {
          unsigned completed =
              completedCount.fetch_add(1, std::memory_order_relaxed) + 1;
          std::lock_guard<std::mutex> lock(progressMutex);
          llvm::errs() << "synth-exact-synthesis [" << completed << "/"
                       << modules.size() << "]: " << hwModule.getModuleName()
                       << "\n";
          llvm::errs().flush();
        }
        return success();
      });
}

struct ExactSynthesisPass
    : public circt::synth::impl::ExactSynthesisBase<ExactSynthesisPass> {
  using circt::synth::impl::ExactSynthesisBase<
      ExactSynthesisPass>::ExactSynthesisBase;

  void runOnOperation() override {
    ExactSynthesisRunOptions options;
    options.objective = objective;
    options.satSolver = satSolver;
    options.cadicalConfig = cadicalConfig;
    options.conflictLimit = conflictLimit;
    if (failed(exactSynthesizeTruthTables(getOperation(), kind, options)))
      signalPassFailure();
  }
};

} // namespace
