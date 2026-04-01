//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an offline exact-MIG database generator and a generic
// file-backed cut-rewrite pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/ExactMIGDatabase.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/SATSolver.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

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

static std::unique_ptr<IncrementalSATSolver>
createExactMIGSATSolver(StringRef backend, int conflictLimit = -1) {
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

struct ExactMIGStep {
  std::array<ExactSignalRef, 3> fanins;
};

struct ExactMIGNetwork {
  unsigned numInputs = 0;
  SmallVector<ExactMIGStep, 4> steps;
  ExactSignalRef output;
};

static void dumpExactSignalRef(llvm::raw_ostream &os, ExactSignalRef signal,
                               unsigned numInputs) {
  if (signal.inverted)
    os << "!";
  if (signal.source == 0) {
    os << "0";
    return;
  }
  if (signal.source <= numInputs) {
    os << "i" << (signal.source - 1);
    return;
  }
  os << "n" << (signal.source - numInputs - 1);
}

static void dumpExactMIGNetwork(llvm::raw_ostream &os,
                                const ExactMIGNetwork &network) {
  os << "ExactMIG(numInputs=" << network.numInputs
     << ", area=" << network.steps.size() << ")\n";
  for (auto [index, step] : llvm::enumerate(network.steps)) {
    os << "  n" << index << " = maj(";
    dumpExactSignalRef(os, step.fanins[0], network.numInputs);
    os << ", ";
    dumpExactSignalRef(os, step.fanins[1], network.numInputs);
    os << ", ";
    dumpExactSignalRef(os, step.fanins[2], network.numInputs);
    os << ")\n";
  }
  os << "  out = ";
  dumpExactSignalRef(os, network.output, network.numInputs);
  os << "\n";
}

static ExactMIGNetwork invertExactMIGOutput(ExactMIGNetwork network) {
  if (network.steps.empty()) {
    network.output.inverted = !network.output.inverted;
    return network;
  }

  for (auto &fanin : network.steps.back().fanins)
    fanin.inverted = !fanin.inverted;
  return network;
}

struct ExactMIGMetrics {
  unsigned area = 0;
  SmallVector<DelayType, 6> delays;

  DelayType getMaxDepth() const {
    DelayType maxDepth = 0;
    for (DelayType delay : delays)
      maxDepth = std::max(maxDepth, delay);
    return maxDepth;
  }
};

static ExactMIGMetrics computeCurrentCutMetrics(const Cut &cut,
                                                const LogicNetwork &network) {
  static constexpr DelayType kUnreachable =
      std::numeric_limits<DelayType>::min() / 4;

  ExactMIGMetrics metrics;
  metrics.delays.assign(cut.getInputSize(), 0);

  DenseMap<uint32_t, unsigned> boundaryPositions;
  boundaryPositions.reserve(cut.inputs.size());
  for (auto [index, input] : llvm::enumerate(cut.inputs))
    boundaryPositions[input] = index;

  DenseSet<uint32_t> internalNodes;
  std::function<void(uint32_t)> collect = [&](uint32_t index) {
    if (boundaryPositions.contains(index))
      return;
    const auto &gate = network.getGate(index);
    if (gate.isAlwaysCutInput())
      return;
    if (!internalNodes.insert(index).second)
      return;
    for (unsigned operand = 0, e = gate.getNumFanins(); operand != e; ++operand)
      collect(gate.edges[operand].getIndex());
  };
  collect(cut.getRootIndex());
  metrics.area = internalNodes.size();

  DenseMap<uint32_t, SmallVector<DelayType, 6>> delayCache;
  std::function<const SmallVector<DelayType, 6> &(uint32_t)> computeDelays =
      [&](uint32_t index) -> const SmallVector<DelayType, 6> & {
    auto it = delayCache.find(index);
    if (it != delayCache.end())
      return it->second;

    SmallVector<DelayType, 6> delays(cut.getInputSize(), kUnreachable);
    if (auto boundaryIt = boundaryPositions.find(index);
        boundaryIt != boundaryPositions.end()) {
      delays[boundaryIt->second] = 0;
      return delayCache.try_emplace(index, std::move(delays)).first->second;
    }

    const auto &gate = network.getGate(index);
    if (gate.isAlwaysCutInput())
      return delayCache.try_emplace(index, std::move(delays)).first->second;

    for (unsigned operand = 0, e = gate.getNumFanins(); operand != e;
         ++operand) {
      const auto &childDelays = computeDelays(gate.edges[operand].getIndex());
      for (unsigned i = 0, inputs = cut.getInputSize(); i != inputs; ++i) {
        if (childDelays[i] == kUnreachable)
          continue;
        delays[i] = std::max(delays[i], childDelays[i] + 1);
      }
    }
    return delayCache.try_emplace(index, std::move(delays)).first->second;
  };

  const auto &rootDelays = computeDelays(cut.getRootIndex());
  for (unsigned i = 0, e = cut.getInputSize(); i != e; ++i)
    if (rootDelays[i] != kUnreachable)
      metrics.delays[i] = rootDelays[i];
  return metrics;
}

static ExactMIGMetrics
computeExactMIGMetrics(const ExactMIGNetwork &network, const Cut &cut,
                       const LogicNetwork &logicNetwork) {
  static constexpr DelayType kUnreachable =
      std::numeric_limits<DelayType>::min() / 4;

  ExactMIGMetrics metrics;
  metrics.delays.assign(cut.getInputSize(), 0);

  if (network.steps.empty()) {
    if (network.output.source == 0)
      return metrics;

    unsigned inputPos = network.output.source - 1;
    if (inputPos >= cut.inputs.size())
      return metrics;

    uint32_t inputIndex = cut.inputs[inputPos];
    const auto &gate = logicNetwork.getGate(inputIndex);
    if (gate.getKind() == LogicNetworkGate::Constant)
      return metrics;

    metrics.area = network.output.inverted ? 1 : 0;
    if (network.output.inverted)
      metrics.delays[inputPos] = 1;
    return metrics;
  }

  SmallVector<SmallVector<DelayType, 6>, 4> stepDelays;
  stepDelays.reserve(network.steps.size());

  auto getSourceDelays =
      [&](ExactSignalRef signal) -> SmallVector<DelayType, 6> {
    SmallVector<DelayType, 6> delays(cut.getInputSize(), kUnreachable);
    if (signal.source == 0)
      return delays;
    if (signal.source <= cut.getInputSize()) {
      delays[signal.source - 1] = 0;
      return delays;
    }
    unsigned stepIndex = signal.source - (cut.getInputSize() + 1);
    assert(stepIndex < stepDelays.size() && "step reference out of order");
    return stepDelays[stepIndex];
  };

  for (const auto &step : network.steps) {
    SmallVector<DelayType, 6> delays(cut.getInputSize(), kUnreachable);
    for (ExactSignalRef signal : step.fanins) {
      auto childDelays = getSourceDelays(signal);
      for (unsigned i = 0, e = cut.getInputSize(); i != e; ++i) {
        if (childDelays[i] == kUnreachable)
          continue;
        delays[i] = std::max(delays[i], childDelays[i] + 1);
      }
    }
    stepDelays.push_back(std::move(delays));
  }

  metrics.area = network.steps.size();
  for (unsigned i = 0, e = cut.getInputSize(); i != e; ++i)
    if (stepDelays.back()[i] != kUnreachable)
      metrics.delays[i] = stepDelays.back()[i];
  return metrics;
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
    if (!encodingBuilt) {
      buildEncoding();
      encodingBuilt = true;
    }
    LLVM_DEBUG(llvm::dbgs() << "ExactMIG SAT solve: numInputs=" << numInputs
                            << " numSteps=" << numSteps << " numMinterms="
                            << numMinterms << " vars=" << nextVar << "\n");
    auto result = solver.solve();
    if (result != IncrementalSATSolver::kSAT) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ExactMIG SAT result: "
                 << (result == IncrementalSATSolver::kUNSAT ? "UNSAT"
                                                            : "UNKNOWN")
                 << " for area " << numSteps << "\n");
      return {.result = result};
    }
    auto network = decodeModel();
    LLVM_DEBUG({
      llvm::dbgs() << "ExactMIG SAT result: SAT for area " << numSteps << "\n";
      dumpExactMIGNetwork(llvm::dbgs(), network);
    });
    return {.result = result, .network = std::move(network)};
  }

  void blockCurrentModel() {
    SmallVector<int, 8> blockingClause;
    blockingClause.reserve(numSteps);
    for (const auto &selectionVars : stepSelectionVars) {
      for (int selectionVar : selectionVars) {
        if (solver.val(selectionVar) != selectionVar)
          continue;
        blockingClause.push_back(-selectionVar);
        break;
      }
    }
    if (!blockingClause.empty())
      solver.addClause(blockingClause);
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

    for (unsigned input = 0; input != numInputs; ++input) {
      for (unsigned minterm = 0; minterm != numMinterms; ++minterm) {
        int lit = getSourceValueVar(1 + input, minterm);
        if ((minterm >> input) & 1)
          solver.addClause({lit});
        else
          solver.addClause({-lit});
      }
    }

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
          int outLit = getSourceValueVar(outSource, minterm);
          int aLit = getSourceLiteral(candidate.fanins[0], minterm,
                                      candidate.inverted[0]);
          int bLit = getSourceLiteral(candidate.fanins[1], minterm,
                                      candidate.inverted[1]);
          int cLit = getSourceLiteral(candidate.fanins[2], minterm,
                                      candidate.inverted[2]);
          addConditionedMajority(selector, outLit, aLit, bLit, cLit);
        }
      }
    }

    unsigned rootSource = totalSources - 1;
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm) {
      int lit = getSourceValueVar(rootSource, minterm);
      if (target[minterm])
        solver.addClause({lit});
      else
        solver.addClause({-lit});
    }
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
  bool encodingBuilt = false;
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

  struct QuerySetResult {
    QueryStatus status = QueryStatus::Error;
    SmallVector<ExactMIGNetwork> networks;
  };

  explicit ExactMIGSynthesizer(StringRef backend, int conflictLimit = -1)
      : backend(backend), conflictLimit(conflictLimit) {}

  QueryResult synthesizeForArea(const BinaryTruthTable &tt, unsigned area) const {
    auto [normalizedTT, invertOutput] = normalize(tt);
    LLVM_DEBUG({
      llvm::dbgs() << "ExactMIG query: inputs=" << tt.numInputs
                   << " area=" << area << " tt=" << tt.table
                   << " normalized=" << normalizedTT.table
                   << " invertOutput=" << invertOutput << "\n";
    });

    auto result = synthesizeNormalized(normalizedTT.numInputs, normalizedTT.table,
                                       area);
    if (result.status != QueryStatus::Solved || !result.network)
      return result;

    ExactMIGNetwork network = *result.network;
    if (invertOutput)
      network = invertExactMIGOutput(std::move(network));
    LLVM_DEBUG({
      llvm::dbgs() << "ExactMIG query: returning "
                   << (invertOutput ? "output-inverted " : "") << "solution\n";
      dumpExactMIGNetwork(llvm::dbgs(), network);
    });
    return {.status = QueryStatus::Solved,
            .network = std::optional<ExactMIGNetwork>(std::move(network))};
  }

  QuerySetResult synthesizeAllForArea(const BinaryTruthTable &tt,
                                      unsigned area) const {
    auto [normalizedTT, invertOutput] = normalize(tt);
    auto result =
        synthesizeAllNormalized(normalizedTT.numInputs, normalizedTT.table, area);
    if (invertOutput) {
      for (auto &network : result.networks)
        network = invertExactMIGOutput(std::move(network));
    }
    return result;
  }

private:
  static std::pair<BinaryTruthTable, bool>
  normalize(const BinaryTruthTable &tt) {
    bool invertOutput = tt.table[0];
    if (!invertOutput)
      return {tt, false};

    BinaryTruthTable normalized = tt;
    normalized.table.flipAllBits();
    LLVM_DEBUG(llvm::dbgs() << "ExactMIG normalize: flipping output polarity "
                            << tt.table << " -> " << normalized.table << "\n");
    return {std::move(normalized), true};
  }

  std::optional<ExactMIGNetwork>
  synthesizeDirect(unsigned numInputs, const llvm::APInt &target) const {
    ExactMIGNetwork network;
    network.numInputs = numInputs;
    if (target.isZero()) {
      network.output = {0, false};
      LLVM_DEBUG(llvm::dbgs()
                 << "ExactMIG direct synthesis: constant 0 function\n");
      return network;
    }
    if (target.isAllOnes()) {
      network.output = {0, true};
      LLVM_DEBUG(llvm::dbgs()
                 << "ExactMIG direct synthesis: constant 1 function\n");
      return network;
    }

    for (unsigned input = 0; input != numInputs; ++input) {
      llvm::APInt mask = circt::createVarMask(numInputs, input, true);
      if (target == mask) {
        network.output = {1 + input, false};
        LLVM_DEBUG(llvm::dbgs()
                   << "ExactMIG direct synthesis: input i" << input << "\n");
        return network;
      }
      llvm::APInt invertedMask = mask;
      invertedMask.flipAllBits();
      if (target == invertedMask) {
        network.output = {1 + input, true};
        LLVM_DEBUG(llvm::dbgs() << "ExactMIG direct synthesis: inverted input i"
                                << input << "\n");
        return network;
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "ExactMIG direct synthesis: no area-0 implementation\n");
    return std::nullopt;
  }

  QueryResult synthesizeNormalized(unsigned numInputs, const llvm::APInt &target,
                                   unsigned area) const {
    if (area == 0) {
      auto direct = synthesizeDirect(numInputs, target);
      return {.status = direct ? QueryStatus::Solved : QueryStatus::NoSolution,
              .network = std::move(direct)};
    }

    auto solver = createExactMIGSATSolver(backend, conflictLimit);
    if (!solver) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ExactMIG synthesis: solver backend unavailable: "
                 << backend << "\n");
      return {.status = QueryStatus::Error};
    }

    ExactMIGSATProblem problem(*solver, numInputs, target, area);
    auto solveResult = problem.solve();
    if (solveResult.result == IncrementalSATSolver::kUNKNOWN) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ExactMIG synthesis: solver hit conflict budget at area "
                 << area << "\n");
      return {.status = QueryStatus::ConflictLimitReached};
    }
    return {.status = solveResult.network ? QueryStatus::Solved
                                          : QueryStatus::NoSolution,
            .network = std::move(solveResult.network)};
  }

  QuerySetResult synthesizeAllNormalized(unsigned numInputs,
                                         const llvm::APInt &target,
                                         unsigned area) const {
    if (area == 0) {
      auto direct = synthesizeDirect(numInputs, target);
      QuerySetResult result;
      result.status = direct ? QueryStatus::Solved : QueryStatus::NoSolution;
      if (direct)
        result.networks.push_back(std::move(*direct));
      return result;
    }

    auto solver = createExactMIGSATSolver(backend, conflictLimit);
    if (!solver) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ExactMIG synthesis: solver backend unavailable: "
                 << backend << "\n");
      return {.status = QueryStatus::Error};
    }

    ExactMIGSATProblem problem(*solver, numInputs, target, area);
    QuerySetResult result;
    while (true) {
      auto solveResult = problem.solve();
      if (solveResult.result == IncrementalSATSolver::kUNKNOWN) {
        result.status = QueryStatus::ConflictLimitReached;
        return result;
      }
      if (solveResult.result == IncrementalSATSolver::kUNSAT) {
        result.status =
            result.networks.empty() ? QueryStatus::NoSolution : QueryStatus::Solved;
        return result;
      }
      if (!solveResult.network) {
        result.status = QueryStatus::Error;
        return result;
      }
      result.networks.push_back(std::move(*solveResult.network));
      problem.blockCurrentModel();
    }
  }

  std::string backend;
  int conflictLimit;
};

static constexpr unsigned kMaxMIGExactInputs = 4;
static constexpr unsigned kMaxMIGExactSearchArea = 32;
static constexpr StringLiteral kCutRewriteCanonicalTTAttr =
    "synth.cut_rewrite.canonical_tt";

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
  SmallVector<DelayType, 6> delays(network.numInputs, 0);

  if (network.steps.empty()) {
    if (network.output.source == 0)
      return delays;
    unsigned inputPos = network.output.source - 1;
    if (inputPos < network.numInputs && network.output.inverted)
      delays[inputPos] = 1;
    return delays;
  }

  SmallVector<SmallVector<DelayType, 6>, 4> stepDelays;
  stepDelays.reserve(network.steps.size());
  auto getSourceDelays = [&](ExactSignalRef signal) {
    SmallVector<DelayType, 6> source(network.numInputs, kUnreachable);
    if (signal.source == 0)
      return source;
    if (signal.source <= network.numInputs) {
      source[signal.source - 1] = 0;
      return source;
    }
    unsigned stepIndex = signal.source - (network.numInputs + 1);
    assert(stepIndex < stepDelays.size() && "step reference out of order");
    return stepDelays[stepIndex];
  };

  for (const auto &step : network.steps) {
    SmallVector<DelayType, 6> stepDelay(network.numInputs, kUnreachable);
    for (ExactSignalRef signal : step.fanins) {
      auto childDelays = getSourceDelays(signal);
      for (unsigned i = 0; i != network.numInputs; ++i) {
        if (childDelays[i] == kUnreachable)
          continue;
        stepDelay[i] = std::max(stepDelay[i], childDelays[i] + 1);
      }
    }
    stepDelays.push_back(std::move(stepDelay));
  }

  for (unsigned i = 0; i != network.numInputs; ++i)
    if (stepDelays.back()[i] != kUnreachable)
      delays[i] = stepDelays.back()[i];
  return delays;
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

  auto getInput = [&](unsigned inputPosition) -> Value {
    assert(inputPosition < inputs.size() && "input index out of range");
    return inputs[inputPosition];
  };

  auto materializeSignal = [&](ExactSignalRef signal,
                               ArrayRef<Value> stepValues) -> Value {
    if (signal.source == 0)
      return getConstant(false);
    if (signal.source <= inputs.size())
      return getInput(signal.source - 1);

    unsigned stepIndex = signal.source - (inputs.size() + 1);
    assert(stepIndex < stepValues.size() && "invalid synthesized step index");
    return stepValues[stepIndex];
  };

  if (network.steps.empty()) {
    if (network.output.source == 0)
      return getConstant(network.output.inverted);

    Value input = getInput(network.output.source - 1);
    if (!network.output.inverted)
      return hw::WireOp::create(builder, loc, input);

    std::array<Value, 1> operands = {input};
    std::array<bool, 1> inverted = {true};
    return synth::mig::MajorityInverterOp::create(builder, loc, operands,
                                                  inverted);
  }

  SmallVector<Value, 4> stepValues;
  stepValues.reserve(network.steps.size());
  for (const auto &step : network.steps) {
    std::array<Value, 3> operands = {
        materializeSignal(step.fanins[0], stepValues),
        materializeSignal(step.fanins[1], stepValues),
        materializeSignal(step.fanins[2], stepValues)};
    std::array<bool, 3> inverted = {step.fanins[0].inverted,
                                    step.fanins[1].inverted,
                                    step.fanins[2].inverted};
    stepValues.push_back(synth::mig::MajorityInverterOp::create(
        builder, loc, operands, inverted));
  }
  return stepValues.back();
}

static unsigned computeMaterializedExactMIGArea(const ExactMIGNetwork &network) {
  if (!network.steps.empty())
    return network.steps.size();
  return network.output.source != 0 && network.output.inverted ? 1 : 0;
}

static DelayType computeExactMIGMaxDepth(const ExactMIGNetwork &network) {
  auto delays = computeExactMIGInputDelays(network);
  DelayType maxDepth = 0;
  for (DelayType delay : delays)
    maxDepth = std::max(maxDepth, delay);
  return maxDepth;
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

struct LoadedExactMIGDatabase {
  struct Entry {
    std::string moduleName;
    ExactMIGNetwork network;
    NPNClass npnClass;
    double area = 0.0;
    SmallVector<DelayType> delay;
    DelayType maxDepth = 0;
  };

  std::vector<Entry> entries;
  unsigned maxInputSize = 0;
};

static FailureOr<ExactSignalRef>
lookupExactSignal(Value value, DenseMap<Value, ExactSignalRef> &valueToSignal,
                  Operation *user) {
  auto it = valueToSignal.find(value);
  if (it == valueToSignal.end()) {
    auto diag = user->emitError(
        "cut-rewrite MIG database module used an unsupported operand");
    diag.attachNote() << "operand: " << value;
    return failure();
  }
  return it->second;
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
  assert(bodyBlock && "database module must have a body block");

  DenseMap<Value, ExactSignalRef> valueToSignal;
  for (auto [index, argument] : llvm::enumerate(bodyBlock->getArguments()))
    valueToSignal[argument] = {static_cast<unsigned>(index + 1), false};

  for (Operation &op : bodyBlock->without_terminator()) {
    if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
      if (!constant.getType().isInteger(1))
        return constant.emitError(
            "cut-rewrite MIG database constants must be i1");
      auto value = constant.getValueAttr().getValue();
      valueToSignal[constant.getResult()] = {0, value[0]};
      continue;
    }

    if (auto wire = dyn_cast<hw::WireOp>(op)) {
      auto signal = lookupExactSignal(wire.getInput(), valueToSignal, wire);
      if (failed(signal))
        return failure();
      valueToSignal[wire.getResult()] = *signal;
      continue;
    }

    if (auto majority = dyn_cast<synth::mig::MajorityInverterOp>(op)) {
      if (!majority.getResult().getType().isInteger(1))
        return majority.emitError(
            "cut-rewrite MIG database majority nodes must be i1");

      SmallVector<ExactSignalRef> operands;
      operands.reserve(majority.getInputs().size());
      for (auto [index, operand] : llvm::enumerate(majority.getInputs())) {
        auto signal = lookupExactSignal(operand, valueToSignal, majority);
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
      lookupExactSignal(output.getOperand(0), valueToSignal, output);
  if (failed(outputSignal))
    return failure();
  network.output = *outputSignal;
  return network;
}

static LogicalResult
loadExactMIGDatabaseFromModule(mlir::ModuleOp dbModule,
                               LoadedExactMIGDatabase &database) {
  database.entries.clear();
  database.maxInputSize = 0;

  for (auto hwModule : dbModule.getOps<hw::HWModuleOp>()) {
    auto canonicalTTAttr =
        hwModule->getAttrOfType<IntegerAttr>(kCutRewriteCanonicalTTAttr);
    if (!canonicalTTAttr)
      return hwModule.emitError("cut-rewrite MIG database module missing '")
             << kCutRewriteCanonicalTTAttr << "'";

    auto network = parseExactMIGNetworkFromModule(hwModule);
    if (failed(network))
      return failure();

    BinaryTruthTable canonicalTT(hwModule.getNumInputPorts(), 1,
                                 canonicalTTAttr.getValue());
    auto delay = computeExactMIGInputDelays(*network);
    database.entries.push_back(
        {.moduleName = hwModule.getModuleName().str(),
         .network = std::move(*network),
         .npnClass = NPNClass::computeNPNCanonicalForm(canonicalTT)});
    auto &entry = database.entries.back();
    entry.area = computeMaterializedExactMIGArea(entry.network);
    entry.delay.assign(delay.begin(), delay.end());
    entry.maxDepth = computeExactMIGMaxDepth(entry.network);

    database.maxInputSize = std::max(
        database.maxInputSize, static_cast<unsigned>(canonicalTT.numInputs));
  }

  if (database.entries.empty())
    return dbModule.emitError("cut-rewrite database did not contain any "
                              "matching library entries");
  return success();
}

struct FileMIGExactPattern : public CutRewritePattern {
  FileMIGExactPattern(MLIRContext *context,
                      const LoadedExactMIGDatabase::Entry &entry)
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

    // if (!cut.getNPNClass().equivalentOtherThanPermutation(entry.npnClass))
    //   return std::nullopt;

    // ExactMIGMetrics currentMetrics =
    //     computeCurrentCutMetrics(cut, logicNetwork);
    // if (!(entry.area < currentMetrics.area ||
    //       (entry.area == currentMetrics.area &&
    //        entry.maxDepth < currentMetrics.getMaxDepth())))
    //   return std::nullopt;

    return MatchResult(entry.area, entry.delay);
  }

  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(entry.npnClass);
    return true;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    const auto &logicNetwork = enumerator.getLogicNetwork();
    ExactMIGNetwork exactNetwork =
        remapExactMIGToCutInputs(entry.network, cut.getNPNClass());
    auto *rootOp = logicNetwork.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");

    SmallVector<Value> inputs;
    inputs.reserve(cut.getInputSize());
    for (uint32_t inputIndex : cut.inputs)
      inputs.push_back(logicNetwork.getValue(inputIndex));
    return materializeExactMIGNetwork(builder, rootOp->getLoc(), inputs,
                                      exactNetwork)
        .getDefiningOp();
  }

  unsigned getNumOutputs() const override { return 1; }
  StringRef getPatternName() const override { return entry.moduleName; }

private:
  const LoadedExactMIGDatabase::Entry &entry;
};

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

    auto database = std::make_shared<LoadedExactMIGDatabase>();
    if (failed(loadExactMIGDatabaseFromModule(**parsedModule, *database)))
      return failure();
    loadedMaxInputSize = database->maxInputSize;
    loadedFileDatabase = std::move(database);
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << "CutRewrite pass: module=" << module.getName()
               << " dbFile=" << dbFile << " maxCutInputSize=" << maxCutInputSize
               << " maxCutsPerRoot=" << maxCutsPerRoot << "\n");

    SmallVector<std::unique_ptr<CutRewritePattern>, 1> patterns;
    assert(loadedFileDatabase && "file database must be initialized");
    for (const auto &entry : loadedFileDatabase->entries)
      patterns.push_back(
          std::make_unique<FileMIGExactPattern>(module->getContext(), entry));

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
    LLVM_DEBUG(llvm::dbgs()
               << "CutRewrite pass complete: cutsCreated="
               << stats.numCutsCreated
               << " cutSetsCreated=" << stats.numCutSetsCreated
               << " cutsRewritten=" << stats.numCutsRewritten << "\n");
    numCutsCreated += stats.numCutsCreated;
    numCutSetsCreated += stats.numCutSetsCreated;
    numCutsRewritten += stats.numCutsRewritten;
  }

private:
  std::shared_ptr<LoadedExactMIGDatabase> loadedFileDatabase;
  unsigned loadedMaxInputSize = 0;
};

} // namespace

LogicalResult
circt::synth::emitExactMIGDatabase(mlir::ModuleOp module,
                                   const ExactMIGDatabaseGenOptions &options) {
  if (options.maxInputs > kMaxMIGExactInputs) {
    module.emitError() << "MIG exact database generation supports at most "
                       << kMaxMIGExactInputs << " inputs";
    return failure();
  }
  if (options.conflictLimit < -1) {
    module.emitError()
        << "'conflict-limit' must be greater than or equal to -1";
    return failure();
  }
  if (!createExactMIGSATSolver(options.satSolver, options.conflictLimit)) {
    module.emitError()
        << "Exact MIG database generation requires a SAT solver backend '"
        << options.satSolver << "'";
    return failure();
  }

  auto *context = module.getContext();
  Builder attrBuilder(context);
  OpBuilder builder(context);
  builder.setInsertionPointToStart(module.getBody());

  auto createTechInfo = [&](const ExactMIGNetwork &network) -> DictionaryAttr {
    auto delays = computeExactMIGInputDelays(network);
    SmallVector<Attribute> delayRows;
    delayRows.reserve(delays.size());
    for (DelayType delay : delays)
      delayRows.push_back(
          attrBuilder.getArrayAttr({attrBuilder.getI64IntegerAttr(delay)}));
    return attrBuilder.getDictionaryAttr({
        attrBuilder.getNamedAttr(
            "area",
            attrBuilder.getF64FloatAttr(
                computeMaterializedExactMIGArea(network))),
        attrBuilder.getNamedAttr("delay", attrBuilder.getArrayAttr(delayRows)),
    });
  };

  auto appendDatabaseEntry = [&](const BinaryTruthTable &canonicalTT,
                                 const ExactMIGNetwork &exactNetwork,
                                 unsigned variantIndex) {
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
    moduleName += "mig_exact_i";
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
    hwModule->setAttr("hw.techlib.info", createTechInfo(exactNetwork));
    hwModule->setAttr(
        kCutRewriteCanonicalTTAttr,
        attrBuilder.getIntegerAttr(
            builder.getIntegerType(1u << canonicalTT.numInputs),
            canonicalTT.table));

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(hwModule.getBodyBlock());
    Value result = materializeExactMIGNetwork(
        builder, hwModule.getLoc(), hwModule.getBodyBlock()->getArguments(),
        exactNetwork);
    hwModule.getBodyBlock()->getTerminator()->setOperands({result});
  };

  for (unsigned numInputs = 1; numInputs <= options.maxInputs; ++numInputs) {
    SmallVector<BinaryTruthTable> canonicalTruthTables;
    LLVM_DEBUG(llvm::dbgs()
                   << "ExactMIG dbgen: collecting canonical truth tables for "
                   << numInputs << " inputs\n";);
    collectCanonicalTruthTables(numInputs, canonicalTruthTables);

    LLVM_DEBUG(llvm::dbgs()
               << "ExactMIG dbgen: building " << canonicalTruthTables.size()
               << " canonical classes for " << numInputs << " inputs\n");

    std::vector<SmallVector<ExactMIGNetwork>> exactNetworks(
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
                if (result.status == ExactMIGSynthesizer::QueryStatus::Error) {
                  module.emitError()
                      << "failed to synthesize exact MIG for canonical truth "
                         "table "
                      << ttString;
                  return failure();
                }
                if (result.status != ExactMIGSynthesizer::QueryStatus::Solved ||
                    !result.network)
                  continue;

                exactNetworks[index].push_back(std::move(*result.network));
                LLVM_DEBUG(llvm::dbgs()
                           << "ExactMIG dbgen: found first solution for tt="
                           << ttString << " at area=" << area << "\n");
                break;
              }
              if (exactNetworks[index].empty() && !hitConflictLimit) {
                module.emitError()
                    << "failed to synthesize exact MIG for canonical truth "
                       "table "
                    << ttString;
                return failure();
              }
              if (hitConflictLimit) {
                module.emitWarning()
                    << "stopping exact MIG search for canonical truth table "
                    << ttString << " due to SAT conflict limit";
              }
              return success();
            })))
      return failure();

    for (auto [index, canonicalTT] : llvm::enumerate(canonicalTruthTables)) {
      for (auto [variantIndex, exactNetwork] :
           llvm::enumerate(exactNetworks[index]))
        appendDatabaseEntry(canonicalTT, exactNetwork, variantIndex);
    }
  }

  return success();
}
