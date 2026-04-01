//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an internal exact-MIG database builder and a generic
// built-in cut-rewrite pass backed by that database.
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
#include "mlir/Pass/Pass.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
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
  auto applyConflictLimit =
      [&](std::unique_ptr<IncrementalSATSolver> solver) {
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

enum class BuiltinCutRewriteDatabaseKind { MIGExact };

static std::optional<BuiltinCutRewriteDatabaseKind>
parseBuiltinCutRewriteDatabase(StringRef db) {
  return llvm::StringSwitch<std::optional<BuiltinCutRewriteDatabaseKind>>(db)
      .Case("MIG_EXACT", BuiltinCutRewriteDatabaseKind::MIGExact)
      .Case("mig-exact", BuiltinCutRewriteDatabaseKind::MIGExact)
      .Default(std::nullopt);
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

static void dumpExactMIGMetrics(llvm::raw_ostream &os,
                                const ExactMIGMetrics &metrics) {
  os << "area=" << metrics.area << " delays=[";
  llvm::interleaveComma(metrics.delays, os);
  os << "] maxDepth=" << metrics.getMaxDepth();
}

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

    for (unsigned operand = 0, e = gate.getNumFanins(); operand != e; ++operand) {
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

static ExactMIGMetrics computeExactMIGMetrics(const ExactMIGNetwork &network,
                                              const Cut &cut,
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

static void enumerateExactMIGCandidates(
    unsigned availableSources,
    SmallVectorImpl<ExactMIGCandidate> &candidates) {
  candidates.clear();
  for (unsigned a = 0; a + 2 < availableSources; ++a)
    for (unsigned b = a + 1; b + 1 < availableSources; ++b)
      for (unsigned c = b + 1; c < availableSources; ++c)
        for (unsigned invMask = 0; invMask != 8; ++invMask)
          candidates.push_back(
              {{{a, b, c}},
               {{static_cast<bool>(invMask & 1),
                 static_cast<bool>(invMask & 2),
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
      : solver(solver), numInputs(numInputs), target(target), numSteps(numSteps),
        numMinterms(1u << numInputs), totalSources(1 + numInputs + numSteps) {}

  SolveResult solve() {
    buildEncoding();
    LLVM_DEBUG(llvm::dbgs() << "ExactMIG SAT solve: numInputs=" << numInputs
                            << " numSteps=" << numSteps
                            << " numMinterms=" << numMinterms
                            << " vars=" << nextVar << "\n");
    auto result = solver.solve();
    if (result != IncrementalSATSolver::kSAT) {
      LLVM_DEBUG(llvm::dbgs() << "ExactMIG SAT result: "
                              << (result == IncrementalSATSolver::kUNSAT
                                      ? "UNSAT"
                                      : "UNKNOWN")
                              << " for area " << numSteps << "\n");
      return {.result = result};
    }
    auto network = decodeModel();
    LLVM_DEBUG({
      llvm::dbgs() << "ExactMIG SAT result: SAT for area " << numSteps
                   << "\n";
      dumpExactMIGNetwork(llvm::dbgs(), network);
    });
    return {.result = result, .network = std::move(network)};
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
};

class ExactMIGSynthesizer {
public:
  explicit ExactMIGSynthesizer(StringRef backend, int conflictLimit = -1)
      : backend(backend), conflictLimit(conflictLimit) {}

  FailureOr<std::optional<ExactMIGNetwork>>
  getOrCompute(const BinaryTruthTable &tt, unsigned maxArea) {
    auto [normalizedTT, invertOutput] = normalize(tt);
    auto key = std::make_pair(normalizedTT.table, normalizedTT.numInputs);
    auto [it, inserted] = cache.try_emplace(key);
    (void)inserted;
    auto &entry = it->second;
    LLVM_DEBUG({
      llvm::dbgs() << "ExactMIG query: inputs=" << tt.numInputs
                   << " maxArea=" << maxArea << " tt=" << tt.table
                   << " normalized=" << normalizedTT.table
                   << " invertOutput=" << invertOutput
                   << " cacheState=";
      if (entry.solution)
        llvm::dbgs() << "hit(area=" << entry.solution->steps.size() << ")";
      else
        llvm::dbgs() << "miss(searchedUpTo=" << entry.searchedUpTo << ")";
      llvm::dbgs() << "\n";
    });

    if (!entry.solution &&
        (entry.searchedUpTo < 0 ||
         maxArea > static_cast<unsigned>(entry.searchedUpTo))) {
      unsigned firstArea =
          entry.searchedUpTo < 0 ? 0u : static_cast<unsigned>(entry.searchedUpTo + 1);
      for (unsigned area = firstArea; area <= maxArea; ++area) {
        LLVM_DEBUG(llvm::dbgs() << "ExactMIG search: trying area " << area
                                << " for normalized function "
                                << normalizedTT.table << "\n");
        auto candidate = synthesizeNormalized(normalizedTT.numInputs,
                                              normalizedTT.table, area);
        if (failed(candidate))
          return failure();
        if (*candidate) {
          LLVM_DEBUG(llvm::dbgs() << "ExactMIG search: found solution at area "
                                  << area << "\n");
          entry.solution = std::move(**candidate);
          entry.searchedUpTo = area;
          break;
        }
        entry.searchedUpTo = area;
      }
    }

    if (!entry.solution) {
      LLVM_DEBUG(llvm::dbgs() << "ExactMIG query: no solution up to area "
                              << maxArea << "\n");
      return std::optional<ExactMIGNetwork>{};
    }

    ExactMIGNetwork result = *entry.solution;
    if (invertOutput)
      result = invertExactMIGOutput(std::move(result));
    LLVM_DEBUG({
      llvm::dbgs() << "ExactMIG query: returning "
                   << (invertOutput ? "output-inverted " : "") << "solution\n";
      dumpExactMIGNetwork(llvm::dbgs(), result);
    });
    return std::optional<ExactMIGNetwork>(std::move(result));
  }

  void setConflictLimit(int limit) { conflictLimit = limit; }

private:
  struct CacheEntry {
    std::optional<ExactMIGNetwork> solution;
    int searchedUpTo = -1;
  };

  static std::pair<BinaryTruthTable, bool>
  normalize(const BinaryTruthTable &tt) {
    bool invertOutput = tt.table[0];
    if (!invertOutput)
      return {tt, false};

    BinaryTruthTable normalized = tt;
    normalized.table.flipAllBits();
    LLVM_DEBUG(llvm::dbgs() << "ExactMIG normalize: flipping output polarity "
                            << tt.table << " -> " << normalized.table
                            << "\n");
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
        LLVM_DEBUG(llvm::dbgs() << "ExactMIG direct synthesis: input i" << input
                                << "\n");
        return network;
      }
      llvm::APInt invertedMask = mask;
      invertedMask.flipAllBits();
      if (target == invertedMask) {
        network.output = {1 + input, true};
        LLVM_DEBUG(llvm::dbgs()
                   << "ExactMIG direct synthesis: inverted input i" << input
                   << "\n");
        return network;
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "ExactMIG direct synthesis: no area-0 implementation\n");
    return std::nullopt;
  }

  FailureOr<std::optional<ExactMIGNetwork>>
  synthesizeNormalized(unsigned numInputs, const llvm::APInt &target,
                       unsigned area) const {
    if (area == 0)
      return synthesizeDirect(numInputs, target);

    auto solver = createExactMIGSATSolver(backend, conflictLimit);
    if (!solver) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ExactMIG synthesis: solver backend unavailable: "
                 << backend << "\n");
      return failure();
    }

    ExactMIGSATProblem problem(*solver, numInputs, target, area);
    auto solveResult = problem.solve();
    if (solveResult.result == IncrementalSATSolver::kUNKNOWN) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ExactMIG synthesis: solver hit conflict budget at area "
                 << area << "\n");
      return failure();
    }
    return std::move(solveResult.network);
  }

  std::string backend;
  int conflictLimit;
  DenseMap<std::pair<APInt, unsigned>, CacheEntry> cache;
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
    stepValues.push_back(
        synth::mig::MajorityInverterOp::create(builder, loc, operands,
                                               inverted));
  }
  return stepValues.back();
}

class MIGExactDatabase {
public:
  static MIGExactDatabase &get() {
    static MIGExactDatabase database;
    return database;
  }

  LogicalResult validate(unsigned maxInputs, int conflictLimit,
                         MLIRContext *context) {
    if (maxInputs > kMaxMIGExactInputs) {
      emitError(UnknownLoc::get(context))
          << "MIG_EXACT supports at most " << kMaxMIGExactInputs
          << " cut inputs";
      return failure();
    }
    if (conflictLimit < -1) {
      emitError(UnknownLoc::get(context))
          << "'conflict-limit' must be greater than or equal to -1";
      return failure();
    }

    if (!createExactMIGSATSolver("auto", conflictLimit)) {
      emitError(UnknownLoc::get(context))
          << "MIG_EXACT requires a SAT solver, but none is available in this "
             "build";
      return failure();
    }
    synthesizer.setConflictLimit(conflictLimit);
    return success();
  }

  LogicalResult ensureInitialized(unsigned maxInputs) {
    std::lock_guard<std::mutex> lock(mutex);
    if (initializedUpTo >= maxInputs)
      return success();
    for (unsigned numInputs = initializedUpTo + 1; numInputs <= maxInputs;
         ++numInputs) {
      if (failed(buildForInputSizeLocked(numInputs)))
        return failure();
      initializedUpTo = numInputs;
    }
    return success();
  }

  const ExactMIGNetwork *lookup(const NPNClass &npnClass) const {
    auto it = entries.find({npnClass.truthTable.table, npnClass.truthTable.numInputs});
    if (it == entries.end())
      return nullptr;
    return &it->second;
  }

private:
  LogicalResult buildForInputSizeLocked(unsigned numInputs) {
    assert(numInputs <= kMaxMIGExactInputs && "unsupported input size");

    uint64_t numFunctions = 1ULL << (1u << numInputs);
    DenseSet<APInt> seenCanonicalTruthTables;
    unsigned classesBuilt = 0;

    LLVM_DEBUG(llvm::dbgs() << "MIG_EXACT db: building " << numInputs
                            << "-input canonical classes from " << numFunctions
                            << " functions\n");

    for (uint64_t value = 0; value != numFunctions; ++value) {
      BinaryTruthTable tt(numInputs, 1, APInt(1u << numInputs, value));
      NPNClass npnClass = NPNClass::computeNPNCanonicalForm(tt);
      if (!seenCanonicalTruthTables.insert(npnClass.truthTable.table).second)
        continue;

      auto exactNetwork =
          synthesizer.getOrCompute(npnClass.truthTable, kMaxMIGExactSearchArea);
      if (failed(exactNetwork) || !*exactNetwork)
        return failure();

      entries.try_emplace({npnClass.truthTable.table, numInputs},
                          std::move(**exactNetwork));
      ++classesBuilt;
      LLVM_DEBUG({
        llvm::dbgs() << "MIG_EXACT db: class " << classesBuilt << " for "
                     << numInputs << " inputs tt="
                     << npnClass.truthTable.table << "\n";
        dumpExactMIGNetwork(llvm::dbgs(), entries.find(
                                             {npnClass.truthTable.table,
                                              numInputs})
                                             ->second);
      });
    }

    LLVM_DEBUG(llvm::dbgs() << "MIG_EXACT db: built " << classesBuilt
                            << " canonical classes for " << numInputs
                            << " inputs\n");
    return success();
  }

  std::mutex mutex;
  unsigned initializedUpTo = 0;
  ExactMIGSynthesizer synthesizer{"auto"};
  DenseMap<std::pair<APInt, unsigned>, ExactMIGNetwork> entries;
};

struct MIGExactPattern : public CutRewritePattern {
  MIGExactPattern(MLIRContext *context, MIGExactDatabase &database)
      : CutRewritePattern(context), database(database) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    const auto &logicNetwork = enumerator.getLogicNetwork();
    if (cut.isTrivialCut() || cut.getOutputSize(logicNetwork) != 1)
      return std::nullopt;

    auto *rootOp = logicNetwork.getGate(cut.getRootIndex()).getOperation();
    if (!rootOp || !rootOp->getResult(0).getType().isInteger(1))
      return std::nullopt;

    for (uint32_t inputIndex : cut.inputs) {
      if (logicNetwork.getGate(inputIndex).getKind() == LogicNetworkGate::Constant)
        continue;
      Value inputValue = logicNetwork.getValue(inputIndex);
      if (!inputValue || !inputValue.getType().isInteger(1))
        return std::nullopt;
    }

    const NPNClass &npnClass = cut.getNPNClass();
    if (failed(database.ensureInitialized(npnClass.truthTable.numInputs)))
      return std::nullopt;
    const ExactMIGNetwork *canonicalNetwork = database.lookup(npnClass);
    if (!canonicalNetwork) {
      LLVM_DEBUG(llvm::dbgs()
                 << "MIG_EXACT match: no database entry for canonical tt "
                 << npnClass.truthTable.table << "\n");
      return std::nullopt;
    }

    ExactMIGNetwork exactNetwork =
        remapExactMIGToCutInputs(*canonicalNetwork, npnClass);
    ExactMIGMetrics currentMetrics =
        computeCurrentCutMetrics(cut, logicNetwork);
    ExactMIGMetrics exactMetrics =
        computeExactMIGMetrics(exactNetwork, cut, logicNetwork);
    bool isBetter = exactMetrics.area < currentMetrics.area ||
                    (exactMetrics.area == currentMetrics.area &&
                     exactMetrics.getMaxDepth() < currentMetrics.getMaxDepth());

    LLVM_DEBUG({
      llvm::dbgs() << "MIG_EXACT match: root="
                   << rootOp->getName().getStringRef() << " canonicalTT="
                   << npnClass.truthTable.table << " current ";
      dumpExactMIGMetrics(llvm::dbgs(), currentMetrics);
      llvm::dbgs() << " candidate ";
      dumpExactMIGMetrics(llvm::dbgs(), exactMetrics);
      llvm::dbgs() << " better=" << isBetter << "\n";
      dumpExactMIGNetwork(llvm::dbgs(), exactNetwork);
    });

    if (!isBetter)
      return std::nullopt;

    MatchResult result;
    result.area = exactMetrics.area;
    result.setOwnedDelays(std::move(exactMetrics.delays));
    return result;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    const auto &logicNetwork = enumerator.getLogicNetwork();
    const NPNClass &npnClass = cut.getNPNClass();
    if (failed(database.ensureInitialized(npnClass.truthTable.numInputs)))
      return failure();
    const ExactMIGNetwork *canonicalNetwork = database.lookup(npnClass);
    if (!canonicalNetwork)
      return failure();

    ExactMIGNetwork exactNetwork =
        remapExactMIGToCutInputs(*canonicalNetwork, npnClass);
    auto *rootOp = logicNetwork.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");
    Location loc = rootOp->getLoc();

    LLVM_DEBUG({
      llvm::dbgs() << "MIG_EXACT rewrite: replacing "
                   << rootOp->getName().getStringRef() << " canonicalTT="
                   << npnClass.truthTable.table << "\n";
      dumpExactMIGNetwork(llvm::dbgs(), exactNetwork);
    });

    Value constZero;
    Value constOne;
    auto getConstant = [&](bool value) -> Value {
      if (!value) {
        if (!constZero)
          constZero = hw::ConstantOp::create(builder, loc, llvm::APInt(1, 0));
        return constZero;
      }
      if (!constOne)
        constOne = hw::ConstantOp::create(builder, loc, llvm::APInt(1, 1));
      return constOne;
    };

    auto getBoundaryValue = [&](unsigned inputPosition) -> Value {
      uint32_t networkIndex = cut.inputs[inputPosition];
      switch (logicNetwork.getGate(networkIndex).getKind()) {
      case LogicNetworkGate::Constant:
        return getConstant(networkIndex == LogicNetwork::kConstant1);
      default:
        return logicNetwork.getValue(networkIndex);
      }
    };

    auto materializeSignal = [&](ExactSignalRef signal,
                                 ArrayRef<Value> stepValues) -> Value {
      if (signal.source == 0)
        return getConstant(false);
      if (signal.source <= cut.getInputSize())
        return getBoundaryValue(signal.source - 1);

      unsigned stepIndex = signal.source - (cut.getInputSize() + 1);
      assert(stepIndex < stepValues.size() && "invalid synthesized step index");
      return stepValues[stepIndex];
    };

    if (exactNetwork.steps.empty()) {
      if (exactNetwork.output.source == 0)
        return getConstant(exactNetwork.output.inverted).getDefiningOp();

      Value input = getBoundaryValue(exactNetwork.output.source - 1);
      if (!exactNetwork.output.inverted)
        return hw::WireOp::create(builder, loc, input).getOperation();

      std::array<Value, 1> operands = {input};
      std::array<bool, 1> inverted = {true};
      Value inverter = synth::mig::MajorityInverterOp::create(builder, loc,
                                                              operands,
                                                              inverted);
      return inverter.getDefiningOp();
    }

    SmallVector<Value, 4> stepValues;
    stepValues.reserve(exactNetwork.steps.size());
    for (const auto &step : exactNetwork.steps) {
      std::array<Value, 3> operands = {
          materializeSignal(step.fanins[0], stepValues),
          materializeSignal(step.fanins[1], stepValues),
          materializeSignal(step.fanins[2], stepValues)};
      std::array<bool, 3> inverted = {step.fanins[0].inverted,
                                      step.fanins[1].inverted,
                                      step.fanins[2].inverted};
      stepValues.push_back(
          synth::mig::MajorityInverterOp::create(builder, loc, operands,
                                                 inverted));
    }
    return stepValues.back().getDefiningOp();
  }

  unsigned getNumOutputs() const override { return 1; }
  StringRef getPatternName() const override { return "MIG_EXACT"; }

private:
  MIGExactDatabase &database;
};

struct LoadedExactMIGDatabase {
  const ExactMIGNetwork *lookup(const BinaryTruthTable &tt) const {
    auto it = entries.find({tt.table, tt.numInputs});
    if (it == entries.end())
      return nullptr;
    return &it->second;
  }

  DenseMap<std::pair<APInt, unsigned>, ExactMIGNetwork> entries;
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
    auto [it, inserted] = database.entries.try_emplace(
        {canonicalTT.table, canonicalTT.numInputs}, std::move(*network));
    if (!inserted)
      return hwModule.emitError("duplicate canonical truth table in "
                                "cut-rewrite MIG database");

    database.maxInputSize =
        std::max(database.maxInputSize, static_cast<unsigned>(canonicalTT.numInputs));
  }

  if (database.entries.empty())
    return dbModule.emitError("cut-rewrite database did not contain any "
                              "matching library entries");
  return success();
}

struct FileMIGExactPattern : public CutRewritePattern {
  FileMIGExactPattern(MLIRContext *context, LoadedExactMIGDatabase &database)
      : CutRewritePattern(context), database(database) {}

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

    const NPNClass &npnClass = cut.getNPNClass();
    const ExactMIGNetwork *canonicalNetwork = database.lookup(npnClass.truthTable);
    if (!canonicalNetwork)
      return std::nullopt;

    ExactMIGNetwork exactNetwork =
        remapExactMIGToCutInputs(*canonicalNetwork, npnClass);
    ExactMIGMetrics currentMetrics =
        computeCurrentCutMetrics(cut, logicNetwork);
    ExactMIGMetrics exactMetrics =
        computeExactMIGMetrics(exactNetwork, cut, logicNetwork);
    if (!(exactMetrics.area < currentMetrics.area ||
          (exactMetrics.area == currentMetrics.area &&
           exactMetrics.getMaxDepth() < currentMetrics.getMaxDepth())))
      return std::nullopt;

    MatchResult result;
    result.area = exactMetrics.area;
    result.setOwnedDelays(std::move(exactMetrics.delays));
    return result;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut) const override {
    const auto &logicNetwork = enumerator.getLogicNetwork();
    const NPNClass &npnClass = cut.getNPNClass();
    const ExactMIGNetwork *canonicalNetwork = database.lookup(npnClass.truthTable);
    if (!canonicalNetwork)
      return failure();

    ExactMIGNetwork exactNetwork =
        remapExactMIGToCutInputs(*canonicalNetwork, npnClass);
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
  StringRef getPatternName() const override { return "MIG_EXACT_FILE_DB"; }

private:
  LoadedExactMIGDatabase &database;
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

struct CutRewritePass : public circt::synth::impl::CutRewriteBase<CutRewritePass> {
  using circt::synth::impl::CutRewriteBase<CutRewritePass>::CutRewriteBase;

  LogicalResult initialize(MLIRContext *context) override {
    loadedMaxInputSize = maxCutInputSize;
    loadedFileDatabase.reset();
    builtinDBKind.reset();

    if (!dbFile.empty()) {
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

    builtinDBKind = parseBuiltinCutRewriteDatabase(db);
    if (!builtinDBKind) {
      emitError(UnknownLoc::get(context))
          << "unknown cut rewrite database '" << db << "'";
      return failure();
    }

    switch (*builtinDBKind) {
    case BuiltinCutRewriteDatabaseKind::MIGExact:
      return MIGExactDatabase::get().validate(maxCutInputSize,
                                              static_cast<int>(conflictLimit),
                                              context);
    }
    llvm_unreachable("unexpected built-in cut rewrite database kind");
  }

  void runOnOperation() override {
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "CutRewrite pass: module=" << module.getName()
                            << " db=" << db
                            << " dbFile=" << dbFile
                            << " maxCutInputSize=" << maxCutInputSize
                            << " maxCutsPerRoot=" << maxCutsPerRoot
                            << " conflictLimit=" << conflictLimit << "\n");

    SmallVector<std::unique_ptr<CutRewritePattern>, 1> patterns;
    if (loadedFileDatabase) {
      patterns.push_back(std::make_unique<FileMIGExactPattern>(
          module->getContext(), *loadedFileDatabase));
    } else {
      assert(builtinDBKind && "built-in database kind must be initialized");
      switch (*builtinDBKind) {
      case BuiltinCutRewriteDatabaseKind::MIGExact: {
        auto &database = MIGExactDatabase::get();
        patterns.push_back(
            std::make_unique<MIGExactPattern>(module->getContext(), database));
        break;
      }
      }
    }

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
    LLVM_DEBUG(llvm::dbgs() << "CutRewrite pass complete: cutsCreated="
                            << stats.numCutsCreated
                            << " cutSetsCreated=" << stats.numCutSetsCreated
                            << " cutsRewritten=" << stats.numCutsRewritten
                            << "\n");
    numCutsCreated += stats.numCutsCreated;
    numCutSetsCreated += stats.numCutSetsCreated;
    numCutsRewritten += stats.numCutsRewritten;
  }

private:
  std::optional<BuiltinCutRewriteDatabaseKind> builtinDBKind;
  std::shared_ptr<LoadedExactMIGDatabase> loadedFileDatabase;
  unsigned loadedMaxInputSize = 0;
};

} // namespace

LogicalResult circt::synth::emitExactMIGDatabase(
    mlir::ModuleOp module, const ExactMIGDatabaseGenOptions &options) {
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

  ExactMIGSynthesizer synthesizer(options.satSolver, options.conflictLimit);

  auto createTechInfo = [&](const ExactMIGNetwork &network) -> DictionaryAttr {
    auto delays = computeExactMIGInputDelays(network);
    SmallVector<Attribute> delayRows;
    delayRows.reserve(delays.size());
    for (DelayType delay : delays)
      delayRows.push_back(attrBuilder.getArrayAttr(
          {attrBuilder.getI64IntegerAttr(delay)}));
    return attrBuilder.getDictionaryAttr({
        attrBuilder.getNamedAttr("area",
                                 attrBuilder.getF64FloatAttr(network.steps.size())),
        attrBuilder.getNamedAttr("delay",
                                 attrBuilder.getArrayAttr(delayRows)),
    });
  };

  for (unsigned numInputs = 1; numInputs <= options.maxInputs; ++numInputs) {
    uint64_t numFunctions = 1ULL << (1u << numInputs);
    DenseSet<APInt> seenCanonicalTruthTables;
    for (uint64_t value = 0; value != numFunctions; ++value) {
      BinaryTruthTable tt(numInputs, 1, APInt(1u << numInputs, value));
      NPNClass npnClass = NPNClass::computeNPNCanonicalForm(tt);
      if (!seenCanonicalTruthTables.insert(npnClass.truthTable.table).second)
        continue;

      auto exactNetwork =
          synthesizer.getOrCompute(npnClass.truthTable, kMaxMIGExactSearchArea);
      if (failed(exactNetwork) || !*exactNetwork) {
        SmallString<32> ttString;
        npnClass.truthTable.table.toStringUnsigned(ttString, 16);
        module.emitError()
            << "failed to synthesize exact MIG for canonical truth table "
            << ttString;
        return failure();
      }

      SmallVector<hw::PortInfo> inputs;
      inputs.reserve(numInputs);
      for (unsigned i = 0; i != numInputs; ++i) {
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
      moduleName += Twine(numInputs).str();
      moduleName += "_tt_";
      SmallString<32> ttString;
      npnClass.truthTable.table.toStringUnsigned(ttString, 16);
      moduleName += ttString;

      hw::HWModuleOp hwModule =
          hw::HWModuleOp::create(builder, module.getLoc(),
                                 attrBuilder.getStringAttr(moduleName),
                                 hw::ModulePortInfo(inputs, {output}));
      hwModule->setAttr("hw.techlib.info", createTechInfo(**exactNetwork));
      hwModule->setAttr(
          kCutRewriteCanonicalTTAttr,
          attrBuilder.getIntegerAttr(
              builder.getIntegerType(1u << numInputs),
              npnClass.truthTable.table));

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(hwModule.getBodyBlock());
      Value result = materializeExactMIGNetwork(
          builder, hwModule.getLoc(), ValueRange(hwModule.getBodyBlock()->getArguments()),
          **exactNetwork);
      hwModule.getBodyBlock()->getTerminator()->setOperands({result});
    }
  }

  return success();
}
