//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SAT-based exact synthesis for small majority-inverter
// graphs and applies the result through the generic cut-rewriting framework.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/SATSolver.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <array>
#include <functional>
#include <limits>
#include <optional>

#define DEBUG_TYPE "synth-exact-mig"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_EXACTMIG
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace {

static std::unique_ptr<IncrementalSATSolver>
createExactMIGSATSolver(StringRef backend) {
  if (backend == "auto") {
    if (auto solver = createCadicalSATSolver())
      return solver;
    return createZ3SATSolver();
  }
  if (backend == "cadical")
    return createCadicalSATSolver();
  if (backend == "z3")
    return createZ3SATSolver();
  return {};
}

static bool isKnownSATSolverBackend(StringRef backend) {
  return backend == "auto" || backend == "z3" || backend == "cadical";
}

static inline llvm::APInt applyExactGateSemantics(LogicNetworkGate::Kind kind,
                                                  const llvm::APInt &a) {
  switch (kind) {
  case LogicNetworkGate::Identity:
    return a;
  default:
    llvm_unreachable("unsupported unary operation");
  }
}

static inline llvm::APInt applyExactGateSemantics(LogicNetworkGate::Kind kind,
                                                  const llvm::APInt &a,
                                                  const llvm::APInt &b) {
  switch (kind) {
  case LogicNetworkGate::And2:
    return a & b;
  case LogicNetworkGate::Xor2:
    return a ^ b;
  default:
    llvm_unreachable("unsupported binary operation");
  }
}

static inline llvm::APInt applyExactGateSemantics(LogicNetworkGate::Kind kind,
                                                  const llvm::APInt &a,
                                                  const llvm::APInt &b,
                                                  const llvm::APInt &c) {
  switch (kind) {
  case LogicNetworkGate::Maj3:
    return (a & b) | (a & c) | (b & c);
  default:
    llvm_unreachable("unsupported ternary operation");
  }
}

static llvm::APInt
simulateExactCutNode(const LogicNetwork &network, uint32_t index,
                     llvm::DenseMap<uint32_t, llvm::APInt> &cache,
                     unsigned numInputs) {
  auto it = cache.find(index);
  if (it != cache.end())
    return it->second;

  const auto &gate = network.getGate(index);
  auto getEdgeTT = [&](Signal edge) {
    llvm::APInt tt =
        simulateExactCutNode(network, edge.getIndex(), cache, numInputs);
    if (edge.isInverted())
      tt.flipAllBits();
    return tt;
  };

  llvm::APInt result;
  switch (gate.getKind()) {
  case LogicNetworkGate::Constant:
    result = index == LogicNetwork::kConstant0
                 ? llvm::APInt::getZero(1U << numInputs)
                 : llvm::APInt::getAllOnes(1U << numInputs);
    break;
  case LogicNetworkGate::PrimaryInput:
    llvm_unreachable("cut boundary inputs must be initialized in cache");
  case LogicNetworkGate::And2:
  case LogicNetworkGate::Xor2:
    result = applyExactGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]),
                                     getEdgeTT(gate.edges[1]));
    break;
  case LogicNetworkGate::Maj3:
    result = applyExactGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]),
                                     getEdgeTT(gate.edges[1]),
                                     getEdgeTT(gate.edges[2]));
    break;
  case LogicNetworkGate::Identity:
    result = applyExactGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]));
    break;
  }

  cache[index] = result;
  return result;
}

static BinaryTruthTable computeExactCutTruthTable(const Cut &cut,
                                                  const LogicNetwork &network) {
  unsigned numInputs = cut.getInputSize();
  llvm::DenseMap<uint32_t, llvm::APInt> cache;
  cache.reserve(cut.inputs.size());
  for (auto [inputIdx, networkIndex] : llvm::enumerate(cut.inputs)) {
    const auto &gate = network.getGate(networkIndex);
    if (gate.getKind() == LogicNetworkGate::Constant) {
      cache[networkIndex] =
          networkIndex == LogicNetwork::kConstant0
              ? llvm::APInt::getZero(1U << numInputs)
              : llvm::APInt::getAllOnes(1U << numInputs);
      continue;
    }
    cache[networkIndex] = circt::createVarMask(numInputs, inputIdx, true);
  }

  llvm::APInt result =
      simulateExactCutNode(network, cut.getRootIndex(), cache, numInputs);
  return BinaryTruthTable(numInputs, 1, result);
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
  ExactMIGSATProblem(IncrementalSATSolver &solver, unsigned numInputs,
                     const llvm::APInt &target, unsigned numSteps)
      : solver(solver), numInputs(numInputs), target(target), numSteps(numSteps),
        numMinterms(1u << numInputs), totalSources(1 + numInputs + numSteps) {}

  std::optional<ExactMIGNetwork> solve() {
    buildEncoding();
    if (solver.solve() != IncrementalSATSolver::kSAT)
      return std::nullopt;
    return decodeModel();
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
  explicit ExactMIGSynthesizer(StringRef backend) : backend(backend) {}

  std::optional<ExactMIGNetwork> getOrCompute(const BinaryTruthTable &tt,
                                              unsigned maxArea) {
    auto [normalizedTT, invertOutput] = normalize(tt);
    auto key = std::make_pair(normalizedTT.table, normalizedTT.numInputs);
    auto [it, inserted] = cache.try_emplace(key);
    (void)inserted;
    auto &entry = it->second;

    if (!entry.solution &&
        (entry.searchedUpTo < 0 ||
         maxArea > static_cast<unsigned>(entry.searchedUpTo))) {
      unsigned firstArea =
          entry.searchedUpTo < 0 ? 0u : static_cast<unsigned>(entry.searchedUpTo + 1);
      for (unsigned area = firstArea; area <= maxArea; ++area) {
        auto candidate = synthesizeNormalized(normalizedTT.numInputs,
                                              normalizedTT.table, area);
        if (candidate) {
          entry.solution = std::move(candidate);
          entry.searchedUpTo = area;
          break;
        }
        entry.searchedUpTo = area;
      }
    }

    if (!entry.solution)
      return std::nullopt;

    ExactMIGNetwork result = *entry.solution;
    if (invertOutput)
      result = invertExactMIGOutput(std::move(result));
    return result;
  }

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

  std::optional<ExactMIGNetwork>
  synthesizeNormalized(unsigned numInputs, const llvm::APInt &target,
                       unsigned area) const {
    if (area == 0)
      return synthesizeDirect(numInputs, target);

    auto solver = createExactMIGSATSolver(backend);
    if (!solver)
      return std::nullopt;

    ExactMIGSATProblem problem(*solver, numInputs, target, area);
    return problem.solve();
  }

  std::string backend;
  DenseMap<std::pair<APInt, unsigned>, CacheEntry> cache;
};

struct ExactMIGPattern : public CutRewritePattern {
  ExactMIGPattern(MLIRContext *context, StringRef backend)
      : CutRewritePattern(context), synthesizer(backend) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    const auto &network = enumerator.getLogicNetwork();
    if (cut.isTrivialCut() || cut.getOutputSize(network) != 1)
      return std::nullopt;

    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    if (!rootOp || !rootOp->getResult(0).getType().isInteger(1))
      return std::nullopt;

    for (uint32_t inputIndex : cut.inputs) {
      if (network.getGate(inputIndex).getKind() == LogicNetworkGate::Constant)
        continue;
      Value inputValue = network.getValue(inputIndex);
      if (!inputValue || !inputValue.getType().isInteger(1))
        return std::nullopt;
    }

    BinaryTruthTable truthTable = computeExactCutTruthTable(cut, network);
    ExactMIGMetrics currentMetrics = computeCurrentCutMetrics(cut, network);
    auto exactNetwork =
        synthesizer.getOrCompute(truthTable, currentMetrics.area);
    if (!exactNetwork)
      return std::nullopt;

    ExactMIGMetrics exactMetrics =
        computeExactMIGMetrics(*exactNetwork, cut, network);
    bool isBetter = exactMetrics.area < currentMetrics.area ||
                    (exactMetrics.area == currentMetrics.area &&
                     exactMetrics.getMaxDepth() < currentMetrics.getMaxDepth());
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
    BinaryTruthTable truthTable = computeExactCutTruthTable(cut, logicNetwork);
    auto exactNetwork = synthesizer.getOrCompute(
        truthTable, std::numeric_limits<unsigned>::max());
    if (!exactNetwork)
      return failure();

    auto *rootOp = logicNetwork.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");
    Location loc = rootOp->getLoc();

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

    if (exactNetwork->steps.empty()) {
      if (exactNetwork->output.source == 0) {
        Value constant = getConstant(exactNetwork->output.inverted);
        return constant.getDefiningOp();
      }

      Value input = getBoundaryValue(exactNetwork->output.source - 1);
      if (!exactNetwork->output.inverted) {
        auto wire = hw::WireOp::create(builder, loc, input);
        return wire.getOperation();
      }

      std::array<Value, 1> operands = {input};
      std::array<bool, 1> inverted = {true};
      Value inverter =
          synth::mig::MajorityInverterOp::create(builder, loc, operands,
                                                 inverted);
      return inverter.getDefiningOp();
    }

    SmallVector<Value, 4> stepValues;
    stepValues.reserve(exactNetwork->steps.size());
    for (const auto &step : exactNetwork->steps) {
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
  StringRef getPatternName() const override { return "exact-mig"; }

private:
  mutable ExactMIGSynthesizer synthesizer;
};

struct ExactMIGPass : public circt::synth::impl::ExactMIGBase<ExactMIGPass> {
  using circt::synth::impl::ExactMIGBase<ExactMIGPass>::ExactMIGBase;

  void runOnOperation() override {
    auto module = getOperation();
    if (!isKnownSATSolverBackend(satSolver)) {
      module->emitError() << "unknown SAT solver backend '" << satSolver << "'";
      return signalPassFailure();
    }

    if (!createExactMIGSATSolver(satSolver)) {
      module->emitError()
          << "ExactMIG requires a SAT solver, but backend '" << satSolver
          << "' is not available in this build";
      return signalPassFailure();
    }

    CutRewriterOptions options;
    options.strategy = OptimizationStrategyArea;
    options.maxCutInputSize = maxCutInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.allowNoMatch = true;

    SmallVector<std::unique_ptr<CutRewritePattern>, 1> patterns;
    patterns.push_back(
        std::make_unique<ExactMIGPattern>(module->getContext(), satSolver));
    CutRewritePatternSet patternSet(std::move(patterns));
    CutRewriter rewriter(options, patternSet);
    if (failed(rewriter.run(module)))
      return signalPassFailure();

    const auto &stats = rewriter.getStats();
    numCutsCreated += stats.numCutsCreated;
    numCutSetsCreated += stats.numCutSetsCreated;
    numCutsRewritten += stats.numCutsRewritten;
  }
};

} // namespace
