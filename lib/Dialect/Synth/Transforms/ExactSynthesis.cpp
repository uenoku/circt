//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements exact synthesis of small Boolean truth tables.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/ExactSynthesis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Naming.h"
#include "circt/Support/SATSolver.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_EXACTSYNTHESIS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace {

static constexpr unsigned kMaxExactSynthesisInputs = 4;
static constexpr unsigned kMaxExactSearchArea = 32;

enum class ExactNodeKind {
  And2,
  Xor2,
  Dot3,
};

struct ExactSignalRef {
  // `source == 0` denotes the constant-false source. All other sources are
  // numbered as 1-based primary inputs followed by synthesized steps.
  unsigned source = 0;
  bool inverted = false;
};

struct ExactNetworkStep {
  ExactNodeKind kind = ExactNodeKind::And2;
  SmallVector<ExactSignalRef, 3> fanins;
};

struct ExactNetwork {
  unsigned numInputs = 0;
  SmallVector<ExactNetworkStep, 4> steps;
  ExactSignalRef output;
};

using ExactCandidate = ExactNetworkStep;

static bool hasEnabledConstructs(const ExactSynthesisPolicy &policy) {
  return policy.allowAnd || policy.allowXor || policy.allowDot;
}

static unsigned getCandidateInversionMask(const ExactCandidate &candidate) {
  unsigned mask = 0;
  for (auto [index, fanin] : llvm::enumerate(candidate.fanins))
    if (fanin.inverted)
      mask |= 1u << index;
  return mask;
}

static bool isCandidateLess(const ExactCandidate &lhs,
                            const ExactCandidate &rhs) {
  if (lhs.fanins.size() != rhs.fanins.size())
    return lhs.fanins.size() < rhs.fanins.size();
  for (size_t i = 0, e = lhs.fanins.size(); i != e; ++i) {
    if (lhs.fanins[i].source != rhs.fanins[i].source)
      return lhs.fanins[i].source < rhs.fanins[i].source;
  }
  if (lhs.kind != rhs.kind)
    return static_cast<unsigned>(lhs.kind) < static_cast<unsigned>(rhs.kind);
  return getCandidateInversionMask(lhs) < getCandidateInversionMask(rhs);
}

static std::optional<ExactNetwork> synthesizeDirect(unsigned numInputs,
                                                    const APInt &target) {
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
    APInt mask = circt::createVarMask(numInputs, input, true);
    if (target == mask) {
      network.output = {1 + input, false};
      return network;
    }
    APInt invertedMask = mask;
    invertedMask.flipAllBits();
    if (target == invertedMask) {
      network.output = {1 + input, true};
      return network;
    }
  }
  return std::nullopt;
}

static void enumerateCandidates(const ExactSynthesisPolicy &policy,
                                unsigned availableSources,
                                SmallVectorImpl<ExactCandidate> &candidates) {
  candidates.clear();

  if (policy.allowAnd) {
    for (unsigned a = 1; a + 1 < availableSources; ++a)
      for (unsigned b = a + 1; b < availableSources; ++b)
        for (unsigned invMask = 0; invMask != 4; ++invMask) {
          ExactCandidate candidate;
          candidate.kind = ExactNodeKind::And2;
          candidate.fanins.push_back({a, static_cast<bool>(invMask & 1)});
          candidate.fanins.push_back({b, static_cast<bool>(invMask & 2)});
          candidates.push_back(std::move(candidate));
        }
  }

  if (policy.allowXor) {
    for (unsigned a = 1; a + 1 < availableSources; ++a)
      for (unsigned b = a + 1; b < availableSources; ++b) {
        ExactCandidate candidate;
        candidate.kind = ExactNodeKind::Xor2;
        candidate.fanins.push_back({a, false});
        candidate.fanins.push_back({b, false});
        candidates.push_back(std::move(candidate));
      }
  }

  if (policy.allowDot) {
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

  llvm::sort(candidates, isCandidateLess);
}

static void addConditionedSemantics(
    IncrementalSATSolver &solver, int selector, int outLit,
    const ExactCandidate &candidate, unsigned minterm,
    llvm::function_ref<int(unsigned source, unsigned minterm, bool inverted)>
        getSourceLiteral) {
  auto addConditionedClause = [&](std::initializer_list<int> lits) {
    SmallVector<int, 8> clause;
    clause.reserve(lits.size() + 1);
    clause.push_back(-selector);
    clause.append(lits.begin(), lits.end());
    solver.addClause(clause);
  };

  switch (candidate.kind) {
  case ExactNodeKind::And2: {
    int aLit = getSourceLiteral(candidate.fanins[0].source, minterm,
                                candidate.fanins[0].inverted);
    int bLit = getSourceLiteral(candidate.fanins[1].source, minterm,
                                candidate.fanins[1].inverted);
    addConditionedClause({-outLit, aLit});
    addConditionedClause({-outLit, bLit});
    addConditionedClause({outLit, -aLit, -bLit});
    return;
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
    return;
  }
  case ExactNodeKind::Dot3: {
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
    return;
  }
  }
  llvm_unreachable("unknown exact synthesis node kind");
}

static Value materializeInverter(OpBuilder &builder, Location loc,
                                 Value input) {
  return aig::AndInverterOp::create(builder, loc, input, true);
}

static Value materializeNetwork(OpBuilder &builder, Location loc,
                                ArrayRef<Value> inputs,
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

  SmallVector<Value, 4> stepValues;
  stepValues.reserve(network.steps.size());
  for (const auto &step : network.steps) {
    switch (step.kind) {
    case ExactNodeKind::And2:
      stepValues.push_back(aig::AndInverterOp::create(
          builder, loc, getRawSignal(step.fanins[0], stepValues),
          getRawSignal(step.fanins[1], stepValues), step.fanins[0].inverted,
          step.fanins[1].inverted));
      break;
    case ExactNodeKind::Xor2:
      stepValues.push_back(XorInverterOp::create(
          builder, loc, getRawSignal(step.fanins[0], stepValues),
          getRawSignal(step.fanins[1], stepValues), step.fanins[0].inverted,
          step.fanins[1].inverted));
      break;
    case ExactNodeKind::Dot3:
      stepValues.push_back(DotOp::create(
          builder, loc, getRawSignal(step.fanins[0], stepValues),
          getRawSignal(step.fanins[1], stepValues),
          getRawSignal(step.fanins[2], stepValues), step.fanins[0].inverted,
          step.fanins[1].inverted, step.fanins[2].inverted));
      break;
    }
  }

  if (network.output.source == 0)
    return getConstant(network.output.inverted);

  Value result = getRawSignal(network.output, stepValues);
  if (!network.output.inverted)
    return result;
  return materializeInverter(builder, loc, result);
}

class GenericExactSATProblem {
public:
  GenericExactSATProblem(const ExactSynthesisPolicy &policy,
                         IncrementalSATSolver &solver, unsigned numInputs,
                         const APInt &target, unsigned numSteps)
      : policy(policy), solver(solver), numInputs(numInputs), target(target),
        numSteps(numSteps), numMinterms(1u << numInputs),
        totalSources(1 + numInputs + numSteps) {}

  std::optional<ExactNetwork> solve() {
    if (!buildEncoding())
      return std::nullopt;
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

  bool buildEncoding() {
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
      enumerateCandidates(policy, availableSources, stepCandidates[step]);
      if (stepCandidates[step].empty())
        return false;

      auto &selectionVars = stepSelectionVars[step];
      selectionVars.reserve(stepCandidates[step].size());
      for (size_t i = 0, e = stepCandidates[step].size(); i != e; ++i)
        selectionVars.push_back(newVar());
      addExactlyOne(selectionVars);
    }

    addAdjacentStepSymmetryBreakingConstraints();
    addCandidateSemanticsConstraints();
    addUseAllStepsConstraints();

    unsigned rootSource = totalSources - 1;
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({target[minterm]
                            ? getSourceValueVar(rootSource, minterm)
                            : -getSourceValueVar(rootSource, minterm)});
    return true;
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
      if (allowedNextSelections.empty())
        continue;

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
           llvm::enumerate(stepCandidates[step]))
        for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
          addConditionedSemantics(
              solver, selectionVars[candidateIndex],
              getSourceValueVar(outSource, minterm), candidate, minterm,
              [&](unsigned source, unsigned currentMinterm, bool inverted) {
                return getSourceLiteral(source, currentMinterm, inverted);
              });
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
        network.steps.push_back(candidates[i]);
        break;
      }
    }
    network.output = {1 + numInputs + numSteps - 1, false};
    return network;
  }

  const ExactSynthesisPolicy &policy;
  IncrementalSATSolver &solver;
  unsigned numInputs;
  APInt target;
  unsigned numSteps;
  unsigned numMinterms;
  unsigned totalSources;
  int nextVar = 0;
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  SmallVector<SmallVector<ExactCandidate, 64>, 8> stepCandidates;
  SmallVector<SmallVector<int, 64>, 8> stepSelectionVars;
};

static FailureOr<Value>
exactSynthesizeAreaMinimized(OpBuilder &builder, Location loc, APInt truthTable,
                             ArrayRef<Value> operands,
                             const ExactSynthesisPolicy &policy) {
  unsigned numInputs = operands.size();
  APInt normalizedTruthTable = truthTable;
  bool invertOutput = false;
  if (normalizedTruthTable[0]) {
    normalizedTruthTable.flipAllBits();
    invertOutput = true;
  }

  auto network = synthesizeDirect(numInputs, normalizedTruthTable);
  if (network) {
    if (invertOutput)
      network->output.inverted = !network->output.inverted;
    return materializeNetwork(builder, loc, operands, *network);
  }

  if (!hasEnabledConstructs(policy))
    return failure();

  for (unsigned area = 1; area <= kMaxExactSearchArea; ++area) {
    auto solver = createIncrementalSATSolver("auto");
    if (!solver)
      return failure();
    GenericExactSATProblem problem(policy, *solver, numInputs,
                                   normalizedTruthTable, area);
    auto solved = problem.solve();
    if (!solved)
      continue;

    if (invertOutput)
      solved->output.inverted = !solved->output.inverted;
    return materializeNetwork(builder, loc, operands, *solved);
  }
  return failure();
}

static APInt convertLookupTableToAPInt(ArrayRef<bool> table) {
  APInt truthTable(table.size(), 0);
  for (size_t index = 0, e = table.size(); index != e; ++index)
    truthTable.setBitVal(index, table[index]);
  return truthTable;
}

struct ExactSynthesisPattern : public OpRewritePattern<comb::TruthTableOp> {
  ExactSynthesisPattern(MLIRContext *context, ExactSynthesisPolicy policy)
      : OpRewritePattern(context), policy(policy) {}

  LogicalResult matchAndRewrite(comb::TruthTableOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() > kMaxExactSynthesisInputs)
      return failure();

    SmallVector<Value> operands;
    operands.reserve(op.getInputs().size());
    for (Value operand : llvm::reverse(op.getInputs()))
      operands.push_back(operand);
    auto result = exactSynthesizeAreaMinimized(
        rewriter, op.getLoc(), convertLookupTableToAPInt(op.getLookupTable()),
        operands, policy);
    if (failed(result))
      return failure();

    replaceOpAndCopyNamehint(rewriter, op, *result);
    return success();
  }

private:
  ExactSynthesisPolicy policy;
};

struct ExactSynthesisPass
    : public circt::synth::impl::ExactSynthesisBase<ExactSynthesisPass> {
  using ExactSynthesisBase::ExactSynthesisBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExactSynthesisPattern>(
        &getContext(), ExactSynthesisPolicy{allowAnd, allowXor, allowDot});
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

FailureOr<Value>
circt::synth::ExactSynthesis(OpBuilder &builder, APInt truthTable,
                             ArrayRef<Value> operands,
                             const ExactSynthesisPolicy &policy) {
  if (operands.size() > kMaxExactSynthesisInputs)
    return failure();
  if (truthTable.getBitWidth() != (1u << operands.size()))
    return failure();
  if (llvm::any_of(operands, [](Value operand) {
        return !operand.getType().isInteger(1);
      }))
    return failure();

  Location loc = builder.getUnknownLoc();
  return exactSynthesizeAreaMinimized(builder, loc, truthTable, operands,
                                      policy);
}
