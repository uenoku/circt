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
#include <array>
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

//===----------------------------------------------------------------------===//
// Exact network model
//===----------------------------------------------------------------------===//

class ExactNodeInfo;

struct ExactSignalRef {
  // `source == 0` denotes the constant-false source. All other sources are
  // numbered as 1-based primary inputs followed by synthesized steps.
  unsigned source = 0;
  bool inverted = false;
};

struct ExactNetworkStep {
  const ExactNodeInfo *info = nullptr;
  SmallVector<ExactSignalRef, 3> fanins;
};

struct ExactNetwork {
  unsigned numInputs = 0;
  SmallVector<ExactNetworkStep, 4> steps;
  ExactSignalRef output;
};

using ExactCandidate = ExactNetworkStep;

/// Declarative description of a node kind that the exact search may place.
class ExactNodeInfo {
public:
  ExactNodeInfo(unsigned arity, bool commutative, uint8_t truthTable)
      : arity(arity), commutative(commutative), truthTable(truthTable) {}
  virtual ~ExactNodeInfo() = default;

  unsigned getArity() const { return arity; }
  bool isCommutative() const { return commutative; }
  uint8_t getTruthTable() const { return truthTable; }

  /// Emit clauses for `selector => outLit == f(fanins...)`.
  ///
  /// By default the SAT semantics are derived directly from the node truth
  /// table. That keeps node definitions compact while still allowing an
  /// override if a future node kind ever needs a specialized encoding.
  virtual void emitConditionedCNF(
      IncrementalSATSolver &solver, int selector, int outLit,
      const ExactCandidate &candidate, unsigned minterm,
      llvm::function_ref<int(unsigned source, unsigned minterm, bool inverted)>
          getSourceLiteral) const {
    for (unsigned assignment = 0, e = 1u << getArity(); assignment != e;
         ++assignment) {
      SmallVector<int, 8> clause;
      clause.reserve(getArity() + 2);
      clause.push_back(-selector);
      for (unsigned operand = 0; operand != getArity(); ++operand) {
        int lit = getSourceLiteral(candidate.fanins[operand].source, minterm,
                                   candidate.fanins[operand].inverted);
        clause.push_back((assignment & (1u << operand)) ? -lit : lit);
      }
      bool value = (truthTable >> assignment) & 1u;
      clause.push_back(value ? outLit : -outLit);
      solver.addClause(clause);
    }
  }
  virtual Value materialize(OpBuilder &builder, Location loc,
                            ArrayRef<Value> operands,
                            ArrayRef<bool> inverted) const = 0;

private:
  unsigned arity;
  bool commutative;
  uint8_t truthTable;
};

class And2NodeInfo final : public ExactNodeInfo {
public:
  And2NodeInfo() : ExactNodeInfo(2, true, 0x8) {}

  Value materialize(OpBuilder &builder, Location loc, ArrayRef<Value> operands,
                    ArrayRef<bool> inverted) const override {
    assert(operands.size() == 2 && "and2 expects two operands");
    assert(inverted.size() == 2 && "and2 expects two inversion flags");
    return aig::AndInverterOp::create(builder, loc, operands[0], operands[1],
                                      inverted[0], inverted[1]);
  }
};

class Xor2NodeInfo final : public ExactNodeInfo {
public:
  Xor2NodeInfo() : ExactNodeInfo(2, true, 0x6) {}

  Value materialize(OpBuilder &builder, Location loc, ArrayRef<Value> operands,
                    ArrayRef<bool> inverted) const override {
    assert(operands.size() == 2 && "xor2 expects two operands");
    assert(inverted.size() == 2 && "xor2 expects two inversion flags");
    return XorInverterOp::create(builder, loc, operands[0], operands[1],
                                 inverted[0], inverted[1]);
  }
};

class Dot3NodeInfo final : public ExactNodeInfo {
public:
  Dot3NodeInfo() : ExactNodeInfo(3, false, 0x1A) {}

  Value materialize(OpBuilder &builder, Location loc, ArrayRef<Value> operands,
                    ArrayRef<bool> inverted) const override {
    assert(operands.size() == 3 && "dot3 expects three operands");
    assert(inverted.size() == 3 && "dot3 expects three inversion flags");
    return DotOp::create(builder, loc, operands[0], operands[1], operands[2],
                         inverted[0], inverted[1], inverted[2]);
  }
};

static ArrayRef<const ExactNodeInfo *> getAllNodeInfos() {
  static const And2NodeInfo and2;
  static const Xor2NodeInfo xor2;
  static const Dot3NodeInfo dot3;
  static const std::array<const ExactNodeInfo *, 3> infos = {&and2, &xor2,
                                                             &dot3};
  return infos;
}

static SmallVector<const ExactNodeInfo *, 3>
getEnabledNodeInfos(const ExactSynthesisPolicy &policy) {
  SmallVector<const ExactNodeInfo *, 3> infos;
  if (policy.allowAnd)
    infos.push_back(getAllNodeInfos()[0]);
  if (policy.allowXor)
    infos.push_back(getAllNodeInfos()[1]);
  if (policy.allowDot)
    infos.push_back(getAllNodeInfos()[2]);
  return infos;
}

static bool hasEnabledConstructs(const ExactSynthesisPolicy &policy) {
  return !getEnabledNodeInfos(policy).empty();
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

//===----------------------------------------------------------------------===//
// Enumeration
//===----------------------------------------------------------------------===//

/// Enumerates all legal candidates for one SAT step.
///
/// Enumeration only depends on structural properties: node arity,
/// commutativity, and per-edge inversion.
class ExactCandidateEnumerator {
public:
  void enumerate(const ExactSynthesisPolicy &policy, unsigned availableSources,
                 SmallVectorImpl<ExactCandidate> &candidates) const {
    candidates.clear();
    for (const auto *info : getEnabledNodeInfos(policy))
      enumerateNodeCandidates(*info, availableSources, candidates);
    llvm::sort(candidates, isCandidateLess);
  }

  static bool isOrderedBefore(const ExactCandidate &lhs,
                              const ExactCandidate &rhs) {
    return isCandidateLess(lhs, rhs);
  }

private:
  static unsigned getInversionMask(const ExactCandidate &candidate) {
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
    if (lhs.info != rhs.info)
      return getNodeOrder(*lhs.info) < getNodeOrder(*rhs.info);
    return getInversionMask(lhs) < getInversionMask(rhs);
  }

  static unsigned getNodeOrder(const ExactNodeInfo &info) {
    for (auto [index, candidateInfo] : llvm::enumerate(getAllNodeInfos()))
      if (candidateInfo == &info)
        return index;
    llvm_unreachable("unknown node info");
  }

  static void enumerateCommutativeOperandSources(
      unsigned availableSources, unsigned arity, unsigned currentArity,
      unsigned nextSource, SmallVectorImpl<unsigned> &sources,
      llvm::function_ref<void(ArrayRef<unsigned>)> emit) {
    if (currentArity == arity) {
      emit(sources);
      return;
    }

    for (unsigned source = nextSource; source < availableSources; ++source) {
      sources.push_back(source);
      enumerateCommutativeOperandSources(
          availableSources, arity, currentArity + 1, source + 1, sources, emit);
      sources.pop_back();
    }
  }

  static void enumerateOrderedOperandSources(
      unsigned availableSources, unsigned arity, unsigned currentArity,
      SmallVectorImpl<unsigned> &sources,
      llvm::function_ref<void(ArrayRef<unsigned>)> emit) {
    if (currentArity == arity) {
      emit(sources);
      return;
    }

    for (unsigned source = 1; source < availableSources; ++source) {
      sources.push_back(source);
      enumerateOrderedOperandSources(availableSources, arity, currentArity + 1,
                                     sources, emit);
      sources.pop_back();
    }
  }

  static void
  enumerateNodeCandidates(const ExactNodeInfo &info, unsigned availableSources,
                          SmallVectorImpl<ExactCandidate> &candidates) {
    SmallVector<unsigned, 3> sources;
    auto emitCandidate = [&](ArrayRef<unsigned> operandSources) {
      for (unsigned invMask = 0, e = 1u << info.getArity(); invMask != e;
           ++invMask) {
        ExactCandidate candidate;
        candidate.info = &info;
        for (auto [index, source] : llvm::enumerate(operandSources))
          candidate.fanins.push_back(
              {source, static_cast<bool>(invMask & (1u << index))});
        candidates.push_back(std::move(candidate));
      }
    };

    if (info.isCommutative()) {
      enumerateCommutativeOperandSources(availableSources, info.getArity(),
                                         /*currentArity=*/0,
                                         /*nextSource=*/1, sources,
                                         emitCandidate);
      return;
    }

    enumerateOrderedOperandSources(availableSources, info.getArity(),
                                   /*currentArity=*/0, sources, emitCandidate);
  }
};

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

static void addConditionedSemantics(
    IncrementalSATSolver &solver, int selector, int outLit,
    const ExactCandidate &candidate, unsigned minterm,
    llvm::function_ref<int(unsigned source, unsigned minterm, bool inverted)>
        getSourceLiteral) {
  assert(candidate.info && "candidate must carry node info");
  candidate.info->emitConditionedCNF(solver, selector, outLit, candidate,
                                     minterm, getSourceLiteral);
}

//===----------------------------------------------------------------------===//
// Materialization
//===----------------------------------------------------------------------===//

/// Lowers a solved exact network back into current Synth IR.
class ExactNetworkMaterializer {
public:
  ExactNetworkMaterializer(OpBuilder &builder, Location loc,
                           ArrayRef<Value> inputs)
      : builder(builder), loc(loc), inputs(inputs) {}

  Value materialize(const ExactNetwork &network) {
    SmallVector<Value, 4> stepValues;
    stepValues.reserve(network.steps.size());
    for (const auto &step : network.steps) {
      assert(step.info && "network step must carry node info");
      const auto &info = *step.info;
      SmallVector<Value, 3> operands;
      SmallVector<bool, 3> inverted;
      operands.reserve(info.getArity());
      inverted.reserve(info.getArity());
      for (const auto &fanin : step.fanins) {
        operands.push_back(getRawSignal(fanin, stepValues));
        inverted.push_back(fanin.inverted);
      }
      stepValues.push_back(info.materialize(builder, loc, operands, inverted));
    }

    if (network.output.source == 0)
      return getConstant(network.output.inverted);

    Value result = getRawSignal(network.output, stepValues);
    if (!network.output.inverted)
      return result;
    return materializeInverter(result);
  }

private:
  Value getConstant(bool value) {
    if (!value) {
      if (!constZero)
        constZero = hw::ConstantOp::create(builder, loc, APInt(1, 0));
      return constZero;
    }
    if (!constOne)
      constOne = hw::ConstantOp::create(builder, loc, APInt(1, 1));
    return constOne;
  }

  Value getRawSignal(ExactSignalRef signal, ArrayRef<Value> stepValues) {
    if (signal.source == 0)
      return getConstant(false);
    if (signal.source <= inputs.size())
      return inputs[signal.source - 1];

    unsigned stepIndex = signal.source - (inputs.size() + 1);
    assert(stepIndex < stepValues.size() && "invalid synthesized step index");
    return stepValues[stepIndex];
  }

  Value materializeInverter(Value input) {
    return aig::AndInverterOp::create(builder, loc, input, true);
  }

  OpBuilder &builder;
  Location loc;
  ArrayRef<Value> inputs;
  Value constZero;
  Value constOne;
};

//===----------------------------------------------------------------------===//
// SAT search
//===----------------------------------------------------------------------===//

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

  /// Build the SAT encoding for a fixed-area exact synthesis problem.
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
      enumerator.enumerate(policy, availableSources, stepCandidates[step]);
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

  /// Enforce a canonical local ordering so equivalent adjacent steps are not
  /// explored in both permutations.
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
        if (!ExactCandidateEnumerator::isOrderedBefore(nextCandidate,
                                                       prevCandidate))
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

  /// Connect each selected candidate to the truth value of the step output at
  /// every minterm.
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

  /// Force every step except the root to feed some later selected step.
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
  ExactCandidateEnumerator enumerator;
};

static FailureOr<Value>
exactSynthesizeAreaMinimized(OpBuilder &builder, Location loc, APInt truthTable,
                             ArrayRef<Value> operands,
                             const ExactSynthesisPolicy &policy) {
  ExactNetworkMaterializer materializer(builder, loc, operands);
  unsigned numInputs = operands.size();
  APInt normalizedTruthTable = truthTable;
  bool invertOutput = false;
  if (normalizedTruthTable[0]) {
    // Normalize to a false-preserving truth table so constant false remains the
    // distinguished zero source in the encoding.
    normalizedTruthTable.flipAllBits();
    invertOutput = true;
  }

  auto network = synthesizeDirect(numInputs, normalizedTruthTable);
  if (network) {
    if (invertOutput)
      network->output.inverted = !network->output.inverted;
    return materializer.materialize(*network);
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
    return materializer.materialize(*solved);
  }
  return failure();
}

static APInt convertLookupTableToAPInt(ArrayRef<bool> table) {
  APInt truthTable(table.size(), 0);
  for (size_t index = 0, e = table.size(); index != e; ++index)
    truthTable.setBitVal(index, table[index]);
  return truthTable;
}

//===----------------------------------------------------------------------===//
// Rewrite plumbing
//===----------------------------------------------------------------------===//

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
