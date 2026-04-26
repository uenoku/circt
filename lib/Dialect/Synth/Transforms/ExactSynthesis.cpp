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

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Naming.h"
#include "circt/Support/SATSolver.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include <array>
#include <optional>
#include <string>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_EXACTSYNTHESIS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

#define DEBUG_TYPE "synth-exact-synthesis"

namespace {

struct ExactSynthesisPolicy {
  struct PrimitiveSpec {
    OperationName opName;
    unsigned arity;

    bool operator==(const PrimitiveSpec &other) const {
      return opName == other.opName && arity == other.arity;
    }
  };

  SmallVector<PrimitiveSpec, 4> allowedPrimitives;
  MLIRContext *context = nullptr;
};

static constexpr unsigned kMaxExactSynthesisInputs = 4;
static constexpr unsigned kMaxExactSearchArea = 32;

static SmallString<32> formatTruthTable(const APInt &truthTable) {
  SmallString<32> text;
  truthTable.toStringUnsigned(text, /*Radix=*/16);
  return text;
}

using BooleanLogicConcept =
    circt::synth::detail::BooleanLogicOpInterfaceInterfaceTraits::Concept;

static void appendPrimitiveSummary(raw_ostream &os,
                                   const ExactSynthesisPolicy &policy) {
  llvm::interleaveComma(policy.allowedPrimitives, os,
                        [&](ExactSynthesisPolicy::PrimitiveSpec spec) {
                          os << spec.opName.getStringRef() << ":" << spec.arity;
                        });
}

static uint8_t deriveNodeTruthTable(const BooleanLogicConcept *iface,
                                    unsigned arity) {
  SmallVector<APInt, 4> inputs;
  inputs.reserve(arity);
  for (unsigned input = 0; input != arity; ++input)
    inputs.push_back(circt::createVarMask(arity, input, true));
  APInt truthTable = iface->evaluateBooleanLogicWithoutInversion(inputs);
  assert(truthTable.getBitWidth() <= 8 &&
         "exact-synthesis nodes expect <= 8 bits");
  return static_cast<uint8_t>(truthTable.getZExtValue());
}

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

  unsigned getInversionMask() const;
  bool operator<(const ExactNetworkStep &rhs) const;
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
  ExactNodeInfo(OperationName opName, unsigned arity, bool commutative,
                uint8_t truthTable, const BooleanLogicConcept *iface)
      : opName(opName), arity(arity), commutative(commutative),
        truthTable(truthTable), iface(iface) {}
  virtual ~ExactNodeInfo() = default;

  OperationName getOperationName() const { return opName; }
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
          getSourceLiteral) const;
  Value materialize(OpBuilder &builder, Location loc, ArrayRef<Value> operands,
                    ArrayRef<bool> inverted) const {
    return iface->materializeBooleanLogic(builder, loc, operands, inverted);
  }

private:
  OperationName opName;
  unsigned arity;
  bool commutative;
  uint8_t truthTable;
  const BooleanLogicConcept *iface;
};

unsigned ExactNetworkStep::getInversionMask() const {
  unsigned mask = 0;
  for (auto [index, fanin] : llvm::enumerate(fanins))
    if (fanin.inverted)
      mask |= 1u << index;
  return mask;
}

bool ExactNetworkStep::operator<(const ExactNetworkStep &rhs) const {
  if (fanins.size() != rhs.fanins.size())
    return fanins.size() < rhs.fanins.size();
  for (size_t i = 0, e = fanins.size(); i != e; ++i) {
    if (fanins[i].source != rhs.fanins[i].source)
      return fanins[i].source < rhs.fanins[i].source;
  }
  if (info != rhs.info)
    return info->getTruthTable() < rhs.info->getTruthTable();
  return getInversionMask() < rhs.getInversionMask();
}

static const ExactNodeInfo *getPrimitiveNodeInfo(OperationName opName,
                                                 unsigned arity) {
  static SmallVector<std::unique_ptr<ExactNodeInfo>, 4> infos;
  for (const auto &info : infos)
    if (info->getArity() == arity && opName == info->getOperationName())
      return info.get();

  auto *iface = opName.getInterface<BooleanLogicOpInterface>();
  assert(iface && "exact-synthesis op must implement BooleanLogicOpInterface");
  bool commutative = iface->areInputsPermutationInvariant();
  infos.push_back(std::make_unique<ExactNodeInfo>(
      opName, arity, commutative, deriveNodeTruthTable(iface, arity), iface));
  return infos.back().get();
}

static SmallVector<const ExactNodeInfo *, 4>
getEnabledNodeInfos(const ExactSynthesisPolicy &policy) {
  SmallVector<const ExactNodeInfo *, 4> infos;
  infos.reserve(policy.allowedPrimitives.size());
  assert(policy.context && "exact-synthesis policy must carry an MLIRContext");
  for (const auto &spec : policy.allowedPrimitives)
    infos.push_back(getPrimitiveNodeInfo(spec.opName, spec.arity));
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
                 SmallVectorImpl<ExactCandidate> &candidates) const;

private:
  static void enumerateCommutativeOperandSources(
      unsigned availableSources, unsigned arity, unsigned currentArity,
      unsigned nextSource, SmallVectorImpl<unsigned> &sources,
      llvm::function_ref<void(ArrayRef<unsigned>)> emit);

  static void enumerateOrderedOperandSources(
      unsigned availableSources, unsigned arity, unsigned currentArity,
      SmallVectorImpl<unsigned> &sources,
      llvm::function_ref<void(ArrayRef<unsigned>)> emit);

  static void
  enumerateNodeCandidates(const ExactNodeInfo &info, unsigned availableSources,
                          SmallVectorImpl<ExactCandidate> &candidates);
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
                           ArrayRef<Value> inputs);

  Value materialize(const ExactNetwork &network);

private:
  Value getConstant(bool value);
  Value getRawSignal(ExactSignalRef signal, ArrayRef<Value> stepValues);
  Value materializeInverter(Value input);

  OpBuilder &builder;
  Location loc;
  ArrayRef<Value> inputs;
  std::array<Value, 2> constValues;
};

//===----------------------------------------------------------------------===//
// SAT search
//===----------------------------------------------------------------------===//

class GenericExactSATProblem {
public:
  GenericExactSATProblem(const ExactSynthesisPolicy &policy,
                         IncrementalSATSolver &solver, unsigned numInputs,
                         const APInt &target, unsigned numSteps);

  std::optional<ExactNetwork> solve();

private:
  int newVar();
  void addExactlyOne(ArrayRef<int> vars);
  /// Return the SAT variable for one source under one concrete input pattern.
  int getSourceValueVar(unsigned source, unsigned minterm) const;
  /// Return that same variable as a literal, optionally negated.
  int getSourceLiteral(unsigned source, unsigned minterm, bool inverted) const;

  /// Build the SAT model for "can this truth table be implemented with exactly
  /// `numSteps` internal nodes?".
  bool buildEncoding();

  /// Say what each step output must be for each candidate and each input
  /// pattern.
  void addCandidateSemanticsConstraints();

  /// Force every step except the root to feed some later selected step.
  void addUseAllStepsConstraints();

  ExactNetwork decodeModel() const;

  const ExactSynthesisPolicy &policy;
  IncrementalSATSolver &solver;
  unsigned numInputs;
  APInt target;
  unsigned numSteps;
  unsigned numMinterms;
  unsigned totalSources;
  int nextVar = 0;
  int rootInvertVar = 0;
  SmallVector<SmallVector<int, 16>, 8> sourceValueVars;
  SmallVector<SmallVector<ExactCandidate, 64>, 8> stepCandidates;
  SmallVector<SmallVector<int, 64>, 8> stepSelectionVars;
  ExactCandidateEnumerator enumerator;
};

void ExactNodeInfo::emitConditionedCNF(
    IncrementalSATSolver &solver, int selector, int outLit,
    const ExactCandidate &candidate, unsigned minterm,
    llvm::function_ref<int(unsigned source, unsigned minterm, bool inverted)>
        getSourceLiteral) const {
  // Encode the selected primitive by enumerating every local input assignment:
  //   selector => ((fanins match assignment) implies outLit = truthTable[row]).
  //
  // The outer SAT problem already fixes one global minterm. This helper only
  // needs to describe how the chosen node transforms its fanins for that row.
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

void ExactCandidateEnumerator::enumerate(
    const ExactSynthesisPolicy &policy, unsigned availableSources,
    SmallVectorImpl<ExactCandidate> &candidates) const {
  candidates.clear();
  for (const auto *info : getEnabledNodeInfos(policy))
    enumerateNodeCandidates(*info, availableSources, candidates);
  llvm::sort(candidates);
  LDBG() << "Enumerated " << candidates.size()
         << " candidates with availableSources=" << availableSources << "\n";
}

void ExactCandidateEnumerator::enumerateCommutativeOperandSources(
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

void ExactCandidateEnumerator::enumerateOrderedOperandSources(
    unsigned availableSources, unsigned arity, unsigned currentArity,
    SmallVectorImpl<unsigned> &sources,
    llvm::function_ref<void(ArrayRef<unsigned>)> emit) {
  if (currentArity == arity) {
    emit(sources);
    return;
  }

  // Ordered nodes such as DOT may reuse sources and may also bind the
  // dedicated constant-false source `0`.
  for (unsigned source = 0; source < availableSources; ++source) {
    sources.push_back(source);
    enumerateOrderedOperandSources(availableSources, arity, currentArity + 1,
                                   sources, emit);
    sources.pop_back();
  }
}

void ExactCandidateEnumerator::enumerateNodeCandidates(
    const ExactNodeInfo &info, unsigned availableSources,
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

  // Structural enumeration is intentionally independent of Boolean semantics.
  // At this stage we only choose which available sources feed the node and
  // which edges are inverted; the node truth table is imposed later in CNF.
  if (info.isCommutative()) {
    // The commutative basis currently consists of AND/XOR. They may use the
    // constant-false source, but we still enumerate their fanins in sorted
    // order so equivalent operand permutations collapse to one candidate.
    enumerateCommutativeOperandSources(availableSources, info.getArity(),
                                       /*currentArity=*/0,
                                       /*nextSource=*/0, sources,
                                       emitCandidate);
    return;
  }

  enumerateOrderedOperandSources(availableSources, info.getArity(),
                                 /*currentArity=*/0, sources, emitCandidate);
}

ExactNetworkMaterializer::ExactNetworkMaterializer(OpBuilder &builder,
                                                   Location loc,
                                                   ArrayRef<Value> inputs)
    : builder(builder), loc(loc), inputs(inputs) {}

Value ExactNetworkMaterializer::materialize(const ExactNetwork &network) {
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

Value ExactNetworkMaterializer::getConstant(bool value) {
  if (constValues[value])
    return constValues[value];
  return constValues[value] =
             hw::ConstantOp::create(builder, loc, APInt(1, value));
}

Value ExactNetworkMaterializer::getRawSignal(ExactSignalRef signal,
                                             ArrayRef<Value> stepValues) {
  if (signal.source == 0)
    return getConstant(false);
  if (signal.source <= inputs.size())
    return inputs[signal.source - 1];

  unsigned stepIndex = signal.source - (inputs.size() + 1);
  assert(stepIndex < stepValues.size() && "invalid synthesized step index");
  return stepValues[stepIndex];
}

Value ExactNetworkMaterializer::materializeInverter(Value input) {
  return aig::AndInverterOp::create(builder, loc, input, true);
}

GenericExactSATProblem::GenericExactSATProblem(
    const ExactSynthesisPolicy &policy, IncrementalSATSolver &solver,
    unsigned numInputs, const APInt &target, unsigned numSteps)
    : policy(policy), solver(solver), numInputs(numInputs), target(target),
      numSteps(numSteps), numMinterms(1u << numInputs),
      totalSources(1 + numInputs + numSteps) {}

std::optional<ExactNetwork> GenericExactSATProblem::solve() {
  LDBG() << "SAT solve start: inputs=" << numInputs << " steps=" << numSteps
         << " minterms=" << numMinterms << " target=0x"
         << formatTruthTable(target) << "\n";
  if (!buildEncoding())
    return std::nullopt;
  auto result = solver.solve();
  LDBG() << "SAT solve result: "
         << (result == IncrementalSATSolver::kSAT     ? "SAT"
             : result == IncrementalSATSolver::kUNSAT ? "UNSAT"
                                                      : "UNKNOWN")
         << "\n";
  if (result != IncrementalSATSolver::kSAT)
    return std::nullopt;
  return decodeModel();
}

int GenericExactSATProblem::newVar() {
  int fresh = ++nextVar;
  solver.reserveVars(fresh);
  return fresh;
}

void GenericExactSATProblem::addExactlyOne(ArrayRef<int> vars) {
  solver.addExactlyOne(vars, [&] { return newVar(); });
}

int GenericExactSATProblem::getSourceValueVar(unsigned source,
                                              unsigned minterm) const {
  return sourceValueVars[source][minterm];
}

int GenericExactSATProblem::getSourceLiteral(unsigned source, unsigned minterm,
                                             bool inverted) const {
  int lit = getSourceValueVar(source, minterm);
  return inverted ? -lit : lit;
}

bool GenericExactSATProblem::buildEncoding() {
  // A minterm is one concrete input pattern, e.g. for 3 inputs:
  //   000, 001, 010, ... , 111.
  //
  // For every source and every minterm, create one SAT variable that means:
  //   "this source has value 1 under this input pattern".
  //
  // Source 0 is the built-in constant 0.
  // Sources [1, numInputs] are the primary inputs.
  // The remaining sources are the synthesized internal steps.
  sourceValueVars.resize(totalSources);
  for (unsigned source = 0; source != totalSources; ++source) {
    sourceValueVars[source].reserve(numMinterms);
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      sourceValueVars[source].push_back(newVar());
  }

  // Source 0 is always false, for every input pattern.
  for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
    solver.addClause({-getSourceValueVar(0, minterm)});

  // Fix the primary input sources to match each input pattern.
  // Example: if minterm = 5 (binary 101), then input 0 is 1, input 1 is 0,
  // and input 2 is 1.
  for (unsigned input = 0; input != numInputs; ++input)
    for (unsigned minterm = 0; minterm != numMinterms; ++minterm)
      solver.addClause({((minterm >> input) & 1)
                            ? getSourceValueVar(1 + input, minterm)
                            : -getSourceValueVar(1 + input, minterm)});

  // For each internal step:
  // 1. list every legal candidate node that could be placed here
  // 2. create one selector variable per candidate
  // 3. force exactly one candidate to be chosen
  stepCandidates.resize(numSteps);
  stepSelectionVars.resize(numSteps);
  for (unsigned step = 0; step != numSteps; ++step) {
    unsigned availableSources = 1 + numInputs + step;
    enumerator.enumerate(policy, availableSources, stepCandidates[step]);
    LDBG() << "  step " << step << ": availableSources=" << availableSources
           << " candidates=" << stepCandidates[step].size() << "\n";
    if (stepCandidates[step].empty())
      return false;

    auto &selectionVars = stepSelectionVars[step];
    selectionVars.reserve(stepCandidates[step].size());
    for (size_t i = 0, e = stepCandidates[step].size(); i != e; ++i)
      selectionVars.push_back(newVar());
    addExactlyOne(selectionVars);
  }

  // Now describe what the chosen candidate means, and forbid dead internal
  // nodes that do not feed anything later.
  // TODO: Add symmetry breaking constraints to reduce the search space.
  addCandidateSemanticsConstraints();
  addUseAllStepsConstraints();

  // The final answer is the value produced by the last internal step, possibly
  // with one final inversion. This single inversion bit is shared by all input
  // patterns, so the solver must choose one global output polarity.
  unsigned rootSource = totalSources - 1;
  rootInvertVar = newVar();
  for (unsigned minterm = 0; minterm != numMinterms; ++minterm) {
    int rootLit = getSourceValueVar(rootSource, minterm);
    if (target[minterm]) {
      // target = root xor rootInvert, with target fixed to 1.
      solver.addClause({rootLit, rootInvertVar});
      solver.addClause({-rootLit, -rootInvertVar});
    } else {
      // target = root xor rootInvert, with target fixed to 0.
      solver.addClause({rootLit, -rootInvertVar});
      solver.addClause({-rootLit, rootInvertVar});
    }
  }
  return true;
}

void GenericExactSATProblem::addCandidateSemanticsConstraints() {
  for (unsigned step = 0; step != numSteps; ++step) {
    unsigned outSource = 1 + numInputs + step;
    const auto &selectionVars = stepSelectionVars[step];
    // For every candidate of this step, and for every input pattern, add the
    // clauses that say:
    //   if this candidate is selected, then the step output must match that
    //   candidate's truth table on that input pattern.
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

void GenericExactSATProblem::addUseAllStepsConstraints() {
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

    // Every non-root step must feed a later selected step. Without this, the
    // area-bounded encoding could satisfy the problem with dead logic that is
    // disconnected from the final output.
    solver.addClause(users);
  }
}

ExactNetwork GenericExactSATProblem::decodeModel() const {
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
  network.output = {1 + numInputs + numSteps - 1,
                    solver.val(rootInvertVar) == rootInvertVar};
  LDBG() << "Decoded network with " << network.steps.size()
         << " steps, rootSource=" << network.output.source
         << " rootInvert=" << network.output.inverted << "\n";
  return network;
}

static FailureOr<Value>
exactSynthesizeAreaMinimized(OpBuilder &builder, Location loc, APInt truthTable,
                             ArrayRef<Value> operands,
                             const ExactSynthesisPolicy &policy) {
  ExactNetworkMaterializer materializer(builder, loc, operands);
  unsigned numInputs = operands.size();
  LDBG() << "Exact synthesis request: inputs=" << numInputs << " truthTable=0x"
         << formatTruthTable(truthTable) << " allowed-primitives=";
  appendPrimitiveSummary(llvm::dbgs(), policy);
  LDBG() << "\n";

  if (!hasEnabledConstructs(policy))
    return failure();

  LDBG() << "Trying direct synthesis for target=0x"
         << formatTruthTable(truthTable) << "\n";
  auto network = synthesizeDirect(numInputs, truthTable);
  if (network) {
    LDBG() << "Using direct synthesis result\n";
    return materializer.materialize(*network);
  }

  for (unsigned area = 1; area <= kMaxExactSearchArea; ++area) {
    LDBG() << "Trying area=" << area << "\n";
    auto solver = createIncrementalSATSolver("auto");
    if (!solver)
      return failure();
    GenericExactSATProblem problem(policy, *solver, numInputs, truthTable,
                                   area);
    auto solved = problem.solve();
    if (!solved) {
      LDBG() << "Area " << area << " has no solution\n";
      continue;
    }
    LDBG() << "Found solution at area=" << area << "\n";
    return materializer.materialize(*solved);
  }
  LDBG() << "No exact solution found up to area limit " << kMaxExactSearchArea
         << "\n";
  return failure();
}

//===----------------------------------------------------------------------===//
// Rewrite Pass
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

    APInt truthTable(op.getLookupTable().size(), 0);
    for (size_t index = 0, e = op.getLookupTable().size(); index != e; ++index)
      truthTable.setBitVal(index, op.getLookupTable()[index]);
    auto result = exactSynthesizeAreaMinimized(rewriter, op.getLoc(),
                                               truthTable, operands, policy);
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

  FailureOr<ExactSynthesisPolicy> parsePolicy(MLIRContext *context) const {
    ExactSynthesisPolicy policy;
    policy.context = context;

    if (allowedOps.empty()) {
      emitError(UnknownLoc::get(context))
          << "synth-exact-synthesis requires at least one "
             "'allowed-ops=name:arity' entry";
      return failure();
    }

    for (const std::string &allowedOp : allowedOps) {
      StringRef spelling = allowedOp;
      auto parts = spelling.split(':');
      StringRef name = parts.first.trim();
      StringRef arityText = parts.second.trim();
      auto registeredInfo = RegisteredOperationName::lookup(name, context);
      if (!registeredInfo) {
        emitError(UnknownLoc::get(context))
            << "unknown allowed exact-synthesis op '" << name << "'";
        return failure();
      }
      auto *iface = registeredInfo->getInterface<BooleanLogicOpInterface>();
      if (!iface) {
        emitError(UnknownLoc::get(context))
            << "op '" << name << "' does not implement BooleanLogicOpInterface";
        return failure();
      }

      auto addUnique = [&](ExactSynthesisPolicy::PrimitiveSpec spec) {
        if (!llvm::is_contained(policy.allowedPrimitives, spec))
          policy.allowedPrimitives.push_back(spec);
      };

      if (arityText.empty()) {
        emitError(UnknownLoc::get(context))
            << "expected explicit arity for '" << name << "', e.g. '" << name
            << ":3'";
        return failure();
      }
      unsigned arity = 0;
      if (arityText.getAsInteger(10, arity)) {
        emitError(UnknownLoc::get(context))
            << "invalid arity in allowed op '" << spelling << "'";
        return failure();
      }
      if (arity < 2 || arity > kMaxExactSynthesisInputs) {
        emitError(UnknownLoc::get(context))
            << "unsupported arity " << arity << " for '" << name << "'";
        return failure();
      }
      if (!iface->isSupportedNumInputs(arity)) {
        emitError(UnknownLoc::get(context))
            << "op '" << name << "' does not support exact-synthesis arity "
            << arity;
        return failure();
      }
      addUnique({OperationName(*registeredInfo), arity});
    }
    return policy;
  }

  LogicalResult initialize(MLIRContext *context) override {
    auto parsedPolicy = parsePolicy(context);
    if (failed(parsedPolicy))
      return failure();
    policy = *parsedPolicy;
    return success();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExactSynthesisPattern>(&getContext(), policy);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }

private:
  ExactSynthesisPolicy policy;
};

} // namespace
