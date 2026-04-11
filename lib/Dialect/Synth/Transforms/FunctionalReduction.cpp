//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements FunctionalReduction (Functionally Reduced And-Inverter
// Graph) optimization. It identifies and merges functionally equivalent nodes
// through simulation-based candidate detection followed by SAT-based
// verification.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/SATSolver.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <memory>
#include <random>

#define DEBUG_TYPE "synth-functional-reduction"

static constexpr llvm::StringLiteral kTestClassAttrName =
    "synth.test.fc_equiv_class";

namespace circt {
namespace synth {
#define GEN_PASS_DEF_FUNCTIONALREDUCTION
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

namespace {
enum class EquivResult { Proved, Disproved, Unknown };

static bool isFunctionalReductionGraphOp(Operation *op) {
  return isa<synth::ChoiceOp, aig::AndInverterOp, comb::AndOp, comb::OrOp,
             comb::XorOp>(op);
}

// Walk SCCs in the local use-def graph used by functional reduction. Nodes are
// logic ops in `block`; edges point from a user to the defining op of each
// operand that is not considered ready. The callback receives the SCC and
// whether it is cyclic.
static void forEachFunctionalReductionSCC(
    Block *block, llvm::function_ref<bool(Value)> isOperandReady,
    llvm::function_ref<void(ArrayRef<Operation *>)> callback) {
  llvm::DenseMap<Operation *, unsigned> indices, lowLinks;
  llvm::SmallPtrSet<Operation *, 32> onStack, eligibleSet;
  SmallVector<Operation *> eligibleOps, stack;
  unsigned nextIndex = 0;

  for (Operation &op : block->getOperations()) {
    if (!isFunctionalReductionGraphOp(&op))
      continue;
    eligibleSet.insert(&op);
    eligibleOps.push_back(&op);
  }

  auto forEachSuccessor = [&](Operation *op,
                              llvm::function_ref<void(Operation *)> fn) {
    for (Value operand : op->getOperands()) {
      Operation *succ = operand.getDefiningOp();
      // Ignore ready values and definitions outside this block-local graph.
      if (!isOperandReady(operand) && succ && eligibleSet.contains(succ))
        fn(succ);
    }
  };

  std::function<void(Operation *)> visit = [&](Operation *op) {
    indices[op] = nextIndex;
    lowLinks[op] = nextIndex;
    ++nextIndex;
    stack.push_back(op);
    onStack.insert(op);

    forEachSuccessor(op, [&](Operation *succ) {
      auto succIndex = indices.find(succ);
      if (succIndex == indices.end()) {
        visit(succ);
        if (lowLinks[succ] < lowLinks[op])
          lowLinks[op] = lowLinks[succ];
        return;
      }
      if (onStack.contains(succ) && succIndex->second < lowLinks[op])
        lowLinks[op] = succIndex->second;
    });

    if (lowLinks[op] != indices[op])
      return;

    SmallVector<Operation *> scc;
    Operation *popped;
    do {
      popped = stack.pop_back_val();
      onStack.erase(popped);
      scc.push_back(popped);
    } while (popped != op);

    callback(scc);
  };

  for (Operation *op : eligibleOps) {
    if (indices.contains(op))
      continue;
    visit(op);
  }
}

std::unique_ptr<IncrementalSATSolver>
createFunctionalReductionSATSolver(llvm::StringRef backend) {
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

class FunctionalReductionSATBuilder {
public:
  FunctionalReductionSATBuilder(IncrementalSATSolver &solver,
                                llvm::DenseMap<Value, int> &satVars,
                                llvm::DenseSet<Value> &encodedValues,
                                int &nextFreshVar);

  EquivResult verify(Value lhs, Value rhs);

private:
  int getOrCreateVar(Value value);
  // Create a fresh SAT variable for an intermediate Boolean subexpression that
  // does not correspond to an MLIR value.
  int createAuxVar();
  int getLiteral(Value value, bool inverted = false);
  void addAndClauses(int outVar, llvm::ArrayRef<int> inputLits);
  void addOrClauses(int outVar, llvm::ArrayRef<int> inputLits);
  void addXorClauses(int outVar, int lhsLit, int rhsLit);
  void addParityClauses(int outVar, llvm::ArrayRef<int> inputLits);
  void encodeValue(Value value);

  IncrementalSATSolver &solver;
  llvm::DenseMap<Value, int> &satVars;
  llvm::DenseSet<Value> &encodedValues;
  int &nextFreshVar;
};

static bool isFunctionalReductionSimulatableOp(Operation *op) {
  return isa<aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp>(op);
}

EquivResult FunctionalReductionSATBuilder::verify(Value lhs, Value rhs) {
  encodeValue(lhs);
  encodeValue(rhs);

  int lhsVar = getOrCreateVar(lhs);
  int rhsVar = getOrCreateVar(rhs);

  // Check the two halves of the XOR miter separately. If either assignment is
  // satisfiable, the solver found a distinguishing input pattern.
  solver.assume(lhsVar);
  solver.assume(-rhsVar);
  auto result = solver.solve();
  if (result == IncrementalSATSolver::kSAT)
    return EquivResult::Disproved;
  if (result != IncrementalSATSolver::kUNSAT)
    return EquivResult::Unknown;

  solver.assume(-lhsVar);
  solver.assume(rhsVar);
  result = solver.solve();
  if (result == IncrementalSATSolver::kSAT)
    return EquivResult::Disproved;
  if (result != IncrementalSATSolver::kUNSAT)
    return EquivResult::Unknown;

  return EquivResult::Proved;
}

int FunctionalReductionSATBuilder::getOrCreateVar(Value value) {
  auto it = satVars.find(value);
  assert(it != satVars.end() && "SAT variable must be preallocated");
  return it->second;
}

int FunctionalReductionSATBuilder::createAuxVar() {
  int freshVar = ++nextFreshVar;
  solver.reserveVars(freshVar);
  return freshVar;
}

int FunctionalReductionSATBuilder::getLiteral(Value value, bool inverted) {
  int lit = getOrCreateVar(value);
  return inverted ? -lit : lit;
}

void FunctionalReductionSATBuilder::addAndClauses(
    int outVar, llvm::ArrayRef<int> inputLits) {
  // Tseitin encoding (https://en.wikipedia.org/wiki/Tseytin_transformation)
  // for `outVar <=> and(inputLits)`. This keeps the CNF linear in the gate size
  // while preserving satisfiability.
  for (int lit : inputLits)
    solver.addClause({-outVar, lit});

  SmallVector<int> clause;
  for (int lit : inputLits)
    clause.push_back(-lit);
  clause.push_back(outVar);
  solver.addClause(clause);
}

void FunctionalReductionSATBuilder::addOrClauses(
    int outVar, llvm::ArrayRef<int> inputLits) {
  // Encode `outVar <=> or(inputLits)`.
  //
  // `(-lit v outVar)` for each input enforces `lit -> outVar`, i.e. any true
  // input forces the OR result high.
  for (int lit : inputLits)
    solver.addClause({-lit, outVar});

  SmallVector<int> clause;
  clause.reserve(inputLits.size() + 1);
  // `(-outVar v lit0 v lit1 ...)` enforces `outVar -> (lit0 v lit1 ...)`.
  // Together these clauses make `outVar` exactly the OR of the inputs.
  clause.push_back(-outVar);
  clause.append(inputLits.begin(), inputLits.end());
  solver.addClause(clause);
}

void FunctionalReductionSATBuilder::addXorClauses(int outVar, int lhsLit,
                                                  int rhsLit) {
  // Encode `outVar <=> (lhsLit xor rhsLit)` with the four satisfying rows of
  // the 2-input XOR truth table. This is the standard definitional CNF for a
  // binary XOR.
  solver.addClause({-lhsLit, -rhsLit, -outVar});
  solver.addClause({lhsLit, rhsLit, -outVar});
  solver.addClause({lhsLit, -rhsLit, outVar});
  solver.addClause({-lhsLit, rhsLit, outVar});
}

void FunctionalReductionSATBuilder::addParityClauses(
    int outVar, llvm::ArrayRef<int> inputLits) {
  assert(!inputLits.empty() && "parity requires at least one input");
  if (inputLits.size() == 1) {
    solver.addClause({-outVar, inputLits.front()});
    solver.addClause({outVar, -inputLits.front()});
    return;
  }

  int accumulatedLit = inputLits.front();
  // Variadic XOR does not have a compact direct CNF encoding like AND/OR, so
  // encode it as a chain of binary XORs and give each intermediate result its
  // own auxiliary SAT variable.
  for (auto [index, lit] : llvm::enumerate(inputLits.drop_front())) {
    bool isLast = index + 2 == inputLits.size();
    int outLit = isLast ? outVar : createAuxVar();
    addXorClauses(outLit, accumulatedLit, lit);
    accumulatedLit = outLit;
  }
}

void FunctionalReductionSATBuilder::encodeValue(Value value) {
  SmallVector<std::pair<Value, bool>> worklist;
  worklist.push_back({value, false});

  while (!worklist.empty()) {
    auto [current, readyToEncode] = worklist.pop_back_val();
    if (encodedValues.contains(current))
      continue;

    Operation *op = current.getDefiningOp();
    if (!op) {
      encodedValues.insert(current);
      continue;
    }

    APInt constantValue;
    if (matchPattern(current, mlir::m_ConstantInt(&constantValue))) {
      encodedValues.insert(current);
      solver.addClause({constantValue.isZero() ? -getOrCreateVar(current)
                                               : getOrCreateVar(current)});
      continue;
    }

    if (!isFunctionalReductionSimulatableOp(op)) {
      // Unsupported operations remain unconstrained, just like block
      // arguments. Since we only prove equivalence from UNSAT, omitting these
      // clauses may miss a proof but cannot create a false proof.
      encodedValues.insert(current);
      continue;
    }

    if (!readyToEncode) {
      worklist.push_back({current, true});
      for (auto input : op->getOperands()) {
        assert(input.getType().isInteger(1) &&
               "only i1 inputs should be simulated or encoded");
        if (!encodedValues.contains(input))
          worklist.push_back({input, false});
      }
      continue;
    }

    encodedValues.insert(current);
    int outVar = getOrCreateVar(current);

    SmallVector<int> inputLits;
    inputLits.reserve(op->getNumOperands());
    TypeSwitch<Operation *>(op)
        .Case<aig::AndInverterOp>([&](auto andOp) {
          for (auto [input, inverted] :
               llvm::zip(andOp.getInputs(), andOp.getInverted()))
            inputLits.push_back(getLiteral(input, inverted));
          addAndClauses(outVar, inputLits);
        })
        .Case<comb::AndOp>([&](auto andOp) {
          for (auto input : andOp.getInputs())
            inputLits.push_back(getLiteral(input));
          addAndClauses(outVar, inputLits);
        })
        .Case<comb::OrOp>([&](auto orOp) {
          for (auto input : orOp.getInputs())
            inputLits.push_back(getLiteral(input));
          addOrClauses(outVar, inputLits);
        })
        .Case<comb::XorOp>([&](auto xorOp) {
          for (auto input : xorOp.getInputs())
            inputLits.push_back(getLiteral(input));
          addParityClauses(outVar, inputLits);
        })
        .Default(
            [](Operation *) { llvm_unreachable("unexpected supported op"); });
  }
}

//===----------------------------------------------------------------------===//
// Core Functional Reduction Implementation
//===----------------------------------------------------------------------===//

class FunctionalReductionSolver {
public:
  FunctionalReductionSolver(hw::HWModuleOp module, unsigned numPatterns,
                            unsigned seed, bool testTransformation,
                            std::unique_ptr<IncrementalSATSolver> satSolver)
      : module(module), numPatterns(numPatterns), seed(seed),
        testTransformation(testTransformation),
        satSolver(std::move(satSolver)) {}

  ~FunctionalReductionSolver() = default;

  /// Run the Functional Reduction algorithm and return statistics.
  struct Stats {
    unsigned numEquivClasses = 0;
    unsigned numProvedEquiv = 0;
    unsigned numDisprovedEquiv = 0;
    unsigned numUnknown = 0;
    unsigned numMergedNodes = 0;
  };
  mlir::FailureOr<Stats> run();

private:
  // Phase 1: Collect i1 values and run simulation
  void collectValues();
  void runSimulation();
  llvm::APInt simulateValue(Value v);

  // Phase 2: Build equivalence classes from simulation
  void buildEquivalenceClasses();

  // Phase 3: SAT-based verification with per-class solver
  void verifyCandidates();
  void initializeSATState();

  // Phase 4: Merge equivalent nodes
  void mergeEquivalentNodes();
  bool pruneChoiceCycles();

  // Test transformation helpers.
  static Attribute getTestEquivClass(Value value);
  static bool matchesTestEquivClass(Value lhs, Value rhs);
  EquivResult verifyEquivalence(Value lhs, Value rhs);

  // Module being processed
  hw::HWModuleOp module;

  // Configuration
  unsigned numPatterns;
  unsigned seed;
  bool testTransformation;

  // Primary inputs (block arguments or results of unknown operations treated as
  // inputs)
  SmallVector<Value> primaryInputs;

  // All i1 values in topological order
  SmallVector<Value> allValues;

  // Simulation signatures: value -> APInt simulation result
  llvm::DenseMap<Value, llvm::APInt> simSignatures;

  // Equivalence candidates: groups of values with identical simulation
  // signatures
  SmallVector<SmallVector<Value>> equivCandidates;

  // Proven equivalences: representative -> proven equivalent members.
  llvm::MapVector<Value, SmallVector<Value>> provenEquivalences;

  std::unique_ptr<IncrementalSATSolver> satSolver;
  std::unique_ptr<FunctionalReductionSATBuilder> satBuilder;
  llvm::DenseMap<Value, int> satVars;
  llvm::DenseSet<Value> encodedValues;
  // Monotonic counter for auxiliary SAT variables introduced by definitional
  // CNF encodings, currently used for variadic XOR.
  int nextFreshVar = 0;
  Stats stats;
};

FunctionalReductionSATBuilder::FunctionalReductionSATBuilder(
    IncrementalSATSolver &solver, llvm::DenseMap<Value, int> &satVars,
    llvm::DenseSet<Value> &encodedValues, int &nextFreshVar)
    : solver(solver), satVars(satVars), encodedValues(encodedValues),
      nextFreshVar(nextFreshVar) {}

Attribute FunctionalReductionSolver::getTestEquivClass(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return {};
  return op->getAttr(kTestClassAttrName);
}

bool FunctionalReductionSolver::matchesTestEquivClass(Value lhs, Value rhs) {
  Attribute lhsClass = getTestEquivClass(lhs);
  Attribute rhsClass = getTestEquivClass(rhs);
  return lhsClass && rhsClass && lhsClass == rhsClass;
}

EquivResult FunctionalReductionSolver::verifyEquivalence(Value lhs, Value rhs) {
  if (testTransformation) {
    if (matchesTestEquivClass(lhs, rhs))
      return EquivResult::Proved;
    return EquivResult::Unknown;
  }
  assert(satBuilder && "SAT builder must be initialized before verification");
  // SAT-based equivalence checking builds a miter for the two candidate nodes
  // and proves that no input assignment can make them differ.
  return satBuilder->verify(lhs, rhs);
}

void FunctionalReductionSolver::initializeSATState() {
  assert(satSolver && "SAT solver must be initialized before SAT state setup");

  satVars.clear();
  encodedValues.clear();
  satVars.reserve(allValues.size());
  for (auto [index, value] : llvm::enumerate(allValues))
    satVars[value] = index + 1;
  nextFreshVar = allValues.size();
  satSolver->reserveVars(allValues.size());

  satBuilder = std::make_unique<FunctionalReductionSATBuilder>(
      *satSolver, satVars, encodedValues, nextFreshVar);
}

//===----------------------------------------------------------------------===//
// Phase 1: Collect values and run simulation
//===----------------------------------------------------------------------===//

void FunctionalReductionSolver::collectValues() {
  // Collect block arguments (primary inputs) that are i1
  for (auto arg : module.getBodyBlock()->getArguments()) {
    if (arg.getType().isInteger(1)) {
      primaryInputs.push_back(arg);
      allValues.push_back(arg);
    }
  }

  // Walk operations and collect i1 results
  // - AIG operations: add to allValues for simulation
  // - Unknown operations: treat as inputs (assign random patterns)
  module.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!result.getType().isInteger(1))
        continue;

      allValues.push_back(result);
      if (!op->hasTrait<OpTrait::ConstantLike>() &&
          !isFunctionalReductionSimulatableOp(op)) {
        // Unknown operations - treat as primary inputs
        primaryInputs.push_back(result);
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "FunctionalReduction: Collected "
                          << primaryInputs.size()
                          << " primary inputs (including unknown ops) and "
                          << allValues.size() << " total i1 values\n");
}

void FunctionalReductionSolver::runSimulation() {
  // Calculate number of 64-bit words needed for numPatterns bits
  unsigned numWords = numPatterns / 64;

  // Create seeded random number generator for deterministic patterns
  std::mt19937_64 rng(seed);

  for (auto input : primaryInputs) {
    // Generate random words using seeded RNG
    SmallVector<uint64_t> words(numWords);
    for (auto &word : words)
      word = rng();

    // Construct APInt directly from words
    llvm::APInt pattern(numPatterns, words);
    simSignatures[input] = pattern;
  }

  // Propagate simulation through the circuit in topological order
  for (auto value : allValues) {
    if (simSignatures.count(value))
      continue; // Already computed (primary input)

    simSignatures[value] = simulateValue(value);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "FunctionalReduction: Simulation complete with "
                 << numPatterns << " patterns\n";
  });
}

llvm::APInt FunctionalReductionSolver::simulateValue(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return simSignatures.at(v);
  return llvm::TypeSwitch<Operation *, llvm::APInt>(op)
      .Case<aig::AndInverterOp>([&](auto op) {
        SmallVector<llvm::APInt> inputSigs;
        for (auto input : op.getInputs())
          inputSigs.push_back(simSignatures.at(input));
        return op.evaluate(inputSigs);
      })
      .Case<comb::AndOp>([&](auto op) {
        APInt result = APInt::getAllOnes(numPatterns);
        for (auto input : op.getInputs())
          result &= simSignatures.at(input);
        return result;
      })
      .Case<comb::OrOp>([&](auto op) {
        APInt result = APInt::getZero(numPatterns);
        for (auto input : op.getInputs())
          result |= simSignatures.at(input);
        return result;
      })
      .Case<comb::XorOp>([&](auto op) {
        APInt result = APInt::getZero(numPatterns);
        for (auto input : op.getInputs())
          result ^= simSignatures.at(input);
        return result;
      })
      .Case([&](hw::ConstantOp op) {
        return op.getValue().isZero() ? APInt::getZero(numPatterns)
                                      : APInt::getAllOnes(numPatterns);
      })
      .Default([&](Operation *) {
        // Unknown operation - treat as input (already assigned a random
        // pattern)
        return simSignatures.at(v);
      });
}

//===----------------------------------------------------------------------===//
// Phase 2: Build equivalence classes from simulation
//===----------------------------------------------------------------------===//

void FunctionalReductionSolver::buildEquivalenceClasses() {
  // Map from signature to list of values
  llvm::MapVector<llvm::APInt, SmallVector<Value>> sigGroups;

  for (auto value : allValues)
    sigGroups[simSignatures.at(value)].push_back(value);

  // Build equivalence candidates for groups with >1 member.
  for (auto &[hash, members] : sigGroups) {
    if (members.size() <= 1)
      continue;
    equivCandidates.push_back(std::move(members));
  }
  stats.numEquivClasses = equivCandidates.size();

  LLVM_DEBUG(llvm::dbgs() << "FunctionalReduction: Built "
                          << equivCandidates.size()
                          << " equivalence candidates\n");
}

//===----------------------------------------------------------------------===//
// Phase 3: SAT-based verification with per-class solvers
//
// For each equivalence class candidates, verify each member against the
// representative using a SAT solver.
//===----------------------------------------------------------------------===//

void FunctionalReductionSolver::verifyCandidates() {
  LLVM_DEBUG(
      llvm::dbgs() << "FunctionalReduction: Starting SAT verification with "
                   << equivCandidates.size() << " equivalence classes\n");

  for (auto &members : equivCandidates) {
    if (members.empty())
      continue;
    auto representative = members.front();
    auto &provenMembers = provenEquivalences[representative];
    // Representative is the canonical node for this class.
    for (auto member : llvm::ArrayRef<Value>(members).drop_front()) {
      EquivResult result = verifyEquivalence(representative, member);
      if (result == EquivResult::Proved) {
        stats.numProvedEquiv++;
        provenMembers.push_back(member);
      } else if (result == EquivResult::Disproved) {
        stats.numDisprovedEquiv++;
        // TODO: Refine equivalence classes based on counterexamples from SAT
        // solver
      } else {
        stats.numUnknown++;
      }
    }
  }

  LLVM_DEBUG(
      llvm::dbgs() << "FunctionalReduction: SAT verification complete. Proved "
                   << stats.numProvedEquiv << " equivalences\n");
}

//===----------------------------------------------------------------------===//
// Phase 4: Merge equivalent nodes
//===----------------------------------------------------------------------===//

void FunctionalReductionSolver::mergeEquivalentNodes() {
  if (provenEquivalences.empty())
    return;

  mlir::OpBuilder builder(module.getContext());
  // Create all choices first, then rewrite uses. Temporary use-before-def is
  // fine here; we clean up any introduced cycles and re-topologically-sort the
  // block afterward.
  SmallVector<synth::ChoiceOp> choices;
  choices.reserve(provenEquivalences.size());
  for (const auto &[representative, members] : provenEquivalences) {
    if (members.empty())
      continue;
    SmallVector<Value> operands;
    operands.reserve(members.size() + 1);
    operands.push_back(representative);
    operands.append(members);
    builder.setInsertionPointAfterValue(members.back());
    auto choice = synth::ChoiceOp::create(builder, representative.getLoc(),
                                          representative.getType(), operands);
    choices.emplace_back(choice);
    stats.numMergedNodes += members.size() + 1;
  }

  for (auto choice : choices) {
    llvm::SmallPtrSet<Operation *, 8> protectedUsers;
    protectedUsers.insert(choice.getOperation());
    for (Value input : choice.getInputs())
      if (Operation *def = input.getDefiningOp())
        protectedUsers.insert(def);

    for (Value value : choice.getInputs())
      value.replaceUsesWithIf(choice.getResult(), [&](OpOperand &use) {
        return !protectedUsers.contains(use.getOwner());
      });
  }

  int64_t numCycles = 0;
  while (pruneChoiceCycles()) {
    numCycles++;
    LLVM_DEBUG(llvm::dbgs()
               << "FunctionalReduction: Pruned cycles introduced by merging ("
               << numCycles << ")\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "FunctionalReduction: Merged "
                          << stats.numMergedNodes << " nodes\n");
}

bool FunctionalReductionSolver::pruneChoiceCycles() {
  // Drop choice operands that keep the choice in a use-def cycle. This lets the
  // merge step rewrite aggressively and then restores an acyclic graph before
  // the pass returns.
  // Ready operands are the leaves of the SCC graph. They cannot participate in
  // a cycle we need to prune.
  const auto isOperandReady = [](Value value) -> bool {
    Operation *op = value.getDefiningOp();
    // Block arguments are always ready.
    if (!op)
      return true;
    return !isFunctionalReductionGraphOp(op);
  };

  SmallVector<std::pair<synth::ChoiceOp, SmallVector<Value>>> choiceRepairs;

  forEachFunctionalReductionSCC(
      module.getBodyBlock(), isOperandReady, [&](ArrayRef<Operation *> scc) {
        // If the SCC has only one node, it cannot have a cycle and we can skip
        // the repair logic.
        if (scc.size() <= 1)
          return;

        llvm::SmallPtrSet<Operation *, 8> sccOps;
        llvm::SmallVector<ChoiceOp> sccChoices;
        for (Operation *op : scc) {
          sccOps.insert(op);
          if (auto choice = dyn_cast<synth::ChoiceOp>(op))
            sccChoices.push_back(choice);
        }

        for (ChoiceOp choice : sccChoices) {
          SmallVector<Value> newOperands;
          newOperands.reserve(choice.getNumOperands());
          for (Value operand : choice.getInputs()) {
            Operation *def = operand.getDefiningOp();
            if (!def || !sccOps.contains(def))
              newOperands.push_back(operand);
          }

          // If all operands are still inside the SCC, this choice has no
          // acyclic replacement in the current SCC snapshot. Repairing another
          // choice in the same SCC may expose one, so leave it for the next
          // prune iteration instead of erasing all alternatives.
          if (newOperands.empty() ||
              newOperands.size() == choice.getNumOperands())
            continue;

          choiceRepairs.push_back({choice, std::move(newOperands)});
        }
      });

  mlir::OpBuilder builder(module.getContext());
  // The SCC walk visits definition SCCs before user SCCs in this graph (edges
  // go from users to defining operations). Apply repairs in reverse so a
  // replacement is materialized before any choice it depends on is erased.
  for (auto &[choice, newOperands] : llvm::reverse(choiceRepairs)) {
    builder.setInsertionPoint(choice);
    Value replacement = builder.createOrFold<ChoiceOp>(
        choice.getLoc(), choice.getType(), newOperands);
    choice.replaceAllUsesWith(replacement);
    choice.erase();
  }

  return !choiceRepairs.empty();
}

//===----------------------------------------------------------------------===//
// Main Functional Reduction algorithm
//===----------------------------------------------------------------------===//

mlir::FailureOr<FunctionalReductionSolver::Stats>
FunctionalReductionSolver::run() {
  LLVM_DEBUG(
      llvm::dbgs() << "FunctionalReduction: Starting functional reduction with "
                   << numPatterns << " simulation patterns\n");

  if (!testTransformation && !satSolver) {
    module->emitError()
        << "FunctionalReduction requires a SAT solver, but none is "
           "available in this build";
    return failure();
  }

  // Topologically sort the values
  if (failed(circt::synth::topologicallySortLogicNetwork(module))) {
    module->emitError()
        << "FunctionalReduction: Failed to topologically sort logic network";
    return failure();
  }

  // Phase 1: Collect values and run simulation
  collectValues();
  if (allValues.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "FunctionalReduction: No i1 values to process\n");
    return stats;
  }

  runSimulation();

  // Phase 2: Build equivalence classes
  buildEquivalenceClasses();
  if (equivCandidates.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "FunctionalReduction: No equivalence candidates found\n");
    return stats;
  }

  // Phase 3: SAT-based verification
  if (!testTransformation)
    initializeSATState();
  verifyCandidates();

  // Phase 4: Merge equivalent nodes
  mergeEquivalentNodes();
  if (failed(circt::synth::topologicallySortLogicNetwork(module))) {
    module->emitError()
        << "FunctionalReduction: Failed to restore topological order after "
           "rewriting choices";
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "FunctionalReduction: Complete. Stats:\n"
                          << "  Equivalence classes: " << stats.numEquivClasses
                          << "\n"
                          << "  Proved: " << stats.numProvedEquiv << "\n"
                          << "  Disproved: " << stats.numDisprovedEquiv << "\n"
                          << "  Unknown (limit): " << stats.numUnknown << "\n"
                          << "  Merged: " << stats.numMergedNodes << "\n");

  return stats;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct FunctionalReductionPass
    : public circt::synth::impl::FunctionalReductionBase<
          FunctionalReductionPass> {
  using FunctionalReductionBase::FunctionalReductionBase;
  void updateStats(const FunctionalReductionSolver::Stats &stats) {
    numEquivClasses += stats.numEquivClasses;
    numProvedEquiv += stats.numProvedEquiv;
    numDisprovedEquiv += stats.numDisprovedEquiv;
    numUnknown += stats.numUnknown;
    numMergedNodes += stats.numMergedNodes;
  }

  void runOnOperation() override {
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Running FunctionalReduction pass on "
                            << module.getName() << "\n");

    if (numRandomPatterns == 0 || (numRandomPatterns & 63U) != 0) {
      module.emitError()
          << "'num-random-patterns' must be a positive multiple of 64";
      return signalPassFailure();
    }
    if (conflictLimit < -1) {
      module.emitError()
          << "'conflict-limit' must be greater than or equal to -1";
      return signalPassFailure();
    }

    std::unique_ptr<IncrementalSATSolver> satSolver;
    if (!testTransformation) {
      satSolver = createFunctionalReductionSATSolver(this->satSolver);
      if (!satSolver) {
        module.emitError() << "unsupported or unavailable SAT solver '"
                           << this->satSolver
                           << "' (expected auto, z3, or cadical)";
        return signalPassFailure();
      }
      satSolver->setConflictLimit(static_cast<int>(conflictLimit));
    }

    FunctionalReductionSolver fcSolver(module, numRandomPatterns, seed,
                                       testTransformation,
                                       std::move(satSolver));
    auto stats = fcSolver.run();
    if (failed(stats))
      return signalPassFailure();
    updateStats(*stats);
    if (stats->numMergedNodes == 0)
      markAllAnalysesPreserved();
  }
};

} // namespace
