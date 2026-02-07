//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements FRAIG (Functionally Reduced And-Inverter Graph)
// optimization using SMT solver with incremental solving. It identifies and
// merges functionally equivalent nodes through simulation-based candidate
// detection followed by SAT-based verification.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SMTAPI.h"
#include <random>

#define DEBUG_TYPE "synth-functional-reduction"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_FUNCTIONALREDUCTION
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

namespace {

//===----------------------------------------------------------------------===//
// FRAIG Solver - Core FRAIG implementation using APInt for simulation
//===----------------------------------------------------------------------===//

class FRAIGSolver {
public:
  FRAIGSolver(hw::HWModuleOp module, unsigned numPatterns,
              unsigned conflictLimit, bool enableFeedback)
      : module(module), numPatterns(numPatterns), conflictLimit(conflictLimit),
        enableFeedback(enableFeedback) {
    solver = llvm::CreateZ3Solver();
  }

  /// Returns true if the solver is valid (Z3 available).
  bool isValid() const { return solver != nullptr; }

  /// Run the FRAIG algorithm and return statistics.
  struct Stats {
    unsigned numEquivClasses = 0;
    unsigned numSATCalls = 0;
    unsigned numProvedEquiv = 0;
    unsigned numDisprovedEquiv = 0;
    unsigned numMergedNodes = 0;
  };
  Stats run();

private:
  // Phase 1: Collect i1 values and run simulation
  void collectValues();
  void runSimulation();
  llvm::APInt simulateValue(Value v);

  // Phase 2: Build equivalence classes from simulation
  // We track (value, isInverted) pairs - two values are equivalent if their
  // signatures match, or complements if one is the bitwise NOT of the other.
  // LLVM's EquivalenceClasses doesn't support the inversion flag, so we use
  // a custom structure that groups by canonical signature.
  void buildEquivalenceClasses();

  // Phase 3: SAT-based verification with incremental solving
  void verifyCandidates();
  llvm::SMTExprRef buildSMTExpr(Value v);
  llvm::SMTExprRef getOrCreateInputSymbol(Value v);

  // Phase 4: Merge equivalent nodes
  void mergeEquivalentNodes();

  // Helper: Compute majority of 3 APInt values (bitwise)
  static llvm::APInt majority3(const llvm::APInt &a, const llvm::APInt &b,
                               const llvm::APInt &c) {
    return (a & b) | (b & c) | (a & c);
  }

  // Module being processed
  hw::HWModuleOp module;

  // Configuration
  unsigned numPatterns;
  [[maybe_unused]] unsigned conflictLimit; // TODO: Use for SAT resource limits
  [[maybe_unused]] bool enableFeedback;    // TODO: Implement CEX feedback

  // SMT solver instance
  llvm::SMTSolverRef solver;

  // All i1 values in topological order (inputs first, then operations)
  SmallVector<Value> allValues;

  // Primary inputs (block arguments)
  SmallVector<Value> primaryInputs;

  // Simulation signatures: value -> APInt simulation result
  // Each bit position represents one simulation pattern
  llvm::DenseMap<Value, llvm::APInt> simSignatures;

  // Input patterns for simulation (random + dynamic from CEX)
  llvm::DenseMap<Value, llvm::APInt> inputPatterns;

  // Equivalence class: representative + members with inversion flag
  // We can't use LLVM's EquivalenceClasses directly because we need to track
  // whether a member is equivalent or complementary to the representative.
  struct EquivClass {
    Value representative;
    // Members: (value, isComplement) - isComplement means value == ~rep
    SmallVector<std::pair<Value, bool>> members;
  };
  SmallVector<EquivClass> equivClasses;

  // Proven equivalences: value -> (representative, isComplement)
  llvm::DenseMap<Value, std::pair<Value, bool>> provenEquiv;

  // SMT expression cache
  llvm::DenseMap<Value, llvm::SMTExprRef> exprCache;

  // Input symbols for SMT (for CEX extraction)
  llvm::DenseMap<Value, llvm::SMTExprRef> inputSymbols;

  // Statistics
  Stats stats;
};

//===----------------------------------------------------------------------===//
// Phase 1: Collect values and run simulation
//===----------------------------------------------------------------------===//

void FRAIGSolver::collectValues() {
  // Collect block arguments (primary inputs) that are i1
  for (auto arg : module.getBodyBlock()->getArguments()) {
    if (arg.getType().isInteger(1)) {
      primaryInputs.push_back(arg);
      allValues.push_back(arg);
    }
  }

  // Walk operations in topological order and collect i1 results
  module.walk([&](Operation *op) {
    // Skip non-synth operations
    if (!isa<aig::AndInverterOp, mig::MajorityInverterOp>(op))
      return;

    for (auto result : op->getResults()) {
      if (result.getType().isInteger(1))
        allValues.push_back(result);
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Collected " << primaryInputs.size()
                          << " primary inputs and " << allValues.size()
                          << " total i1 values\n");
}

void FRAIGSolver::runSimulation() {
  // Generate random patterns for primary inputs using APInt
  std::random_device rd;
  std::mt19937_64 gen(rd());

  for (auto input : primaryInputs) {
    // Generate random APInt with numPatterns bits
    llvm::APInt pattern(numPatterns, 0);
    // Fill with random bits, 64 bits at a time
    for (unsigned i = 0; i < numPatterns; i += 64) {
      uint64_t randVal = gen();
      unsigned bitsToSet = std::min(64u, numPatterns - i);
      for (unsigned j = 0; j < bitsToSet; ++j) {
        if ((randVal >> j) & 1)
          pattern.setBit(i + j);
      }
    }
    inputPatterns[input] = pattern;
    simSignatures[input] = pattern;
  }

  // Propagate simulation through the circuit in topological order
  for (auto value : allValues) {
    if (simSignatures.count(value))
      continue; // Already computed (primary input)

    simSignatures[value] = simulateValue(value);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "FRAIG: Simulation complete with " << numPatterns
                 << " patterns. Sample signatures:\n";
    unsigned count = 0;
    for (auto &[v, sig] : simSignatures) {
      if (count++ >= 5)
        break;
      llvm::dbgs() << "  " << v << " -> ";
      sig.print(llvm::dbgs(), /*isSigned=*/false);
      llvm::dbgs() << "\n";
    }
  });
}

llvm::APInt FRAIGSolver::simulateValue(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return inputPatterns.lookup(v);

  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    // AND of all inputs with inversions
    llvm::APInt result = llvm::APInt::getAllOnes(numPatterns);
    auto inputs = andOp.getInputs();
    auto inverted = andOp.getInverted();

    for (auto [input, inv] : llvm::zip(inputs, inverted)) {
      llvm::APInt sig = simSignatures.lookup(input);
      if (inv)
        sig.flipAllBits();
      result &= sig;
    }
    return result;
  }

  if (auto majOp = dyn_cast<mig::MajorityInverterOp>(op)) {
    auto inputs = majOp.getInputs();
    auto inverted = majOp.getInverted();

    // Get simulation values with inversions applied
    SmallVector<llvm::APInt> sigs;
    for (auto [input, inv] : llvm::zip(inputs, inverted)) {
      llvm::APInt sig = simSignatures.lookup(input);
      if (inv)
        sig.flipAllBits();
      sigs.push_back(sig);
    }

    // Compute majority - for 3 inputs, use direct formula
    // For more inputs (must be odd), use recursive tree
    llvm::APInt result(numPatterns, 0);
    if (sigs.size() == 3) {
      result = majority3(sigs[0], sigs[1], sigs[2]);
    } else {
      // Recursive majority for >3 inputs
      // MAJ(a,b,c,d,e) = MAJ(MAJ(a,b,c), d, e)
      while (sigs.size() > 3) {
        llvm::APInt maj = majority3(sigs[0], sigs[1], sigs[2]);
        sigs.erase(sigs.begin(), sigs.begin() + 3);
        sigs.insert(sigs.begin(), maj);
      }
      result = (sigs.size() == 3) ? majority3(sigs[0], sigs[1], sigs[2])
                                  : (sigs.size() == 1 ? sigs[0] : result);
    }

    return result;
  }

  return llvm::APInt(numPatterns, 0);
}

//===----------------------------------------------------------------------===//
// Phase 2: Build equivalence classes from simulation
//===----------------------------------------------------------------------===//

void FRAIGSolver::buildEquivalenceClasses() {
  // Group values by their simulation signature (using hash for efficiency).
  // Values with signature S are equivalent candidates.
  // Values with signature ~S are complement candidates.
  //
  // We use the canonical form: min(sig, ~sig) as the key, and track whether
  // each value has the inverted form.

  struct SigInfo {
    Value value;
    bool isInverted; // true if value's sig == ~canonical
  };

  // Map from canonical signature hash to list of values
  llvm::DenseMap<llvm::hash_code, SmallVector<SigInfo>> sigGroups;

  for (auto value : allValues) {
    llvm::APInt sig = simSignatures.lookup(value);
    llvm::APInt invSig = ~sig;

    // Use lexicographically smaller as canonical
    bool useInverted = sig.ugt(invSig);
    const llvm::APInt &canonical = useInverted ? invSig : sig;

    // Hash the canonical signature
    llvm::hash_code hash = llvm::hash_combine_range(
        canonical.getRawData(),
        canonical.getRawData() + canonical.getNumWords());

    sigGroups[hash].push_back({value, useInverted});
  }

  // Build equivalence classes for groups with >1 value
  for (auto &[hash, group] : sigGroups) {
    if (group.size() <= 1)
      continue;

    // Verify that signatures actually match (hash collision check)
    // and separate into equivalence classes
    llvm::DenseMap<llvm::APInt, SmallVector<SigInfo>> exactGroups;
    for (auto &info : group) {
      llvm::APInt sig = simSignatures.lookup(info.value);
      llvm::APInt invSig = ~sig;
      const llvm::APInt &canonical = info.isInverted ? invSig : sig;
      exactGroups[canonical].push_back(info);
    }

    for (auto &[canonical, members] : exactGroups) {
      if (members.size() <= 1)
        continue;

      EquivClass ec;
      ec.representative = members[0].value;
      bool repInverted = members[0].isInverted;

      // Add other members, computing relative inversion
      for (size_t i = 1; i < members.size(); ++i) {
        // If member has same inversion as rep, they're equivalent
        // If different, they're complements
        bool isComplement = (members[i].isInverted != repInverted);
        ec.members.push_back({members[i].value, isComplement});
      }

      equivClasses.push_back(std::move(ec));
      stats.numEquivClasses++;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Built " << equivClasses.size()
                          << " equivalence classes\n");
}

//===----------------------------------------------------------------------===//
// Phase 3: SAT-based verification with incremental solving
//===----------------------------------------------------------------------===//

llvm::SMTExprRef FRAIGSolver::getOrCreateInputSymbol(Value v) {
  auto it = inputSymbols.find(v);
  if (it != inputSymbols.end())
    return it->second;

  // Create a unique name for this input
  std::string name = "input_" + std::to_string(inputSymbols.size());
  llvm::SMTSortRef boolSort = solver->getBoolSort();
  llvm::SMTExprRef sym = solver->mkSymbol(name.c_str(), boolSort);
  inputSymbols[v] = sym;
  return sym;
}

llvm::SMTExprRef FRAIGSolver::buildSMTExpr(Value v) {
  // Check cache first
  auto it = exprCache.find(v);
  if (it != exprCache.end())
    return it->second;

  llvm::SMTExprRef expr;

  // Check if it's a primary input
  Operation *op = v.getDefiningOp();
  if (!op) {
    // Block argument - create/get input symbol
    expr = getOrCreateInputSymbol(v);
    exprCache[v] = expr;
    return expr;
  }

  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    // Build AND expression
    auto inputs = andOp.getInputs();
    auto inverted = andOp.getInverted();

    // Start with first input
    expr = buildSMTExpr(inputs[0]);
    if (inverted[0])
      expr = solver->mkNot(expr);

    // AND with remaining inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
      llvm::SMTExprRef inputExpr = buildSMTExpr(inputs[i]);
      if (inverted[i])
        inputExpr = solver->mkNot(inputExpr);
      expr = solver->mkAnd(expr, inputExpr);
    }

    exprCache[v] = expr;
    return expr;
  }

  if (auto majOp = dyn_cast<mig::MajorityInverterOp>(op)) {
    auto inputs = majOp.getInputs();
    auto inverted = majOp.getInverted();

    // Get input expressions with inversions
    SmallVector<llvm::SMTExprRef> inputExprs;
    for (auto [input, inv] : llvm::zip(inputs, inverted)) {
      llvm::SMTExprRef inputExpr = buildSMTExpr(input);
      if (inv)
        inputExpr = solver->mkNot(inputExpr);
      inputExprs.push_back(inputExpr);
    }

    // Build majority expression
    // MAJ(a,b,c) = (a ∧ b) ∨ (b ∧ c) ∨ (a ∧ c)
    auto buildMaj3 = [this](llvm::SMTExprRef a, llvm::SMTExprRef b,
                            llvm::SMTExprRef c) -> llvm::SMTExprRef {
      auto ab = solver->mkAnd(a, b);
      auto bc = solver->mkAnd(b, c);
      auto ac = solver->mkAnd(a, c);
      return solver->mkOr(solver->mkOr(ab, bc), ac);
    };

    if (inputExprs.size() == 3) {
      expr = buildMaj3(inputExprs[0], inputExprs[1], inputExprs[2]);
    } else {
      // Recursive majority for >3 inputs
      while (inputExprs.size() > 3) {
        llvm::SMTExprRef maj =
            buildMaj3(inputExprs[0], inputExprs[1], inputExprs[2]);
        inputExprs.erase(inputExprs.begin(), inputExprs.begin() + 3);
        inputExprs.insert(inputExprs.begin(), maj);
      }
      expr = (inputExprs.size() == 3)
                 ? buildMaj3(inputExprs[0], inputExprs[1], inputExprs[2])
                 : inputExprs[0];
    }

    exprCache[v] = expr;
    return expr;
  }

  // Unknown operation - create a fresh symbol
  expr = getOrCreateInputSymbol(v);
  exprCache[v] = expr;
  return expr;
}

void FRAIGSolver::verifyCandidates() {
  for (auto &ec : equivClasses) {
    llvm::SMTExprRef repExpr = buildSMTExpr(ec.representative);

    for (auto &[member, isComplement] : ec.members) {
      // Skip if already proven equivalent to something
      if (provenEquiv.count(member))
        continue;

      stats.numSATCalls++;

      // Use incremental solving: push, add constraint, check, pop
      solver->push();

      llvm::SMTExprRef memberExpr = buildSMTExpr(member);

      // Test non-equivalence: if UNSAT, they are equivalent
      // For equivalence: NOT(rep == member) must be UNSAT
      // For complement: NOT(rep == NOT(member)) must be UNSAT
      llvm::SMTExprRef targetExpr =
          isComplement ? solver->mkNot(memberExpr) : memberExpr;
      llvm::SMTExprRef neqConstraint =
          solver->mkNot(solver->mkEqual(repExpr, targetExpr));
      solver->addConstraint(neqConstraint);

      std::optional<bool> result = solver->check();

      if (result.has_value() && !result.value()) {
        // UNSAT - equivalence proven!
        provenEquiv[member] = {ec.representative, isComplement};
        stats.numProvedEquiv++;

        LLVM_DEBUG(llvm::dbgs()
                   << "FRAIG: Proved " << member << " == "
                   << (isComplement ? "NOT(" : "") << ec.representative
                   << (isComplement ? ")" : "") << "\n");
      } else if (result.has_value() && result.value()) {
        // SAT - counterexample found
        stats.numDisprovedEquiv++;

        LLVM_DEBUG(llvm::dbgs() << "FRAIG: Disproved equivalence for " << member
                                << "\n");

        // TODO: Extract CEX and refine equivalence classes if enableFeedback
      }
      // else: UNKNOWN (timeout) - skip

      solver->pop();
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "FRAIG: SAT verification complete. Proved "
                          << stats.numProvedEquiv << " equivalences\n");
}

//===----------------------------------------------------------------------===//
// Phase 4: Merge equivalent nodes
//===----------------------------------------------------------------------===//

void FRAIGSolver::mergeEquivalentNodes() {
  if (provenEquiv.empty())
    return;

  OpBuilder builder(module.getContext());

  for (auto &[value, equiv] : provenEquiv) {
    auto [representative, isComplement] = equiv;

    // Skip if value has no uses
    if (value.use_empty())
      continue;

    Value replacement;
    if (isComplement) {
      // Need to insert a NOT gate (AND with single inverted input)
      // Find insertion point after the representative
      Operation *repOp = representative.getDefiningOp();
      if (repOp) {
        builder.setInsertionPointAfter(repOp);
      } else {
        // Representative is a block argument - insert at beginning
        builder.setInsertionPointToStart(module.getBodyBlock());
      }

      // Create NOT as single-input AND with inversion
      replacement = aig::AndInverterOp::create(builder, value.getLoc(),
                                               representative, /*invert=*/true);
    } else {
      replacement = representative;
    }

    // Replace all uses
    value.replaceAllUsesWith(replacement);
    stats.numMergedNodes++;

    LLVM_DEBUG(llvm::dbgs() << "FRAIG: Merged " << value << " -> "
                            << (isComplement ? "NOT(" : "") << representative
                            << (isComplement ? ")" : "") << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Merged " << stats.numMergedNodes
                          << " nodes\n");
}

//===----------------------------------------------------------------------===//
// Main FRAIG algorithm
//===----------------------------------------------------------------------===//

FRAIGSolver::Stats FRAIGSolver::run() {
  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Starting functional reduction with "
                          << numPatterns << " simulation patterns\n");

  // Phase 1: Collect values and run simulation
  collectValues();
  if (allValues.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "FRAIG: No i1 values to process\n");
    return stats;
  }

  runSimulation();

  // Phase 2: Build equivalence classes
  buildEquivalenceClasses();
  if (equivClasses.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "FRAIG: No equivalence candidates found\n");
    return stats;
  }

  // Phase 3: SAT-based verification
  verifyCandidates();

  // Phase 4: Merge equivalent nodes
  mergeEquivalentNodes();

  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Complete. Stats:\n"
                          << "  Equivalence classes: " << stats.numEquivClasses
                          << "\n"
                          << "  SAT calls: " << stats.numSATCalls << "\n"
                          << "  Proved: " << stats.numProvedEquiv << "\n"
                          << "  Disproved: " << stats.numDisprovedEquiv << "\n"
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

  void runOnOperation() override {
    auto module = getOperation();

    LLVM_DEBUG(llvm::dbgs()
               << "Running FunctionalReduction (FRAIG) pass on "
               << module.getName() << "\n");

    // Create and run FRAIG solver
    FRAIGSolver fraig(module, numRandomPatterns, satConflictLimit,
                      enableCEXFeedback);

    if (!fraig.isValid()) {
      module.emitWarning() << "Z3 solver not available, skipping FRAIG";
      return;
    }

    auto stats = fraig.run();

    // Update pass statistics
    numEquivClasses = stats.numEquivClasses;
    numSATCalls = stats.numSATCalls;
    numProvedEquiv = stats.numProvedEquiv;
    numDisprovedEquiv = stats.numDisprovedEquiv;
    numMergedNodes = stats.numMergedNodes;
  }
};

} // namespace
