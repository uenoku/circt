//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements FRAIG (Functionally Reduced And-Inverter Graph)
// optimization using a built-in minimal CDCL SAT solver. It identifies and
// merges functionally equivalent nodes through simulation-based candidate
// detection followed by SAT-based verification.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/SatSolver.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/RandomNumberGenerator.h"

#include <cmath>

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

// SAT solver is now in circt/Support/SatSolver.h
using circt::MiniSATSolver;

//===----------------------------------------------------------------------===//
// FRAIG Solver - Core FRAIG implementation using built-in SAT solver
//===----------------------------------------------------------------------===//

class FRAIGSolver {
public:
  FRAIGSolver(hw::HWModuleOp module, unsigned numPatterns,
              unsigned conflictLimit, bool enableFeedback)
      : module(module), numPatterns(numPatterns), conflictLimit(conflictLimit),
        enableFeedback(enableFeedback) {}

  ~FRAIGSolver() = default;

  /// Run the FRAIG algorithm and return statistics.
  struct Stats {
    unsigned numEquivClasses = 0;
    unsigned numSATCalls = 0;
    unsigned numProvedEquiv = 0;
    unsigned numDisprovedEquiv = 0;
    unsigned numUnknown = 0;
    unsigned numMergedNodes = 0;
  };
  Stats run();

private:
  // Phase 1: Collect i1 values and run simulation
  void collectValues();
  void runSimulation();
  llvm::APInt simulateValue(Value v);

  // Phase 2: Build equivalence classes from simulation
  void buildEquivalenceClasses();

  // Phase 2b: Sort equivalence classes and members by priority
  void sortByPriority();
  unsigned computeDepth(Value v);

  // Phase 3: SAT-based verification with per-class solver
  void verifyCandidates();

  int getOrCreateLocalVar(Value v, llvm::DenseMap<Value, int> &localVarMap,
                          int &localNextVar);
  void addLocalStructuralConstraints(MiniSATSolver &s, Value v,
                                     llvm::DenseMap<Value, int> &localVarMap,
                                     int &localNextVar,
                                     llvm::DenseSet<Value> &visited);

  // CEX feedback: refine equivalence classes using counterexample from SAT
  void refineByCEX(MiniSATSolver &solver,
                   llvm::DenseMap<Value, int> &localVarMap);

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
  unsigned conflictLimit;
  bool enableFeedback;
  // All i1 values in topological order (inputs first, then operations)
  SmallVector<Value> allValues;

  // Primary inputs (block arguments)
  SmallVector<Value> primaryInputs;

  // Simulation signatures: value -> APInt simulation result
  llvm::DenseMap<Value, llvm::APInt> simSignatures;

  // Input patterns for simulation
  llvm::DenseMap<Value, llvm::APInt> inputPatterns;

  // Equivalence class structure
  struct EquivClass {
    Value representative;
    SmallVector<std::pair<Value, bool>> members; // (value, isComplement)
  };
  SmallVector<EquivClass> equivClasses;

  // Proven equivalences: value -> (representative, isComplement)
  llvm::DenseMap<Value, std::pair<Value, bool>> provenEquiv;

  // Depth cache for priority ordering (memoized)
  llvm::DenseMap<Value, unsigned> depthCache;

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

  // Walk operations and collect i1 results
  // - AIG/MIG operations: add to allValues for simulation
  // - Unknown operations: treat as inputs (assign random patterns)
  module.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!result.getType().isInteger(1))
        continue;

      if (isa<aig::AndInverterOp, mig::MajorityInverterOp>(op)) {
        // Known synth operations - will be simulated
        allValues.push_back(result);
      } else {
        // Unknown operations - treat as primary inputs
        primaryInputs.push_back(result);
        allValues.push_back(result);
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Collected " << primaryInputs.size()
                          << " primary inputs (including unknown ops) and "
                          << allValues.size() << " total i1 values\n");
}

void FRAIGSolver::runSimulation() {
  // Calculate number of 64-bit words needed for numPatterns bits
  unsigned numWords = (numPatterns + 63) / 64;

  for (auto input : primaryInputs) {
    // Generate random words and construct APInt directly
    SmallVector<uint64_t> words(numWords);

    // Use LLVM's getRandomBytes for efficient random generation
    if (llvm::getRandomBytes(words.data(), numWords * sizeof(uint64_t))) {
      // Fallback: if getRandomBytes fails, use zeros (unlikely)
      std::fill(words.begin(), words.end(), 0);
    }

    // Mask the last word if numPatterns is not a multiple of 64
    if (unsigned remainder = numPatterns % 64)
      words.back() &= (1ULL << remainder) - 1;

    // Construct APInt directly from words (most efficient)
    llvm::APInt pattern(numPatterns, words);
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
    return inputPatterns.at(v);

  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    // AND of all inputs with inversions
    llvm::APInt result = llvm::APInt::getAllOnes(numPatterns);
    auto inputs = andOp.getInputs();
    auto inverted = andOp.getInverted();

    for (auto [input, inv] : llvm::zip(inputs, inverted)) {
      llvm::APInt sig = simSignatures.at(input);
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
      llvm::APInt sig = simSignatures.at(input);
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
    llvm::APInt sig = simSignatures.at(value);
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
      llvm::APInt sig = simSignatures.at(info.value);
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

unsigned FRAIGSolver::computeDepth(Value v) {
  auto it = depthCache.find(v);
  if (it != depthCache.end())
    return it->second;

  Operation *op = v.getDefiningOp();
  if (!op) {
    depthCache[v] = 0;
    return 0;
  }

  unsigned maxInputDepth = 0;
  for (Value operand : op->getOperands()) {
    if (operand.getType().isInteger(1))
      maxInputDepth = std::max(maxInputDepth, computeDepth(operand));
  }

  unsigned depth = maxInputDepth + 1;
  depthCache[v] = depth;
  return depth;
}

void FRAIGSolver::sortByPriority() {
  // Priority ordering for SAT efficiency:
  // 1. Within each class, pick the shallowest node as representative
  //    (smallest fanin cone = smallest SAT problem baseline).
  // 2. Sort members by depth (shallowest first = smallest COI overlap,
  //    cheapest SAT calls first).
  // 3. Sort classes by size (smallest first = fewer SAT calls, proven
  //    equivalences help prune later classes via learned clauses).

  for (auto &ec : equivClasses) {
    // Find shallowest node among representative + all members
    unsigned repDepth = computeDepth(ec.representative);
    size_t bestIdx = SIZE_MAX; // SIZE_MAX means representative is best
    unsigned bestDepth = repDepth;
    bool bestComplement = false;

    for (size_t i = 0; i < ec.members.size(); ++i) {
      unsigned d = computeDepth(ec.members[i].first);
      if (d < bestDepth) {
        bestDepth = d;
        bestIdx = i;
        bestComplement = ec.members[i].second;
      }
    }

    // Swap representative if a shallower member was found
    if (bestIdx != SIZE_MAX) {
      Value oldRep = ec.representative;
      ec.representative = ec.members[bestIdx].first;
      // Old rep becomes a member; inversion is relative to new rep
      ec.members[bestIdx] = {oldRep, bestComplement};
      // If new rep was a complement of old rep, flip all member inversions
      if (bestComplement) {
        for (auto &[member, isComp] : ec.members)
          isComp = !isComp;
      }
    }

    // Sort members by depth (shallowest first -> cheapest SAT calls first)
    llvm::sort(ec.members, [this](const std::pair<Value, bool> &a,
                                  const std::pair<Value, bool> &b) {
      return computeDepth(a.first) < computeDepth(b.first);
    });
  }

  // Sort classes: smallest first (fewer members = processed quickly,
  // proven equivalences add clauses that help later classes)
  llvm::sort(equivClasses, [](const EquivClass &a, const EquivClass &b) {
    return a.members.size() < b.members.size();
  });

  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Sorted " << equivClasses.size()
                          << " classes by priority\n");
}

//===----------------------------------------------------------------------===//
// Phase 3: SAT-based verification with per-class solvers
//
// For each equivalence class, we create a fresh MiniSATSolver and encode
// only the cone-of-influence (COI) of the representative and its members.
// This keeps clause databases small and propagation fast.
//===----------------------------------------------------------------------===//

int FRAIGSolver::getOrCreateLocalVar(Value v,
                                     llvm::DenseMap<Value, int> &localVarMap,
                                     int &localNextVar) {
  auto it = localVarMap.find(v);
  if (it != localVarMap.end())
    return it->second;
  int var = ++localNextVar;
  localVarMap[v] = var;
  return var;
}

void FRAIGSolver::addLocalStructuralConstraints(
    MiniSATSolver &s, Value v, llvm::DenseMap<Value, int> &localVarMap,
    int &localNextVar, llvm::DenseSet<Value> &visited) {
  if (!visited.insert(v).second)
    return;

  Operation *op = v.getDefiningOp();
  if (!op)
    return;

  int outVar = getOrCreateLocalVar(v, localVarMap, localNextVar);

  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    auto inputs = andOp.getInputs();
    auto inverted = andOp.getInverted();

    SmallVector<int> inputLits;
    for (auto [input, inv] : llvm::zip(inputs, inverted)) {
      addLocalStructuralConstraints(s, input, localVarMap, localNextVar,
                                    visited);
      int var = getOrCreateLocalVar(input, localVarMap, localNextVar);
      inputLits.push_back(inv ? -var : var);
    }

    // Tseitin: outVar <=> AND(inputLits)
    for (int lit : inputLits) {
      s.add(-outVar);
      s.add(lit);
      s.add(0);
    }
    for (int lit : inputLits)
      s.add(-lit);
    s.add(outVar);
    s.add(0);
    return;
  }

  if (auto majOp = dyn_cast<mig::MajorityInverterOp>(op)) {
    auto inputs = majOp.getInputs();
    auto inverted = majOp.getInverted();

    SmallVector<int> inputLits;
    for (auto [input, inv] : llvm::zip(inputs, inverted)) {
      addLocalStructuralConstraints(s, input, localVarMap, localNextVar,
                                    visited);
      int var = getOrCreateLocalVar(input, localVarMap, localNextVar);
      inputLits.push_back(inv ? -var : var);
    }

    if (inputLits.size() == 3) {
      int a = inputLits[0], b = inputLits[1], c = inputLits[2];
      s.add(-outVar); s.add(a); s.add(b); s.add(0);
      s.add(-outVar); s.add(a); s.add(c); s.add(0);
      s.add(-outVar); s.add(b); s.add(c); s.add(0);
      s.add(outVar); s.add(-a); s.add(-b); s.add(0);
      s.add(outVar); s.add(-a); s.add(-c); s.add(0);
      s.add(outVar); s.add(-b); s.add(-c); s.add(0);
    } else {
      while (inputLits.size() > 3) {
        int a = inputLits[0], b = inputLits[1], c = inputLits[2];
        int aux = ++localNextVar;
        s.add(-aux); s.add(a); s.add(b); s.add(0);
        s.add(-aux); s.add(a); s.add(c); s.add(0);
        s.add(-aux); s.add(b); s.add(c); s.add(0);
        s.add(aux); s.add(-a); s.add(-b); s.add(0);
        s.add(aux); s.add(-a); s.add(-c); s.add(0);
        s.add(aux); s.add(-b); s.add(-c); s.add(0);
        inputLits.erase(inputLits.begin(), inputLits.begin() + 3);
        inputLits.insert(inputLits.begin(), aux);
      }
      if (inputLits.size() == 3) {
        int a = inputLits[0], b = inputLits[1], c = inputLits[2];
        s.add(-outVar); s.add(a); s.add(b); s.add(0);
        s.add(-outVar); s.add(a); s.add(c); s.add(0);
        s.add(-outVar); s.add(b); s.add(c); s.add(0);
        s.add(outVar); s.add(-a); s.add(-b); s.add(0);
        s.add(outVar); s.add(-a); s.add(-c); s.add(0);
        s.add(outVar); s.add(-b); s.add(-c); s.add(0);
      } else if (inputLits.size() == 1) {
        s.add(-outVar); s.add(inputLits[0]); s.add(0);
        s.add(outVar); s.add(-inputLits[0]); s.add(0);
      }
    }
    return;
  }
}

void FRAIGSolver::refineByCEX(MiniSATSolver &solver,
                              llvm::DenseMap<Value, int> &localVarMap) {
  // Extract counterexample from SAT solver and use it as an additional
  // simulation pattern to split false equivalence classes.
  // This is the key ABC optimization: each CEX immediately eliminates
  // many false candidates without needing SAT calls.

  // Extract input values from the counterexample
  llvm::DenseMap<Value, bool> cexValues;
  for (auto input : primaryInputs) {
    auto it = localVarMap.find(input);
    if (it == localVarMap.end())
      continue;
    int val = solver.val(it->second);
    cexValues[input] = (val > 0);
  }

  // Simulate the CEX through all values
  for (auto value : allValues) {
    if (cexValues.count(value))
      continue;

    Operation *op = value.getDefiningOp();
    if (!op) {
      // Unvisited primary input without a SAT variable — default to false
      cexValues[value] = false;
      continue;
    }

    if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
      bool result = true;
      auto inputs = andOp.getInputs();
      auto inverted = andOp.getInverted();
      for (auto [input, inv] : llvm::zip(inputs, inverted)) {
        bool v = cexValues.at(input);
        result &= (inv ? !v : v);
      }
      cexValues[value] = result;
      continue;
    }

    if (auto majOp = dyn_cast<mig::MajorityInverterOp>(op)) {
      auto inputs = majOp.getInputs();
      auto inverted = majOp.getInverted();
      SmallVector<bool> vals;
      for (auto [input, inv] : llvm::zip(inputs, inverted)) {
        bool v = cexValues.at(input);
        vals.push_back(inv ? !v : v);
      }
      // Count true values — majority wins
      unsigned trueCount = llvm::count(vals, true);
      cexValues[value] = (trueCount > vals.size() / 2);
      continue;
    }

    cexValues[value] = false;
  }

  // Refine equivalence classes: remove members where the CEX distinguishes
  // them from their representative. Modifies equivClasses in-place so that
  // callers iterating by index remain valid.
  unsigned removedMembers = 0;
  for (auto &ec : equivClasses) {
    bool repVal = cexValues.at(ec.representative);

    auto it = llvm::remove_if(ec.members, [&](std::pair<Value, bool> &entry) {
      auto [member, isComplement] = entry;
      if (provenEquiv.count(member))
        return false; // Keep proven members
      bool memberVal = cexValues.at(member);
      bool expectedEqual = !isComplement;
      bool matches = (repVal == memberVal) == expectedEqual;
      if (!matches)
        ++removedMembers;
      return !matches;
    });
    ec.members.erase(it, ec.members.end());
  }

  LLVM_DEBUG(if (removedMembers) llvm::dbgs()
             << "FRAIG: CEX feedback removed " << removedMembers
             << " candidates\n");
}

void FRAIGSolver::verifyCandidates() {
  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Starting SAT verification with "
                          << equivClasses.size() << " equivalence classes\n");

  for (size_t ecIdx = 0; ecIdx < equivClasses.size(); ++ecIdx) {
    auto &ec = equivClasses[ecIdx];
    if (ec.members.empty())
      continue;

    // Fresh solver per equivalence class, encoding only the COI.
    MiniSATSolver solver;

    llvm::DenseMap<Value, int> localVarMap;
    llvm::DenseSet<Value> visited;
    int localNextVar = 0;

    addLocalStructuralConstraints(solver, ec.representative, localVarMap,
                                  localNextVar, visited);
    int repVar =
        getOrCreateLocalVar(ec.representative, localVarMap, localNextVar);

    for (size_t mIdx = 0; mIdx < ec.members.size(); ++mIdx) {
      auto [member, isComplement] = ec.members[mIdx];
      if (provenEquiv.count(member))
        continue;

      addLocalStructuralConstraints(solver, member, localVarMap, localNextVar,
                                    visited);
      int memberVar = getOrCreateLocalVar(member, localVarMap, localNextVar);
      int targetLit = isComplement ? -memberVar : memberVar;

      // Check 1: Can rep=1 and target=0?
      stats.numSATCalls++;
      solver.assume(repVar);
      solver.assume(-targetLit);
      auto result1 = solver.solve(conflictLimit);

      if (result1 == MiniSATSolver::kSAT) {
        stats.numDisprovedEquiv++;
        LLVM_DEBUG(llvm::dbgs() << "FRAIG: Disproved " << member << "\n");
        if (enableFeedback)
          refineByCEX(solver, localVarMap);
        continue;
      }

      if (result1 != MiniSATSolver::kUNSAT) {
        stats.numUnknown++;
        LLVM_DEBUG(llvm::dbgs()
                   << "FRAIG: Unknown (conflict limit) for " << member << "\n");
        continue;
      }

      // Check 2: Can rep=0 and target=1?
      stats.numSATCalls++;
      solver.assume(-repVar);
      solver.assume(targetLit);
      auto result2 = solver.solve(conflictLimit);

      if (result2 == MiniSATSolver::kSAT) {
        stats.numDisprovedEquiv++;
        LLVM_DEBUG(llvm::dbgs() << "FRAIG: Disproved " << member << "\n");
        if (enableFeedback)
          refineByCEX(solver, localVarMap);
        continue;
      }

      if (result2 != MiniSATSolver::kUNSAT) {
        stats.numUnknown++;
        LLVM_DEBUG(llvm::dbgs()
                   << "FRAIG: Unknown (conflict limit) for " << member << "\n");
        continue;
      }

      // Both UNSAT — equivalence proven.
      provenEquiv[member] = {ec.representative, isComplement};
      stats.numProvedEquiv++;

      // Add as permanent clauses to help future members.
      solver.add(-repVar); solver.add(targetLit); solver.add(0);
      solver.add(-targetLit); solver.add(repVar); solver.add(0);

      LLVM_DEBUG(llvm::dbgs() << "FRAIG: Proved\n");
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
  // Topologically sort the values

  if (failed(circt::synth::topologicallySortLogicNetwork(module))) {
    LLVM_DEBUG(llvm::dbgs()
               << "FRAIG: Failed to topologically sort logic network\n");
    module->emitError() << "FRAIG: Failed to topologically sort logic network";
    return stats;
  }

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

  // Phase 2b: Sort by priority for SAT efficiency
  sortByPriority();

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

  void runOnOperation() override {
    auto module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Running FunctionalReduction (FRAIG) pass on "
                            << module.getName() << "\n");

    // Create and run FRAIG solver
    FRAIGSolver fraig(module, numRandomPatterns, satConflictLimit,
                      enableCEXFeedback);

    auto stats = fraig.run();

    // Update pass statistics
    numEquivClasses = stats.numEquivClasses;
    numSATCalls = stats.numSATCalls;
    numProvedEquiv = stats.numProvedEquiv;
    numDisprovedEquiv = stats.numDisprovedEquiv;
    numUnknown = stats.numUnknown;
    numMergedNodes = stats.numMergedNodes;
  }
};

} // namespace
