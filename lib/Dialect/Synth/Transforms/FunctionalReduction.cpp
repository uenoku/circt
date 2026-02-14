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

#include <random>

#include <cmath>

#ifdef CIRCT_CADICAL_ENABLED
#include <cadical.hpp>
#endif

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

template <typename DerivedSATSolverT>
class FRAIGSolver {
public:
  FRAIGSolver(hw::HWModuleOp module, unsigned numPatterns,
              unsigned conflictLimit, unsigned seed)
      : module(module), numPatterns(numPatterns), conflictLimit(conflictLimit),
        seed(seed) {}

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
  void addLocalStructuralConstraints(DerivedSATSolverT &s, Value v,
                                     llvm::DenseMap<Value, int> &localVarMap,
                                     int &localNextVar,
                                     llvm::DenseSet<Value> &visited);

  /// Helper: Add Tseitin clauses for majority-3 gate: outVar <=> MAJ(a, b, c).
  /// MAJ(a,b,c) is true when at least 2 of {a,b,c} are true.
  static void addMajority3Clauses(DerivedSATSolverT &s, int outVar, int a,
                                  int b, int c);

  // CEX feedback: refine equivalence classes using counterexample from SAT
  void refineByCEX(DerivedSATSolverT &solver,
                   llvm::DenseMap<Value, int> &varMap,
                   const llvm::DenseSet<Value> &visited);

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
  unsigned seed;
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
  llvm::MapVector<Value, std::pair<Value, bool>> provenEquiv;

  // Depth cache for priority ordering (memoized)
  llvm::DenseMap<Value, unsigned> depthCache;

  // Statistics
  Stats stats;
};

//===----------------------------------------------------------------------===//
// Phase 1: Collect values and run simulation
//===----------------------------------------------------------------------===//

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::collectValues() {
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

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::runSimulation() {
  // Calculate number of 64-bit words needed for numPatterns bits
  unsigned numWords = (numPatterns + 63) / 64;

  // Create seeded random number generator for deterministic patterns
  std::mt19937_64 rng(seed);

  for (auto input : primaryInputs) {
    // Generate random words using seeded RNG
    SmallVector<uint64_t> words(numWords);
    for (auto &word : words)
      word = rng();

    // Mask the last word if numPatterns is not a multiple of 64
    if (unsigned remainder = numPatterns % 64)
      words.back() &= (1ULL << remainder) - 1;

    // Construct APInt directly from words
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

template <typename SATSolverT>
llvm::APInt FRAIGSolver<SATSolverT>::simulateValue(Value v) {
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

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::buildEquivalenceClasses() {
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
  llvm::MapVector<llvm::hash_code, SmallVector<SigInfo>> sigGroups;

  for (auto value : allValues) {
    llvm::APInt sig = simSignatures.at(value);
    llvm::APInt invSig = ~sig;

    // Use lexicographically smaller as canonical
    bool useInverted = sig.ugt(invSig);
    const llvm::APInt &canonical = useInverted ? invSig : sig;

    // NOTE: Hash the canonical signature. Because APInt is too hearvy to store
    // directly as a DenseMap key, we use its hash. We anyway verify the
    // equivalence of values with SAT solving.
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
    llvm::MapVector<llvm::APInt, SmallVector<SigInfo>> exactGroups;
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

template <typename SATSolverT>
unsigned FRAIGSolver<SATSolverT>::computeDepth(Value v) {
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

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::sortByPriority() {
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

template <typename SATSolverT>
int FRAIGSolver<SATSolverT>::getOrCreateLocalVar(
    Value v, llvm::DenseMap<Value, int> &localVarMap, int &localNextVar) {
  auto it = localVarMap.find(v);
  if (it != localVarMap.end())
    return it->second;
  int var = ++localNextVar;
  localVarMap[v] = var;
  return var;
}

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::addMajority3Clauses(SATSolverT &s, int outVar,
                                                  int a, int b, int c) {
  // Tseitin encoding for: outVar <=> MAJ(a, b, c)
  // MAJ is true when at least 2 of {a, b, c} are true.
  //
  // Forward direction (outVar => at least 2 true):
  //   outVar => (a AND b) OR (a AND c) OR (b AND c)
  //   Clauses: (!outVar OR a OR b), (!outVar OR a OR c), (!outVar OR b OR c)
  s.add(-outVar);
  s.add(a);
  s.add(b);
  s.add(0);
  s.add(-outVar);
  s.add(a);
  s.add(c);
  s.add(0);
  s.add(-outVar);
  s.add(b);
  s.add(c);
  s.add(0);

  // Backward direction (at least 2 true => outVar):
  //   (a AND b) OR (a AND c) OR (b AND c) => outVar
  //   Clauses: (outVar OR !a OR !b), (outVar OR !a OR !c), (outVar OR !b OR !c)
  s.add(outVar);
  s.add(-a);
  s.add(-b);
  s.add(0);
  s.add(outVar);
  s.add(-a);
  s.add(-c);
  s.add(0);
  s.add(outVar);
  s.add(-b);
  s.add(-c);
  s.add(0);
}

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::addLocalStructuralConstraints(
    SATSolverT &s, Value v, llvm::DenseMap<Value, int> &localVarMap,
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
      // Direct majority-3 encoding
      addMajority3Clauses(s, outVar, inputLits[0], inputLits[1], inputLits[2]);
    } else {
      // Recursive decomposition for >3 inputs
      while (inputLits.size() > 3) {
        int a = inputLits[0], b = inputLits[1], c = inputLits[2];
        int aux = ++localNextVar;
        addMajority3Clauses(s, aux, a, b, c);
        inputLits.erase(inputLits.begin(), inputLits.begin() + 3);
        inputLits.insert(inputLits.begin(), aux);
      }
      if (inputLits.size() == 3) {
        addMajority3Clauses(s, outVar, inputLits[0], inputLits[1],
                            inputLits[2]);
      } else if (inputLits.size() == 1) {
        // Degenerate case: single input (shouldn't happen in practice)
        s.add(-outVar);
        s.add(inputLits[0]);
        s.add(0);
        s.add(outVar);
        s.add(-inputLits[0]);
        s.add(0);
      }
    }
    return;
  }
}

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::refineByCEX(
    SATSolverT &solver, llvm::DenseMap<Value, int> &varMap,
    const llvm::DenseSet<Value> &visited) {
  // Extract counterexample from SAT solver and use it as an additional
  // simulation pattern to split false equivalence classes.
  // This is the key ABC optimization: each CEX immediately eliminates
  // many false candidates without needing SAT calls.

  // Extract input values from the counterexample
  llvm::DenseMap<Value, bool> cexValues;
  for (auto input : primaryInputs) {
    auto it = varMap.find(input);
    if (it == varMap.end() || !visited.count(input))
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

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::verifyCandidates() {
  LLVM_DEBUG(llvm::dbgs() << "FRAIG: Starting SAT verification with "
                          << equivClasses.size() << " equivalence classes\n");

  // Check if user wants to dump all SAT queries
  const char *dumpDir = std::getenv("FRAIG_DUMP_SAT");
  bool dumpAll = (dumpDir != nullptr);

  // Global solver with bookmark/rollback (ABC-style):
  // - Pre-allocate variables for all values so they persist across rollbacks
  // - Learned clauses accumulate across equivalence classes (knowledge sharing)
  // - VSIDS activity stays warm (tuned variable ordering)
  SATSolverT solver;

  // Pre-allocate a SAT variable for every i1 value
  llvm::DenseMap<Value, int> globalVarMap;
  int maxVar = 0;
  for (auto value : allValues)
    globalVarMap[value] = ++maxVar;
  solver.reserveVars(maxVar);

  // Bookmark: save state with all variables allocated but no clauses
  solver.bookmark();

  for (size_t ecIdx = 0; ecIdx < equivClasses.size(); ++ecIdx) {
    auto &ec = equivClasses[ecIdx];
    if (ec.members.empty())
      continue;

    llvm::DenseSet<Value> visited;
    // Aux variables (for MAJ>3 decomposition) start after global variables
    int localNextVar = maxVar;

    addLocalStructuralConstraints(solver, ec.representative, globalVarMap,
                                  localNextVar, visited);
    int repVar = globalVarMap[ec.representative];

    for (size_t mIdx = 0; mIdx < ec.members.size(); ++mIdx) {
      auto [member, isComplement] = ec.members[mIdx];
      if (provenEquiv.count(member))
        continue;

      addLocalStructuralConstraints(solver, member, globalVarMap, localNextVar,
                                    visited);
      int memberVar = globalVarMap[member];
      int targetLit = isComplement ? -memberVar : memberVar;

      // Check 1: Can rep=1 and target=0?
      stats.numSATCalls++;

      // Optionally dump SAT query for debugging
      if (dumpAll) {
        std::string filename = std::string(dumpDir) + "/sat_query_" +
                               std::to_string(stats.numSATCalls) + "_check1.cnf";
        std::error_code EC;
        llvm::raw_fd_ostream out(filename, EC);
        if (!EC) {
          solver.dumpDIMACS(out, {repVar, -targetLit});
          LLVM_DEBUG(llvm::dbgs()
                     << "FRAIG: Dumped SAT query to " << filename << "\n");
        }
      }

      solver.assume(repVar);
      solver.assume(-targetLit);
      auto result1 = solver.solve(conflictLimit);

      if (result1 == IncrementalSATSolverBase<SATSolverT>::kSAT) {
        stats.numDisprovedEquiv++;
        LLVM_DEBUG(llvm::dbgs() << "FRAIG: Disproved " << member << "\n");
        refineByCEX(solver, globalVarMap, visited);
        continue;
      }

      if (result1 != IncrementalSATSolverBase<SATSolverT>::kUNSAT) {
        stats.numUnknown++;
        LLVM_DEBUG(llvm::dbgs()
                   << "FRAIG: Unknown (conflict limit) for " << member << "\n");
        continue;
      }

      // Check 2: Can rep=0 and target=1?
      stats.numSATCalls++;

      // Optionally dump SAT query for debugging
      if (dumpAll) {
        std::string filename = std::string(dumpDir) + "/sat_query_" +
                               std::to_string(stats.numSATCalls) + "_check2.cnf";
        std::error_code EC;
        llvm::raw_fd_ostream out(filename, EC);
        if (!EC) {
          solver.dumpDIMACS(out, {-repVar, targetLit});
          LLVM_DEBUG(llvm::dbgs()
                     << "FRAIG: Dumped SAT query to " << filename << "\n");
        }
      }

      solver.assume(-repVar);
      solver.assume(targetLit);
      auto result2 = solver.solve(conflictLimit);

      if (result2 == IncrementalSATSolverBase<SATSolverT>::kSAT) {
        stats.numDisprovedEquiv++;
        LLVM_DEBUG(llvm::dbgs() << "FRAIG: Disproved " << member << "\n");
        refineByCEX(solver, globalVarMap, visited);
        continue;
      }

      if (result2 != IncrementalSATSolverBase<SATSolverT>::kUNSAT) {
        stats.numUnknown++;
        LLVM_DEBUG(llvm::dbgs()
                   << "FRAIG: Unknown (conflict limit) for " << member << "\n");
        continue;
      }

      // Both UNSAT — equivalence proven.
      provenEquiv[member] = {ec.representative, isComplement};
      stats.numProvedEquiv++;

      // Add as permanent clauses to help future members within this class.
      solver.add(-repVar);
      solver.add(targetLit);
      solver.add(0);
      solver.add(-targetLit);
      solver.add(repVar);
      solver.add(0);

      LLVM_DEBUG(llvm::dbgs() << "FRAIG: Proved\n");
    }

    // Rollback: remove cone-specific clauses, keep learned clauses and VSIDS
    solver.rollback();
  }

  LLVM_DEBUG(llvm::dbgs() << "FRAIG: SAT verification complete. Proved "
                          << stats.numProvedEquiv << " equivalences\n");
}

//===----------------------------------------------------------------------===//
// Phase 4: Merge equivalent nodes
//===----------------------------------------------------------------------===//

template <typename SATSolverT>
void FRAIGSolver<SATSolverT>::mergeEquivalentNodes() {
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

template <typename SATSolverT>
typename FRAIGSolver<SATSolverT>::Stats FRAIGSolver<SATSolverT>::run() {
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

#ifdef CIRCT_CADICAL_ENABLED
#include <cadical.hpp>

/// CaDiCaL-based SAT solver using external CaDiCaL library.
///
/// Provides the same interface as MiniSATSolver but uses CaDiCaL's optimized
/// CDCL implementation. CaDiCaL is a high-performance SAT solver that often
/// outperforms custom implementations on large industrial problems.
class CadicalSATSolver : public IncrementalSATSolverBase<CadicalSATSolver> {
public:
  // Note: Result enum inherited from CRTP base class

  CadicalSATSolver() : solver(new CaDiCaL::Solver()) {
    // Disable BVA (bounded variable addition) to avoid CaDiCaL introducing
    // new variables that conflict with our variable numbering.
    solver->set("factor", 0);
    solver->set("factorcheck", 0);
  }

  ~CadicalSATSolver() { delete solver; }

  /// Add a literal to the current clause being built. 0 terminates.
  void add(int lit) {
    // Co-simulate with MiniSAT
    testSolver.add(lit);

    if (lit == 0) {
      // Terminate clause
      solver->add(0);
      if (!currentClause.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "CaDiCaL: Adding clause: ";
                   for (int l : currentClause) llvm::dbgs() << l << " ";
                   llvm::dbgs() << "0\n");
        clauses.push_back(currentClause);
        currentClause.clear();
      }
    } else {
      solver->add(lit);
      currentClause.push_back(lit);
    }
  }

  void assume(int lit) {
    LLVM_DEBUG(llvm::dbgs() << "Assuming: " << lit << "\n");
    solver->assume(lit);
    testSolver.assume(lit);
    currentAssumptions.push_back(lit);
  }

  [[nodiscard]] Result solve(int64_t confLimit = -1) {
    llvm::errs() << "PRE-SOLVE: CaDiCaL clauses=" << clauses.size()
                 << " MiniSAT clauses=" << testSolver.getNumClauses() << "\n";

    if (confLimit > 0) {
      solver->limit("conflicts", confLimit);
    }
    int res = solver->solve();
    Result cadicalResult = (res == CaDiCaL::SATISFIABLE
                                ? kSAT
                                : (res == CaDiCaL::UNSATISFIABLE ? kUNSAT
                                                                 : kUNKNOWN));
    auto miniSatResult = testSolver.solve(confLimit);

    LLVM_DEBUG(llvm::dbgs()
               << "SOLVE: CaDiCaL="
               << (cadicalResult == kSAT
                       ? "SAT"
                       : (cadicalResult == kUNSAT ? "UNSAT" : "UNKNOWN"))
               << " MiniSAT="
               << (miniSatResult == MiniSATSolver::kSAT
                       ? "SAT"
                       : (miniSatResult == MiniSATSolver::kUNSAT ? "UNSAT"
                                                                  : "UNKNOWN"))
               << "\n");

    if (static_cast<int>(cadicalResult) != static_cast<int>(miniSatResult)) {
      llvm::errs() << "ERROR: Solver mismatch! CaDiCaL="
                   << (cadicalResult == kSAT
                           ? "SAT"
                           : (cadicalResult == kUNSAT ? "UNSAT" : "UNKNOWN"))
                   << " MiniSAT="
                   << (miniSatResult == MiniSATSolver::kSAT
                           ? "SAT"
                           : (miniSatResult == MiniSATSolver::kUNSAT ? "UNSAT"
                                                                      : "UNKNOWN"))
                   << "\n";

      // Dump DIMACS reproducer for debugging
      static int mismatchCount = 0;
      std::string filename =
          "/tmp/cadical_mismatch_" + std::to_string(++mismatchCount) + ".cnf";
      std::error_code EC;
      llvm::raw_fd_ostream out(filename, EC);
      if (!EC) {
        dumpDIMACS(out, currentAssumptions);
        llvm::errs() << "Dumped reproducer to " << filename << "\n";
      } else {
        llvm::errs() << "Failed to dump reproducer: " << EC.message() << "\n";
      }

      // Also dump MiniSAT's view
      std::string miniFilename =
          "/tmp/minisat_mismatch_" + std::to_string(mismatchCount) + ".cnf";
      llvm::raw_fd_ostream miniOut(miniFilename, EC);
      if (!EC) {
        // Convert assumptions to proper format for dumpDIMACS
        llvm::SmallVector<int> extAssumptions;
        for (int lit : currentAssumptions) {
          extAssumptions.push_back(lit);
        }
        testSolver.dumpDIMACS(miniOut, extAssumptions);
        llvm::errs() << "Dumped MiniSAT view to " << miniFilename << "\n";
      }
    }

    // Clear assumptions for next solve
    currentAssumptions.clear();

    return cadicalResult;
  }

  /// Query model value after kSAT. Returns v (true) or -v (false).
  [[nodiscard]] int val(int v) const {
    int cadicalVal = solver->val(v);
    int miniSatVal = testSolver.val(v);

    if (cadicalVal != miniSatVal) {
      llvm::errs() << "ERROR: Model value mismatch for var " << v
                   << "! CaDiCaL=" << cadicalVal << " MiniSAT=" << miniSatVal
                   << "\n";
    }

    return cadicalVal; // CaDiCaL returns v for true, -v for false
  }

  /// Save current solver state by storing all clauses added so far.
  void bookmark() {
    LLVM_DEBUG(llvm::dbgs() << "BOOKMARK: " << clauses.size() << " clauses\n");
    bookmarkedClauses = clauses;
    hasBookmark_ = true;
    testSolver.bookmark();
  }

  void reserveVars(int maxVar) {
    LLVM_DEBUG(llvm::dbgs() << "RESERVE_VARS: " << maxVar << "\n");
    solver->resize(maxVar);
    testSolver.reserveVars(maxVar);
  }

  /// Restore to the bookmarked state by recreating solver and re-adding
  /// clauses.
  void rollback() {
    assert(hasBookmark_ && "No bookmark to rollback to");

    LLVM_DEBUG(llvm::dbgs()
               << "ROLLBACK: Restoring to " << bookmarkedClauses.size()
               << " clauses (was " << clauses.size() << ")\n");

    // Recreate solver and restore state
    delete solver;
    solver = new CaDiCaL::Solver();
    solver->set("factor", 0);
    solver->set("factorcheck", 0);

    // Re-add all clauses up to the bookmark
    for (const auto &clause : bookmarkedClauses) {
      for (int lit : clause) {
        solver->add(lit);
      }
      solver->add(0); // Terminate clause
    }

    // Restore clauses list to bookmarked state
    clauses = bookmarkedClauses;
    // Clear current clause buffer
    currentClause.clear();

    // Co-simulate rollback with MiniSAT
    testSolver.rollback();
  }

  [[nodiscard]] bool hasBookmark() const { return hasBookmark_; }

  /// Export current formula in DIMACS CNF format for debugging.
  void dumpDIMACS(llvm::raw_ostream &os,
                  llvm::ArrayRef<int> assumptions = {}) const {
    // Count total clauses
    size_t totalClauses = clauses.size() + assumptions.size();

    // DIMACS header (need to track maxVar)
    int maxVar = 0;
    for (const auto &clause : clauses) {
      for (int lit : clause) {
        maxVar = std::max(maxVar, std::abs(lit));
      }
    }
    for (int lit : assumptions) {
      maxVar = std::max(maxVar, std::abs(lit));
    }

    os << "p cnf " << maxVar << " " << totalClauses << "\n";

    // All clauses
    for (const auto &clause : clauses) {
      for (int lit : clause)
        os << lit << " ";
      os << "0\n";
    }

    // Assumptions as unit clauses
    for (int lit : assumptions) {
      os << lit << " 0\n";
    }
  }

private:
  CaDiCaL::Solver *solver;
  bool hasBookmark_ = false;
  llvm::SmallVector<llvm::SmallVector<int, 4>, 0> clauses; // All clauses added
  llvm::SmallVector<llvm::SmallVector<int, 4>, 0>
      bookmarkedClauses;                   // Clauses at bookmark
  llvm::SmallVector<int, 4> currentClause; // Current clause being built
  llvm::SmallVector<int, 4> currentAssumptions; // Current assumptions

  MiniSATSolver testSolver;
};

#endif // CIRCT_CADICAL_ENABLED

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

    // Use CaDiCaL SAT solver if available
#ifdef CIRCT_CADICAL_ENABLED
    if (getenv("USE_CADICAL")) {
      FRAIGSolver<CadicalSATSolver> fraig(module, numRandomPatterns,
                                          satConflictLimit, seed);
      auto stats = fraig.run();
      // Update pass statistics
      numEquivClasses = stats.numEquivClasses;
      numSATCalls = stats.numSATCalls;
      numProvedEquiv = stats.numProvedEquiv;
      numDisprovedEquiv = stats.numDisprovedEquiv;
      numUnknown = stats.numUnknown;
      numMergedNodes = stats.numMergedNodes;
      return;
    }
#endif
    // Fall back to built-in MiniSAT solver
    FRAIGSolver<MiniSATSolver> fraig(module, numRandomPatterns,
                                     satConflictLimit, seed);

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
