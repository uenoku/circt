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

//===----------------------------------------------------------------------===//
// Minimal CDCL SAT Solver
//
// A lightweight MiniSat-style CDCL solver optimized for FRAIG workloads:
// thousands of small SAT queries on Tseitin-encoded combinational cones.
//
// Design choices for performance parity with ABC's sat_solver:
// - Two-literal watching with binary clause optimization
// - VSIDS variable activity with indexed max-heap
// - 1UIP conflict analysis with self-subsumption minimization
// - Luby restart scheme with per-call conflict limits
// - Assumption-based solving (no push/pop overhead)
// - No preprocessing — pure CDCL only
// - IndexedMap for per-variable data (auto-growing, cache-friendly)
//===----------------------------------------------------------------------===//

/// A max-heap keyed by external scores, with O(log n) insert/pop/increase-key
/// and O(1) membership test. Each element is an integer index in [0, N).
///
/// Used for VSIDS decision ordering: the variable with the highest activity
/// is always at the top.
class IndexedMaxHeap {
public:
  using ScoreFn = llvm::function_ref<double(int)>;

  IndexedMaxHeap() = default;

  void clear() {
    heap.clear();
    pos.clear();
  }

  void grow(int n) { pos.grow(n - 1); }

  bool empty() const { return heap.empty(); }

  bool contains(int v) const { return pos.inBounds(v) && pos[v] != kAbsent; }

  /// Insert element `v` if not already present.
  void insert(int v, ScoreFn score) {
    if (contains(v))
      return;
    pos[v] = static_cast<int>(heap.size());
    heap.push_back(v);
    siftUp(pos[v], score);
  }

  /// Remove and return the maximum element.
  int pop(ScoreFn score) {
    assert(!empty());
    int top = heap[0];
    pos[top] = kAbsent;
    if (heap.size() > 1) {
      heap[0] = heap.back();
      pos[heap[0]] = 0;
      heap.pop_back();
      siftDown(0, score);
    } else {
      heap.pop_back();
    }
    return top;
  }

  /// Notify that the score of `v` has increased (sift up).
  void increased(int v, ScoreFn score) {
    if (contains(v))
      siftUp(pos[v], score);
  }

private:
  static constexpr int kAbsent = -1;

  void siftUp(int i, ScoreFn score) {
    int v = heap[i];
    double sv = score(v);
    while (i > 0) {
      int parent = (i - 1) / 2;
      if (score(heap[parent]) >= sv)
        break;
      heap[i] = heap[parent];
      pos[heap[i]] = i;
      i = parent;
    }
    heap[i] = v;
    pos[v] = i;
  }

  void siftDown(int i, ScoreFn score) {
    int v = heap[i];
    double sv = score(v);
    int sz = static_cast<int>(heap.size());
    for (;;) {
      int child = 2 * i + 1;
      if (child >= sz)
        break;
      if (child + 1 < sz && score(heap[child + 1]) > score(heap[child]))
        child++;
      if (sv >= score(heap[child]))
        break;
      heap[i] = heap[child];
      pos[heap[i]] = i;
      i = child;
    }
    heap[i] = v;
    pos[v] = i;
  }

  SmallVector<int, 0> heap;              // heap array
  llvm::IndexedMap<int> pos{kAbsent};    // element -> position (kAbsent = not in heap)
};

/// A lightweight CDCL SAT solver using LLVM data structures.
///
/// External API uses 1-indexed IPASIR-style variables: positive literal = v,
/// negative literal = -v. Internally, literals are encoded as:
///   pos(v) = 2*(v-1), neg(v) = 2*(v-1)+1
/// so that negation is just XOR with 1, and the variable index is lit >> 1.
class MiniSATSolver {
public:
  /// IPASIR-compatible result codes.
  enum Result : int { kSAT = 10, kUNSAT = 20, kUNKNOWN = 0 };

  MiniSATSolver() = default;

  /// Add a literal to the current clause being built. 0 terminates.
  void add(int lit) {
    if (lit == 0) {
      commitClause();
      clauseBuf.clear();
    } else {
      ensureVar(std::abs(lit));
      clauseBuf.push_back(lit);
    }
  }

  /// Register an assumption for the next solve() call.
  void assume(int lit) {
    ensureVar(std::abs(lit));
    assumptionBuf.push_back(lit);
  }

  /// Solve under current assumptions. Returns kSAT, kUNSAT, or kUNKNOWN.
  Result solve(int64_t confLimit = -1) {
    Result result = solveImpl(confLimit);
    assumptionBuf.clear();
    return result;
  }

  /// Model value for variable `v` after kSAT. Returns v (true) or -v (false).
  int val(int v) const {
    assert(v > 0 && v <= numVars);
    return modelVals[v - 1] == kTrue ? v : -v;
  }

  /// Number of clauses currently in the solver.
  int getNumClauses() const { return numClauses; }

private:
  // ---- Constants ----

  /// Tri-state variable assignment.
  enum Assign : int8_t { kUndef = 0, kTrue = 1, kFalse = -1 };

  /// Sentinel values for reason indices.
  static constexpr int kNoReason = -1;
  static constexpr int kDecisionReason = -2;

  /// Sentinel return value for binary-clause conflicts in propagate().
  static constexpr int kBinaryConflict = -2;

  /// Watch entry tag: high bit set means the entry stores the other literal
  /// of a binary clause (encoded as uint32_t) rather than a clause index.
  static constexpr uint32_t kBinaryTag = 0x80000000u;

  /// Clause index tag: bit 30 set means the index refers to the learnt
  /// clause pool; otherwise it's a problem clause index.
  static constexpr uint32_t kLearntTag = 0x40000000u;
  static constexpr uint32_t kLearntMask = ~kLearntTag;

  /// VSIDS activity rescaling thresholds.
  static constexpr double kVarActRescale = 1e100;
  static constexpr double kVarActRescaleInv = 1e-100;
  static constexpr double kVarActDecayFactor = 1.0 / 0.95;
  static constexpr double kClaActRescale = 1e20;
  static constexpr double kClaActRescaleInv = 1e-20;
  static constexpr double kClaActDecayFactor = 1.0 / 0.999;

  /// Luby restart: base number of conflicts per restart round.
  static constexpr int64_t kLubyBase = 100;
  static constexpr double kLubyFactor = 2.0;

  /// Learned clause reduction: keep at most numVars * kLearntsMultiplier +
  /// kLearntsOffset clauses before triggering reduction.
  static constexpr int kLearntsMultiplier = 3;
  static constexpr int kLearntsOffset = 1000;

  // ---- Literal encoding ----

  static int encodeLit(int lit) {
    return lit > 0 ? 2 * (lit - 1) : 2 * (-lit - 1) + 1;
  }
  static int decodeLit(uint32_t enc) {
    int var = static_cast<int>(enc >> 1) + 1;
    return (enc & 1) ? -var : var;
  }
  static int negLit(int enc) { return enc ^ 1; }
  static int litVar(int enc) { return enc >> 1; }

  // ---- Watch list entry ----

  struct WatchEntry {
    uint32_t data;
    WatchEntry() = default;
    explicit WatchEntry(uint32_t d) : data(d) {}
    bool isBinary() const { return (data & kBinaryTag) != 0; }
    int otherLit() const { return decodeLit(data & ~kBinaryTag); }
    uint32_t clauseIdx() const { return data; }
  };

  // ---- Clause storage ----

  struct Clause {
    uint32_t size;
    bool learnt;
    float activity;
    SmallVector<int, 4> lits;
  };

  Clause &getClause(uint32_t idx) {
    return (idx & kLearntTag) ? learntClauses[idx & kLearntMask]
                              : problemClauses[idx];
  }
  const Clause &getClause(uint32_t idx) const {
    return (idx & kLearntTag) ? learntClauses[idx & kLearntMask]
                              : problemClauses[idx];
  }

  // ---- Variable storage ----

  /// Ensure variable `v` (1-indexed) exists in all per-variable maps.
  void ensureVar(int v) {
    assert(v > 0 && "variables are 1-indexed");
    if (v <= numVars)
      return;
    int old = numVars;
    numVars = v;
    assigns.grow(v);
    varLevels.grow(v);
    varReasons.grow(v);
    varActivity.grow(v);
    varPolarity.grow(v);
    varSeen.grow(v);
    modelVals.grow(v);
    watchLists.resize(numVars * 2);
    auto scoreFn = [this](int x) { return varActivity[x + 1]; };
    vsidsHeap.grow(numVars);
    for (int i = old; i < numVars; i++)
      vsidsHeap.insert(i, scoreFn);
  }

  // ---- VSIDS activity ----

  IndexedMaxHeap::ScoreFn scoreFn() {
    return [this](int v) { return varActivity[v + 1]; };
  }

  void bumpVarActivity(int v) {
    varActivity[v + 1] += varActInc;
    if (varActivity[v + 1] > kVarActRescale) {
      for (int i = 1; i <= numVars; i++)
        varActivity[i] *= kVarActRescaleInv;
      varActInc *= kVarActRescaleInv;
    }
    vsidsHeap.increased(v, scoreFn());
  }

  void decayVarActivity() { varActInc *= kVarActDecayFactor; }

  void bumpClauseActivity(Clause &c) {
    c.activity += static_cast<float>(claActInc);
    if (c.activity > static_cast<float>(kClaActRescale)) {
      for (auto &cl : learntClauses)
        cl.activity *= static_cast<float>(kClaActRescaleInv);
      claActInc *= kClaActRescaleInv;
    }
  }

  void decayClauseActivity() { claActInc *= kClaActDecayFactor; }

  // ---- Trail / Assignment ----

  Assign evalLit(int enc) const {
    Assign a = assigns[litVar(enc) + 1];
    if (a == kUndef)
      return kUndef;
    return static_cast<Assign>((enc & 1) ? -a : a);
  }

  int decisionLevel() const { return static_cast<int>(trailLimits.size()); }

  bool enqueue(int p, int reason) {
    int v = litVar(p);
    if (assigns[v + 1] != kUndef)
      return assigns[v + 1] == ((p & 1) ? kFalse : kTrue);
    assigns[v + 1] = (p & 1) ? kFalse : kTrue;
    varLevels[v + 1] = decisionLevel();
    varReasons[v + 1] = reason;
    trail.push_back(p);
    return true;
  }

  void newDecisionLevel() {
    trailLimits.push_back(static_cast<int>(trail.size()));
  }

  void backtrack(int level) {
    if (decisionLevel() <= level)
      return;
    for (int i = static_cast<int>(trail.size()) - 1;
         i >= trailLimits[level]; i--) {
      int v = litVar(trail[i]);
      assigns[v + 1] = kUndef;
      varReasons[v + 1] = kNoReason;
      varPolarity[v + 1] = (trail[i] & 1) ? 1 : 0;
      if (!vsidsHeap.contains(v))
        vsidsHeap.insert(v, scoreFn());
    }
    trail.resize(trailLimits[level]);
    trailLimits.resize(level);
    propagHead = static_cast<int>(trail.size());
  }

  // ---- Boolean Constraint Propagation ----

  /// Returns -1 if no conflict, kBinaryConflict for binary clause conflicts,
  /// or a clause index for longer clause conflicts.
  int propagate() {
    while (propagHead < static_cast<int>(trail.size())) {
      int p = trail[propagHead++];
      int falseLit = negLit(p);

      auto &ws = watchLists[falseLit];
      size_t i = 0, j = 0;
      while (i < ws.size()) {
        WatchEntry w = ws[i];

        if (w.isBinary()) {
          int otherEnc = encodeLit(w.otherLit());
          if (!enqueue(otherEnc, kNoReason)) {
            binConflLits[0] = falseLit;
            binConflLits[1] = otherEnc;
            // Copy remaining watches and return conflict.
            ws[j++] = ws[i++];
            while (i < ws.size())
              ws[j++] = ws[i++];
            ws.resize(j);
            return kBinaryConflict;
          }
          ws[j++] = ws[i++];
          continue;
        }

        uint32_t ci = w.clauseIdx();
        Clause &c = getClause(ci);
        int *lits = c.lits.data();
        int sz = static_cast<int>(c.size);

        // Put the false literal at position 1.
        if (lits[0] == falseLit)
          std::swap(lits[0], lits[1]);
        assert(lits[1] == falseLit);

        // Clause already satisfied by lits[0]?
        if (evalLit(lits[0]) == kTrue) {
          ws[j++] = ws[i++];
          continue;
        }

        // Search for a new watch literal.
        bool found = false;
        for (int k = 2; k < sz; k++) {
          if (evalLit(lits[k]) != kFalse) {
            std::swap(lits[1], lits[k]);
            watchLists[lits[1]].push_back(WatchEntry(ci));
            found = true;
            break;
          }
        }
        if (found) {
          i++;
          continue;
        }

        // Unit or conflict.
        ws[j++] = ws[i++];
        if (!enqueue(lits[0], static_cast<int>(ci))) {
          while (i < ws.size())
            ws[j++] = ws[i++];
          ws.resize(j);
          return static_cast<int>(ci);
        }
      }
      ws.resize(j);
    }
    return -1; // no conflict
  }

  // ---- Conflict Analysis (1UIP) ----

  void analyze(int conflIdx, SmallVectorImpl<int> &outLearnt,
               int &outBackLevel) {
    int pathCount = 0;
    int p = -1;
    int trailIdx = static_cast<int>(trail.size()) - 1;
    outLearnt.clear();
    outLearnt.push_back(-1); // placeholder for asserting literal

    auto processLit = [&](int lit) {
      int v = litVar(lit);
      if (!varSeen[v + 1] && varLevels[v + 1] > 0) {
        varSeen[v + 1] = 1;
        bumpVarActivity(v);
        if (varLevels[v + 1] == decisionLevel())
          pathCount++;
        else
          outLearnt.push_back(lit);
      }
    };

    // Seed with the conflict clause literals.
    if (conflIdx == kBinaryConflict) {
      processLit(binConflLits[0]);
      processLit(binConflLits[1]);
    } else {
      Clause &c = getClause(conflIdx);
      if (c.learnt)
        bumpClauseActivity(c);
      for (unsigned k = 0; k < c.size; k++)
        processLit(c.lits[k]);
    }

    // Walk trail backwards to find 1UIP.
    while (pathCount > 0) {
      while (!varSeen[litVar(trail[trailIdx]) + 1])
        trailIdx--;
      p = trail[trailIdx--];
      varSeen[litVar(p) + 1] = 0;
      pathCount--;

      if (pathCount > 0) {
        int reason = varReasons[litVar(p) + 1];
        if (reason >= 0) {
          Clause &c = getClause(reason);
          if (c.learnt)
            bumpClauseActivity(c);
          for (unsigned k = 0; k < c.size; k++)
            if (c.lits[k] != p)
              processLit(c.lits[k]);
        }
      }
    }

    outLearnt[0] = negLit(p);

    // Minimize via self-subsumption.
    unsigned levelMask = 0;
    for (size_t i = 1; i < outLearnt.size(); i++)
      levelMask |= 1u << (varLevels[litVar(outLearnt[i]) + 1] & 31);

    size_t writePos = 1;
    for (size_t i = 1; i < outLearnt.size(); i++) {
      int v = litVar(outLearnt[i]);
      if (varReasons[v + 1] < 0 || !isRedundant(v, levelMask))
        outLearnt[writePos++] = outLearnt[i];
    }
    outLearnt.resize(writePos);

    for (auto lit : outLearnt)
      varSeen[litVar(lit) + 1] = 0;

    // Find backtrack level = second-highest decision level in learnt clause.
    if (outLearnt.size() == 1) {
      outBackLevel = 0;
    } else {
      int maxIdx = 1;
      int maxLev = varLevels[litVar(outLearnt[1]) + 1];
      for (size_t i = 2; i < outLearnt.size(); i++) {
        if (varLevels[litVar(outLearnt[i]) + 1] > maxLev) {
          maxLev = varLevels[litVar(outLearnt[i]) + 1];
          maxIdx = static_cast<int>(i);
        }
      }
      std::swap(outLearnt[1], outLearnt[maxIdx]);
      outBackLevel = maxLev;
    }
  }

  /// Check if literal at variable `v` (0-indexed) is redundant (can be
  /// removed from the learned clause by recursive self-subsumption).
  bool isRedundant(int v, unsigned levelMask) {
    analyzeStack.clear();
    analyzeStack.push_back(v);
    int top = static_cast<int>(clearList.size());

    while (!analyzeStack.empty()) {
      int x = analyzeStack.pop_back_val();
      int reason = varReasons[x + 1];
      if (reason < 0)
        return cleanupRedundancyCheck(top);

      Clause &c = getClause(reason);
      for (unsigned k = 0; k < c.size; k++) {
        int lv = litVar(c.lits[k]);
        if (lv == x || varSeen[lv + 1] || varLevels[lv + 1] == 0)
          continue;
        if (varReasons[lv + 1] >= 0 &&
            ((1u << (varLevels[lv + 1] & 31)) & levelMask)) {
          varSeen[lv + 1] = 1;
          analyzeStack.push_back(lv);
          clearList.push_back(lv);
        } else {
          return cleanupRedundancyCheck(top);
        }
      }
    }
    return true;
  }

  bool cleanupRedundancyCheck(int top) {
    for (int i = top; i < static_cast<int>(clearList.size()); i++)
      varSeen[clearList[i] + 1] = 0;
    clearList.resize(top);
    return false;
  }

  // ---- Clause construction ----

  void addWatchPair(int lit0, int lit1, uint32_t ci) {
    watchLists[lit0].push_back(WatchEntry(ci));
    watchLists[lit1].push_back(WatchEntry(ci));
  }

  void addBinaryWatch(int lit0, int lit1) {
    watchLists[lit0].push_back(
        WatchEntry(kBinaryTag | static_cast<uint32_t>(lit1)));
    watchLists[lit1].push_back(
        WatchEntry(kBinaryTag | static_cast<uint32_t>(lit0)));
  }

  void commitClause() {
    if (!ok)
      return;

    // Encode, sort, deduplicate.
    tmpLits.clear();
    for (int lit : clauseBuf)
      tmpLits.push_back(encodeLit(lit));
    llvm::sort(tmpLits);

    size_t w = 0;
    for (size_t i = 0; i < tmpLits.size(); i++) {
      if (w > 0 && tmpLits[i] == tmpLits[w - 1])
        continue;
      if (w > 0 && tmpLits[i] == negLit(tmpLits[w - 1]))
        return; // tautology
      if (decisionLevel() == 0 && evalLit(tmpLits[i]) == kTrue)
        return; // satisfied at root
      if (decisionLevel() == 0 && evalLit(tmpLits[i]) == kFalse)
        continue; // falsified at root
      tmpLits[w++] = tmpLits[i];
    }
    tmpLits.resize(w);

    if (tmpLits.empty()) {
      ok = false;
      return;
    }
    if (tmpLits.size() == 1) {
      ok = enqueue(tmpLits[0], kNoReason);
      return;
    }

    numClauses++;

    if (tmpLits.size() == 2) {
      addBinaryWatch(tmpLits[0], tmpLits[1]);
      return;
    }

    auto ci = static_cast<uint32_t>(problemClauses.size());
    problemClauses.push_back(
        {static_cast<uint32_t>(tmpLits.size()), /*learnt=*/false, 0.0f,
         SmallVector<int, 4>(tmpLits.begin(), tmpLits.end())});
    addWatchPair(tmpLits[0], tmpLits[1], ci);
  }

  void recordLearnt(SmallVectorImpl<int> &lits) {
    if (lits.size() == 1) {
      enqueue(lits[0], kNoReason);
      return;
    }
    if (lits.size() == 2) {
      addBinaryWatch(lits[0], lits[1]);
      enqueue(lits[0], kNoReason);
      return;
    }

    auto ci =
        static_cast<uint32_t>(learntClauses.size()) | kLearntTag;
    learntClauses.push_back(
        {static_cast<uint32_t>(lits.size()), /*learnt=*/true, 0.0f,
         SmallVector<int, 4>(lits.begin(), lits.end())});
    addWatchPair(lits[0], lits[1], ci);
    enqueue(lits[0], static_cast<int>(ci));
  }

  // ---- Learned clause reduction ----

  void reduceLearnts() {
    // Sort by activity ascending; remove the less active half.
    SmallVector<int> order(learntClauses.size());
    for (int i = 0, e = static_cast<int>(order.size()); i < e; i++)
      order[i] = i;
    llvm::sort(order, [&](int a, int b) {
      return learntClauses[a].activity < learntClauses[b].activity;
    });

    size_t target = learntClauses.size() / 2;
    size_t removed = 0;
    for (int idx : order) {
      if (removed >= target)
        break;
      Clause &c = learntClauses[idx];
      if (c.size <= 2)
        continue;
      // Don't remove if clause is the reason for a current assignment.
      int v = litVar(c.lits[0]);
      if (assigns[v + 1] != kUndef &&
          varReasons[v + 1] == static_cast<int>(idx | kLearntTag))
        continue;
      c.size = 0; // mark deleted
      removed++;
    }

    // Purge dangling watch references.
    for (int i = 0; i < numVars * 2; i++) {
      auto &ws = watchLists[i];
      llvm::erase_if(ws, [&](WatchEntry &w) {
        return !w.isBinary() && getClause(w.clauseIdx()).size == 0;
      });
    }
  }

  // ---- Decision ----

  int pickBranchVar() {
    auto sf = scoreFn();
    while (!vsidsHeap.empty()) {
      int v = vsidsHeap.pop(sf);
      if (assigns[v + 1] == kUndef)
        return v;
    }
    return -1;
  }

  // ---- Luby restart sequence ----

  static double luby(double y, int i) {
    int sz, seq;
    for (sz = 1, seq = 0; sz < i + 1; seq++, sz = 2 * sz + 1)
      ;
    while (sz - 1 != i) {
      sz = (sz - 1) >> 1;
      seq--;
      i = i % sz;
    }
    return std::pow(y, seq);
  }

  // ---- Main search loop ----

  Result search(int64_t confBudget) {
    int64_t conflicts = 0;
    SmallVector<int> learntClause;

    for (;;) {
      int confl = propagate();
      if (confl != -1) {
        conflicts++;
        if (decisionLevel() == rootLevel)
          return kUNSAT;
        int backLevel;
        analyze(confl, learntClause, backLevel);
        backtrack(std::max(backLevel, rootLevel));
        recordLearnt(learntClause);
        decayVarActivity();
        decayClauseActivity();
      } else {
        if (confBudget >= 0 && conflicts >= confBudget) {
          backtrack(rootLevel);
          return kUNKNOWN;
        }
        if (static_cast<int>(learntClauses.size()) >
            numVars * kLearntsMultiplier + kLearntsOffset)
          reduceLearnts();

        int next = pickBranchVar();
        if (next == -1) {
          // All variables assigned — satisfiable.
          for (int i = 1; i <= numVars; i++)
            modelVals[i] = assigns[i];
          backtrack(rootLevel);
          return kSAT;
        }
        newDecisionLevel();
        enqueue(varPolarity[next + 1] ? (2 * next + 1) : (2 * next),
                kDecisionReason);
      }
    }
  }

  Result solveImpl(int64_t confLimit) {
    if (!ok)
      return kUNSAT;

    if (propagate() != -1) {
      ok = false;
      return kUNSAT;
    }

    // Push assumptions as decision levels with BCP after each.
    rootLevel = 0;
    for (int extLit : assumptionBuf) {
      int enc = encodeLit(extLit);
      newDecisionLevel();
      rootLevel = decisionLevel();
      if (!enqueue(enc, kDecisionReason) || propagate() != -1) {
        backtrack(0);
        rootLevel = 0;
        return kUNSAT;
      }
    }

    Result result = kUNKNOWN;
    for (int iter = 0; result == kUNKNOWN; iter++) {
      int64_t budget =
          static_cast<int64_t>(kLubyBase * luby(kLubyFactor, iter));
      if (confLimit >= 0)
        budget = std::min(budget, confLimit);
      result = search(budget);
      if (confLimit >= 0) {
        confLimit -= budget;
        if (confLimit <= 0 && result == kUNKNOWN)
          break;
      }
    }

    backtrack(0);
    rootLevel = 0;
    return result;
  }

  // ---- Solver state ----
  int numVars = 0;
  int numClauses = 0;
  int propagHead = 0;
  int rootLevel = 0;
  bool ok = true;
  double varActInc = 1.0;
  double claActInc = 1.0;

  // Per-variable maps (1-indexed via IndexedMap; index 0 unused).
  llvm::IndexedMap<Assign> assigns{kUndef};
  llvm::IndexedMap<int> varLevels{0};
  llvm::IndexedMap<int> varReasons{kNoReason};
  llvm::IndexedMap<double> varActivity{0.0};
  llvm::IndexedMap<int8_t> varPolarity{0};
  llvm::IndexedMap<int8_t> varSeen{0};
  llvm::IndexedMap<Assign> modelVals{kUndef};

  // Per-literal watch lists (indexed by encoded literal).
  SmallVector<SmallVector<WatchEntry, 4>, 0> watchLists;

  // Decision trail.
  SmallVector<int> trail;
  SmallVector<int> trailLimits;

  // VSIDS decision heap.
  IndexedMaxHeap vsidsHeap;

  // Clause pools.
  SmallVector<Clause, 0> problemClauses;
  SmallVector<Clause, 0> learntClauses;

  // Temporaries.
  SmallVector<int> clauseBuf;
  SmallVector<int> assumptionBuf;
  SmallVector<int> tmpLits;
  int binConflLits[2] = {};
  SmallVector<int> analyzeStack;
  SmallVector<int> clearList;
};

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

    // Sort members by depth (shallowest first → cheapest SAT calls first)
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
        bool v = cexValues.lookup(input);
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
        bool v = cexValues.lookup(input);
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
    bool repVal = cexValues.lookup(ec.representative);

    auto it = llvm::remove_if(ec.members, [&](std::pair<Value, bool> &entry) {
      auto [member, isComplement] = entry;
      if (provenEquiv.count(member))
        return false; // Keep proven members
      bool memberVal = cexValues.lookup(member);
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
      int memberVar =
          getOrCreateLocalVar(member, localVarMap, localNextVar);
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

    LLVM_DEBUG(llvm::dbgs()
               << "Running FunctionalReduction (FRAIG) pass on "
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
