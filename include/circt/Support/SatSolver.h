//===- SatSolver.h - Lightweight CDCL SAT Solver ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A lightweight MiniSat-style CDCL SAT solver optimized for FRAIG workloads:
// thousands of small SAT queries on Tseitin-encoded combinational cones.
//
// Design choices for performance parity with ABC's sat_solver:
// - Two-literal watching with binary clause optimization
// - VSIDS variable activity with indexed max-heap
// - 1UIP conflict analysis with self-subsumption minimization
// - Luby restart scheme with per-call conflict limits
// - Assumption-based solving (no push/pop overhead)
// - No preprocessing — pure CDCL only
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_SATSOLVER_H
#define CIRCT_SUPPORT_SATSOLVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

namespace circt {

/// Max-heap of integer indices keyed by external scores.
///
/// Provides O(log n) insert/pop/increase-key and O(1) membership test.
/// Cannot use llvm::PriorityQueue because it lacks efficient increase-key
/// (its reheapify() is O(n)). VSIDS decision ordering requires frequent
/// activity bumps, making O(log n) increase-key essential.
class IndexedMaxHeap {
public:
  using ScoreFn = llvm::function_ref<double(int)>;

  void clear() {
    heap.clear();
    pos.clear();
  }

  /// Ensure capacity for indices [0, n).
  void grow(int n) {
    if (n > static_cast<int>(pos.size()))
      pos.resize(n, kAbsent);
  }

  [[nodiscard]] bool empty() const { return heap.empty(); }

  [[nodiscard]] bool contains(int idx) const {
    return idx >= 0 && idx < static_cast<int>(pos.size()) &&
           pos[idx] != kAbsent;
  }

  /// Insert element if not already present.
  void insert(int idx, ScoreFn score) {
    if (contains(idx))
      return;
    pos[idx] = static_cast<int>(heap.size());
    heap.push_back(idx);
    siftUp(pos[idx], score);
  }

  /// Remove and return the maximum element.
  [[nodiscard]] int pop(ScoreFn score) {
    assert(!empty() && "cannot pop from empty heap");
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

  /// Notify that the score of idx has increased. O(log n) sift-up.
  void increased(int idx, ScoreFn score) {
    if (contains(idx))
      siftUp(pos[idx], score);
  }

private:
  static constexpr int kAbsent = -1;

  void siftUp(int heapIdx, ScoreFn score) {
    int elem = heap[heapIdx];
    double elemScore = score(elem);
    while (heapIdx > 0) {
      int parentIdx = (heapIdx - 1) / 2;
      if (score(heap[parentIdx]) >= elemScore)
        break;
      heap[heapIdx] = heap[parentIdx];
      pos[heap[heapIdx]] = heapIdx;
      heapIdx = parentIdx;
    }
    heap[heapIdx] = elem;
    pos[elem] = heapIdx;
  }

  void siftDown(int heapIdx, ScoreFn score) {
    int elem = heap[heapIdx];
    double elemScore = score(elem);
    int heapSize = static_cast<int>(heap.size());
    for (;;) {
      int childIdx = 2 * heapIdx + 1;
      if (childIdx >= heapSize)
        break;
      // Choose the larger child
      if (childIdx + 1 < heapSize &&
          score(heap[childIdx + 1]) > score(heap[childIdx]))
        childIdx++;
      if (elemScore >= score(heap[childIdx]))
        break;
      heap[heapIdx] = heap[childIdx];
      pos[heap[heapIdx]] = heapIdx;
      heapIdx = childIdx;
    }
    heap[heapIdx] = elem;
    pos[elem] = heapIdx;
  }

  llvm::SmallVector<int, 0> heap; // Heap array storing element indices
  llvm::SmallVector<int, 0> pos;  // pos[elem] = heap position or kAbsent
};

/// CRTP base class for incremental SAT solving, supporting assumptions and
/// state bookmarking for efficient repeated queries on a shared base formula.
/// Uses compile-time polymorphism to avoid virtual function overhead.
template <typename Derived>
class IncrementalSATSolverBase {
public:
  /// Result of SAT query. Follows IPASIR convention for interoperability with
  /// other solvers.
  enum Result : int { kSAT = 10, kUNSAT = 20, kUNKNOWN = 0 };

  /// Add a literal to the current clause being built.
  /// Clauses are terminated by calling add(0). Literals are 1-indexed, with
  /// negative values representing negated literals (e.g. -3 means NOT x3).
  void add(int lit) { derived().add(lit); }

  /// Add an assumption literal for the next solve() call. Assumptions are
  /// temporary unit clauses that are only active for a single solve()
  /// invocation.
  void assume(int lit) { derived().assume(lit); }

  /// Solve under current assumptions. Returns kSAT, kUNSAT, or kUNKNOWN (e.g.
  /// due to conflict limit).
  Result solve(int64_t confLimit = -1) { return derived().solve(confLimit); }

  /// Query model value after kSAT. Returns v (true) or -v (false).
  int val(int v) const { return derived().val(v); }

  /// Save current state for later rollback. It's ensured that bookmark is only
  /// called at decision level 0 (i.e. no outstanding assumptions). This allows
  /// the implementation to optimize for the FRAIG use case of many queries on a
  /// shared base formula with different assumptions, without needing to support
  /// arbitrary push/pop. Enables incremental solving by allowing the solver to
  /// return to this state while preserving learned clauses and VSIDS activity
  /// across rollbacks.
  void bookmark() { derived().bookmark(); }

  /// Restore to last bookmarked state. Removes variables and problem clauses
  /// added after bookmark, but preserves learned clauses and VSIDS activity.
  /// This enables sharing knowledge across multiple SAT queries.
  void rollback() { derived().rollback(); }

  /// Check if a bookmark exists.
  bool hasBookmark() const { return derived().hasBookmark(); }

private:
  Derived &derived() { return static_cast<Derived &>(*this); }
  const Derived &derived() const { return static_cast<const Derived &>(*this); }
};

/// Lightweight CDCL SAT solver using LLVM data structures.
///
/// External API uses 1-indexed IPASIR-style variables: positive literal = v,
/// negative literal = -v. Internally, literals are encoded as:
///   pos(v) = 2*(v-1), neg(v) = 2*(v-1)+1
/// so that negation is XOR 1, and the variable index is lit >> 1.
class MiniSATSolver : public IncrementalSATSolverBase<MiniSATSolver> {
public:
  // Note: Result enum inherited from CRTP base class

  /// Add a literal to the current clause being built. 0 terminates.
  void add(int lit);

  void assume(int lit);

  [[nodiscard]] Result solve(int64_t confLimit = -1);

  /// Query model value after kSAT. Returns v (true) or -v (false).
  [[nodiscard]] int val(int v) const;

  [[nodiscard]] int getNumClauses() const { return numClauses; }

  /// Pre-allocate variables up to maxVar (1-indexed) without adding clauses.
  void reserveVars(int maxVar) { ensureVar(maxVar); }

  /// Export current formula in DIMACS CNF format. Optionally includes
  /// assumptions as unit clauses. Useful for debugging and creating reproducers.
  /// Format: "p cnf <vars> <clauses>\n" followed by clauses (space-separated
  /// literals terminated by 0).
  void dumpDIMACS(llvm::raw_ostream &os,
                  llvm::ArrayRef<int> assumptions = {}) const;

  /// Save current solver state. Must be at decision level 0 with no pending
  /// propagations. On rollback, variables/clauses added after bookmark are
  /// removed, watch lists are rebuilt from surviving clauses, and VSIDS
  /// activity/polarity are preserved. Learned clauses added before the
  /// bookmark also survive (they reference persistent variables).
  void bookmark();

  /// Restore to the bookmarked state. Removes post-bookmark variables and
  /// clauses, rebuilds watch lists from surviving clauses (like ABC's
  /// sat_solver_rollback), preserves VSIDS activity and phase-saving polarity.
  void rollback();

private:
  enum Assign : int8_t { kUndef = 0, kTrue = 1, kFalse = -1 };

  static constexpr int kNoReason = -1;
  static constexpr int kDecisionReason = -2;

  // Watch list tags: high bit distinguishes binary clauses from clause indices
  static constexpr uint32_t kBinaryTag = 0x80000000u;
  static constexpr uint32_t kLearntTag = 0x40000000u;
  static constexpr uint32_t kLearntMask = ~kLearntTag;

  // VSIDS activity rescaling to prevent overflow
  static constexpr double kVarActRescale = 1e100;
  static constexpr double kVarActRescaleInv = 1e-100;
  static constexpr double kVarActDecayFactor = 1.0 / 0.95;
  static constexpr double kClaActRescale = 1e20;
  static constexpr double kClaActRescaleInv = 1e-20;
  static constexpr double kClaActDecayFactor = 1.0 / 0.999;

  // Luby restart strategy parameters
  static constexpr int64_t kLubyBase = 100;
  static constexpr double kLubyFactor = 2.0;

  // Learned clause reduction trigger
  static constexpr int kLearntsMultiplier = 3;
  static constexpr int kLearntsOffset = 1000;

  // Literal encoding helpers - force inlined for performance (hot path)

  LLVM_ATTRIBUTE_ALWAYS_INLINE static int encodeLit(int lit) {
    return lit > 0 ? 2 * (lit - 1) : 2 * (-lit - 1) + 1;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE static int decodeLit(uint32_t enc) {
    int var = static_cast<int>(enc >> 1) + 1;
    return (enc & 1) ? -var : var;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE static int negLit(int enc) { return enc ^ 1; }
  LLVM_ATTRIBUTE_ALWAYS_INLINE static int litVar(int enc) { return enc >> 1; }

  // Watch list entry: encodes either a binary clause's other literal or a
  // clause index, distinguished by kBinaryTag.
  struct WatchEntry {
    uint32_t data = 0;
    WatchEntry() = default;
    explicit WatchEntry(uint32_t d) : data(d) {}
    [[nodiscard]] bool isBinary() const { return (data & kBinaryTag) != 0; }
    [[nodiscard]] int otherLit() const { return decodeLit(data & ~kBinaryTag); }
    [[nodiscard]] uint32_t clauseIdx() const { return data; }
  };

  struct Clause {
    uint32_t size;
    bool learnt;
    float activity;
    llvm::SmallVector<int, 4> lits;
  };

  LLVM_ATTRIBUTE_ALWAYS_INLINE Clause &getClause(uint32_t idx) {
    return (idx & kLearntTag) ? learntClauses[idx & kLearntMask]
                              : problemClauses[idx];
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE const Clause &getClause(uint32_t idx) const {
    return (idx & kLearntTag) ? learntClauses[idx & kLearntMask]
                              : problemClauses[idx];
  }

  /// Per-variable state. vars[i] corresponds to external variable i+1.
  /// Fields ordered for cache locality: hot fields (propagate/analyze) first.
  struct Variable {
    // Hot path fields - accessed together during propagate (12 bytes)
    int level = 0;            // Decision level where assigned
    int reason = kNoReason;   // Clause index that implied this assignment
    Assign assign = kUndef;   // Current assignment (HOT!)
    Assign modelVal = kUndef; // Saved model value after SAT
    int8_t polarity = 0;      // Preferred phase (0 or 1)
    int8_t seen = 0;          // Temporary mark for conflict analysis
    // Cold field - only accessed during VSIDS operations (8 bytes)
    double activity = 0.0; // VSIDS score
  };

  /// Result from Boolean Constraint Propagation.
  struct Conflict {
    int index = -1;      // -1 = no conflict, kBinary = binary clause, else idx
    int binLits[2] = {}; // Valid when isBinary()

    static constexpr int kBinary = -2;
    [[nodiscard]] bool isNone() const { return index == -1; }
    [[nodiscard]] bool isBinary() const { return index == kBinary; }
  };

  // Variable management
  void ensureVar(int v);

  // VSIDS activity bumping and decay
  void bumpVarActivity(int v);
  void decayVarActivity();
  void bumpClauseActivity(Clause &c);
  void decayClauseActivity();

  // Assignment and trail management
  [[nodiscard]] LLVM_ATTRIBUTE_ALWAYS_INLINE Assign evalLit(int enc) const;
  [[nodiscard]] LLVM_ATTRIBUTE_ALWAYS_INLINE int decisionLevel() const;
  bool enqueue(int lit, int reason);
  void newDecisionLevel();
  void backtrack(int level);

  // Boolean Constraint Propagation (unit propagation with two-literal watching)
  Conflict propagate();

  // 1UIP conflict analysis with self-subsumption minimization
  void analyze(const Conflict &confl, llvm::SmallVectorImpl<int> &outLearnt,
               int &outBackLevel);
  bool isRedundant(int v, unsigned levelMask, llvm::SmallVectorImpl<int> &stack,
                   llvm::SmallVectorImpl<int> &toClear);
  void cleanupRedundancyCheck(int top, llvm::SmallVectorImpl<int> &toClear);

  // Clause construction and watch list maintenance
  void addWatchPair(int lit0, int lit1, uint32_t clauseIdx);
  void addBinaryWatch(int lit0, int lit1);
  void commitClause();
  void recordLearnt(llvm::SmallVectorImpl<int> &lits);

  // Periodically remove low-activity learned clauses
  void reduceLearnts();

  // VSIDS decision heuristic
  [[nodiscard]] int pickBranchVar();

  // Luby restart sequence
  [[nodiscard]] static double luby(double y, int i);

  // Main search loop
  Result search(int64_t confBudget);
  Result solveImpl(int64_t confLimit);

  // Rebuild all watch lists from clause storage (used by rollback).
  // ABC does this by iterating its arena; we iterate our clause vectors.
  void rebuildWatchLists();

  // Bookmark pivot points for incremental rollback (ABC-style)
  struct Bookmark {
    int numVars;
    int numClauses;
    int numProblemClauses;     // problemClauses.size()
    int numBinaryClauses;      // binaryClauses.size()
    int numLearntClauses;      // learntClauses.size()
    int numLearntBinaries;     // learntBinaryClauses.size()
    int trailSize;
    int propagHead;
    double varActInc;
    double claActInc;
  };
  std::optional<Bookmark> bookmark_;

  // Solver state
  int numVars = 0;
  int numClauses = 0;
  int propagHead = 0;     // Trail position for next propagation
  int rootLevel = 0;      // Decision level of assumptions
  bool ok = true;         // False if UNSAT detected during clause addition
  bool heapNeedsRebuild = false; // Lazy heap rebuild after rollback
  double varActInc = 1.0; // VSIDS activity increment
  double claActInc = 1.0; // Clause activity increment

  llvm::SmallVector<Variable, 0> vars; // vars[i] = external variable i+1
  llvm::SmallVector<llvm::SmallVector<WatchEntry, 4>, 0> watchLists;
  llvm::SmallVector<int> trail, trailLimits; // Assignment trail and levels
  IndexedMaxHeap vsidsHeap;
  llvm::SmallVector<Clause, 4> problemClauses, learntClauses;
  // Binary clauses stored separately (internal encoded literal pairs) so that
  // watch lists can be rebuilt from clause storage on rollback. Without this,
  // binary clauses exist only in watch lists and are lost when lists are cleared.
  llvm::SmallVector<std::pair<int, int>, 0> binaryClauses;
  llvm::SmallVector<std::pair<int, int>, 0> learntBinaryClauses;
  llvm::SmallVector<int> clauseBuf, assumptionBuf; // Input accumulators

  // Reusable workspace for analyze() - avoids allocations in hot path
  llvm::SmallVector<int> analyzeStack, analyzeToClear;
};

//===----------------------------------------------------------------------===//
// Inline implementations of hot-path methods
//===----------------------------------------------------------------------===//

inline MiniSATSolver::Assign MiniSATSolver::evalLit(int enc) const {
  Assign a = vars[litVar(enc)].assign;
  if (a == kUndef)
    return kUndef;
  return static_cast<Assign>((enc & 1) ? -a : a);
}

inline int MiniSATSolver::decisionLevel() const {
  return static_cast<int>(trailLimits.size());
}

} // namespace circt

#endif // CIRCT_SUPPORT_SATSOLVER_H
