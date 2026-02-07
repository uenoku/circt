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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
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

/// Lightweight CDCL SAT solver using LLVM data structures.
///
/// External API uses 1-indexed IPASIR-style variables: positive literal = v,
/// negative literal = -v. Internally, literals are encoded as:
///   pos(v) = 2*(v-1), neg(v) = 2*(v-1)+1
/// so that negation is XOR 1, and the variable index is lit >> 1.
class MiniSATSolver {
public:
  enum Result : int { kSAT = 10, kUNSAT = 20, kUNKNOWN = 0 };

  /// Add a literal to the current clause being built. 0 terminates.
  void add(int lit);

  void assume(int lit);

  [[nodiscard]] Result solve(int64_t confLimit = -1);

  /// Query model value after kSAT. Returns v (true) or -v (false).
  [[nodiscard]] int val(int v) const;

  [[nodiscard]] int getNumClauses() const { return numClauses; }

  /// Save current state for later rollback. Must be called at decision level 0.
  /// Enables incremental solving by allowing the solver to return to this state
  /// while preserving learned clauses and VSIDS activity across rollbacks.
  void bookmark();

  /// Restore to last bookmarked state. Removes variables and problem clauses
  /// added after bookmark, but preserves learned clauses and VSIDS activity.
  /// This enables sharing knowledge across multiple SAT queries.
  void rollback();

  /// Check if a bookmark exists.
  [[nodiscard]] bool hasBookmark() const { return bookmark_.has_value(); }

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
    int level = 0;            // Decision level where assigned
    int reason = kNoReason;   // Clause index that implied this assignment
    Assign assign = kUndef;   // Current assignment
    Assign modelVal = kUndef; // Saved model value after SAT
    int8_t polarity = 0;      // Preferred phase (0 or 1)
    int8_t seen = 0;          // Temporary mark for conflict analysis
    double activity = 0.0;    // VSIDS score
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

  // Bookmark structure for incremental solving
  struct Bookmark {
    int numVars;                                 // Variables to preserve
    int numProblemClauses;                       // Problem clauses to preserve
    int trailSize;                               // Trail state (usually 0)
    int propagHead;                              // Propagation queue position
    llvm::SmallVector<size_t, 0> watchListSizes; // Watch list sizes per literal
    double varActInc;                            // VSIDS decay state
    double claActInc;                            // Clause activity decay state
  };

  std::optional<Bookmark> bookmark_;

  // Solver state
  int numVars = 0;
  int numClauses = 0;
  int propagHead = 0;     // Trail position for next propagation
  int rootLevel = 0;      // Decision level of assumptions
  bool ok = true;         // False if UNSAT detected during clause addition
  double varActInc = 1.0; // VSIDS activity increment
  double claActInc = 1.0; // Clause activity increment

  llvm::SmallVector<Variable, 0> vars; // vars[i] = external variable i+1
  llvm::SmallVector<llvm::SmallVector<WatchEntry, 4>, 0> watchLists;
  llvm::SmallVector<int> trail, trailLimits; // Assignment trail and levels
  IndexedMaxHeap vsidsHeap;
  llvm::SmallVector<Clause, 4> problemClauses, learntClauses;
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
