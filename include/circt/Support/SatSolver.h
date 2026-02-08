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
#include <cassert>
#include <cstdint>

namespace circt {

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

  void grow(int n) {
    if (n > static_cast<int>(pos.size()))
      pos.resize(n, kAbsent);
  }

  bool empty() const { return heap.empty(); }

  bool contains(int v) const {
    return v >= 0 && v < static_cast<int>(pos.size()) && pos[v] != kAbsent;
  }

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

  llvm::SmallVector<int, 0> heap;
  llvm::SmallVector<int, 0> pos;
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
  void add(int lit);

  /// Register an assumption for the next solve() call.
  void assume(int lit);

  /// Solve under current assumptions. Returns kSAT, kUNSAT, or kUNKNOWN.
  Result solve(int64_t confLimit = -1);

  /// Model value for variable `v` after kSAT. Returns v (true) or -v (false).
  int val(int v) const;

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

  /// Learned clause reduction thresholds.
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
    llvm::SmallVector<int, 4> lits;
  };

  Clause &getClause(uint32_t idx);
  const Clause &getClause(uint32_t idx) const;

  // ---- Per-variable state ----

  /// All data for one SAT variable, grouped for clarity.
  /// `vars` is 0-indexed: external variable v (1-indexed) maps to vars[v-1].
  /// Internal litVar() already returns 0-indexed, so vars[litVar(enc)] works
  /// directly — no +1 offset needed anywhere.
  struct Variable {
    double activity = 0.0;
    int level = 0;
    int reason = kNoReason;
    Assign assign = kUndef;
    Assign modelVal = kUndef;
    int8_t polarity = 0;
    int8_t seen = 0;
  };

  // ---- Variable management ----
  void ensureVar(int v);

  // ---- VSIDS activity ----
  void bumpVarActivity(int v);
  void decayVarActivity();
  void bumpClauseActivity(Clause &c);
  void decayClauseActivity();

  // ---- Trail / Assignment ----
  Assign evalLit(int enc) const;
  int decisionLevel() const;
  bool enqueue(int p, int reason);
  void newDecisionLevel();
  void backtrack(int level);

  // ---- Boolean Constraint Propagation ----
  int propagate();

  // ---- Conflict Analysis (1UIP) ----
  void analyze(int conflIdx, llvm::SmallVectorImpl<int> &outLearnt,
               int &outBackLevel);
  bool isRedundant(int v, unsigned levelMask);
  bool cleanupRedundancyCheck(int top);

  // ---- Clause construction ----
  void addWatchPair(int lit0, int lit1, uint32_t ci);
  void addBinaryWatch(int lit0, int lit1);
  void commitClause();
  void recordLearnt(llvm::SmallVectorImpl<int> &lits);

  // ---- Learned clause reduction ----
  void reduceLearnts();

  // ---- Decision ----
  int pickBranchVar();

  // ---- Luby restart sequence ----
  static double luby(double y, int i);

  // ---- Main search loop ----
  Result search(int64_t confBudget);
  Result solveImpl(int64_t confLimit);

  // ---- Solver state ----
  int numVars = 0;
  int numClauses = 0;
  int propagHead = 0;
  int rootLevel = 0;
  bool ok = true;
  double varActInc = 1.0;
  double claActInc = 1.0;

  /// Per-variable data, 0-indexed. vars[i] is external variable i+1.
  llvm::SmallVector<Variable, 0> vars;

  // Per-literal watch lists (indexed by encoded literal).
  llvm::SmallVector<llvm::SmallVector<WatchEntry, 4>, 0> watchLists;

  // Decision trail.
  llvm::SmallVector<int> trail;
  llvm::SmallVector<int> trailLimits;

  // VSIDS decision heap.
  IndexedMaxHeap vsidsHeap;

  // Clause pools.
  llvm::SmallVector<Clause, 0> problemClauses;
  llvm::SmallVector<Clause, 0> learntClauses;

  // Temporaries.
  llvm::SmallVector<int> clauseBuf;
  llvm::SmallVector<int> assumptionBuf;
  llvm::SmallVector<int> tmpLits;
  int binConflLits[2] = {};
  llvm::SmallVector<int> analyzeStack;
  llvm::SmallVector<int> clearList;
};

} // namespace circt

#endif // CIRCT_SUPPORT_SATSOLVER_H
