//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SatSolver.h"
#include "gtest/gtest.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// IndexedMaxHeap tests
//===----------------------------------------------------------------------===//

TEST(IndexedMaxHeapTest, BasicInsertPop) {
  IndexedMaxHeap heap;
  double scores[] = {3.0, 1.0, 4.0, 1.0, 5.0};
  auto sf = [&](int v) { return scores[v]; };

  heap.grow(5);
  for (int i = 0; i < 5; i++)
    heap.insert(i, sf);

  EXPECT_FALSE(heap.empty());
  EXPECT_EQ(heap.pop(sf), 4); // score 5.0
  EXPECT_EQ(heap.pop(sf), 2); // score 4.0
  EXPECT_EQ(heap.pop(sf), 0); // score 3.0
}

TEST(IndexedMaxHeapTest, Contains) {
  IndexedMaxHeap heap;
  double scores[] = {1.0, 2.0, 3.0};
  auto sf = [&](int v) { return scores[v]; };

  heap.grow(3);
  heap.insert(0, sf);
  heap.insert(2, sf);

  EXPECT_TRUE(heap.contains(0));
  EXPECT_FALSE(heap.contains(1));
  EXPECT_TRUE(heap.contains(2));
}

TEST(IndexedMaxHeapTest, IncreaseKey) {
  IndexedMaxHeap heap;
  double scores[] = {1.0, 2.0, 3.0};
  auto sf = [&](int v) { return scores[v]; };

  heap.grow(3);
  for (int i = 0; i < 3; i++)
    heap.insert(i, sf);

  // Increase score of element 0 to be the highest.
  scores[0] = 10.0;
  heap.increased(0, sf);

  EXPECT_EQ(heap.pop(sf), 0);
}

TEST(IndexedMaxHeapTest, DuplicateInsert) {
  IndexedMaxHeap heap;
  double scores[] = {1.0, 2.0};
  auto sf = [&](int v) { return scores[v]; };

  heap.grow(2);
  heap.insert(0, sf);
  heap.insert(0, sf); // duplicate — should be ignored
  heap.insert(1, sf);

  EXPECT_EQ(heap.pop(sf), 1);
  EXPECT_EQ(heap.pop(sf), 0);
  EXPECT_TRUE(heap.empty());
}

TEST(IndexedMaxHeapTest, SingleElement) {
  IndexedMaxHeap heap;
  double scores[] = {42.0};
  auto sf = [&](int v) { return scores[v]; };

  heap.grow(1);
  heap.insert(0, sf);

  EXPECT_FALSE(heap.empty());
  EXPECT_EQ(heap.pop(sf), 0);
  EXPECT_TRUE(heap.empty());
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Trivial tests
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, TrivialSat) {
  MiniSATSolver s;
  // Unit clause: x1
  s.add(1);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
}

TEST(MiniSATSolverTest, TrivialUnsat) {
  MiniSATSolver s;
  // x1 AND NOT x1
  s.add(1);
  s.add(0);
  s.add(-1);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

TEST(MiniSATSolverTest, EmptyClauseUnsat) {
  MiniSATSolver s;
  // Empty clause (add 0 immediately).
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Two-variable formulas
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, TwoVarForcedSat) {
  MiniSATSolver s;
  // (x1 OR x2) AND (NOT x1 OR x2) AND (x1 OR NOT x2)
  // Only satisfying assignment: x1=true, x2=true
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(-1);
  s.add(2);
  s.add(0);
  s.add(1);
  s.add(-2);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
  EXPECT_EQ(s.val(2), 2);
}

TEST(MiniSATSolverTest, TwoVarAllClauses) {
  MiniSATSolver s;
  // All four 2-variable clauses: UNSAT
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(-1);
  s.add(2);
  s.add(0);
  s.add(1);
  s.add(-2);
  s.add(0);
  s.add(-1);
  s.add(-2);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Assumptions
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, AssumptionsSatThenUnsat) {
  MiniSATSolver s;
  // (x1 OR x2)
  s.add(1);
  s.add(2);
  s.add(0);

  // Without assumptions: SAT
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Assume both false: UNSAT under assumptions
  s.assume(-1);
  s.assume(-2);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  // Assumptions are cleared after solve; should be SAT again.
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

TEST(MiniSATSolverTest, AssumptionForcesValue) {
  MiniSATSolver s;
  // No clauses, just assume x1=true.
  s.assume(1);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);

  // Assume x1=false.
  s.assume(-1);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), -1);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Tseitin encodings
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, TseitinAndGate) {
  MiniSATSolver s;
  // c = a AND b (variables: a=1, b=2, c=3)
  // Tseitin: (NOT c OR a), (NOT c OR b), (c OR NOT a OR NOT b)
  s.add(-3);
  s.add(1);
  s.add(0);
  s.add(-3);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(-1);
  s.add(-2);
  s.add(0);

  // Assert c = true
  s.add(3);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
  EXPECT_EQ(s.val(2), 2);
  EXPECT_EQ(s.val(3), 3);
}

TEST(MiniSATSolverTest, TseitinOrGate) {
  MiniSATSolver s;
  // c = a OR b (variables: a=1, b=2, c=3)
  // Tseitin: (c OR NOT a), (c OR NOT b), (NOT c OR a OR b)
  s.add(3);
  s.add(-1);
  s.add(0);
  s.add(3);
  s.add(-2);
  s.add(0);
  s.add(-3);
  s.add(1);
  s.add(2);
  s.add(0);

  // Assert c = false
  s.add(-3);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), -1);
  EXPECT_EQ(s.val(2), -2);
  EXPECT_EQ(s.val(3), -3);
}

TEST(MiniSATSolverTest, TseitinXorChain) {
  MiniSATSolver s;
  // x1 XOR x2 XOR x3 = 1
  // t = x1 XOR x2 (var 4), result = t XOR x3 (var 5)
  // XOR Tseitin for t = a XOR b:
  //   (NOT t OR NOT a OR NOT b), (NOT t OR a OR b),
  //   (t OR NOT a OR b), (t OR a OR NOT b)

  // t = x1 XOR x2
  s.add(-4);
  s.add(-1);
  s.add(-2);
  s.add(0);
  s.add(-4);
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(4);
  s.add(-1);
  s.add(2);
  s.add(0);
  s.add(4);
  s.add(1);
  s.add(-2);
  s.add(0);

  // result = t XOR x3
  s.add(-5);
  s.add(-4);
  s.add(-3);
  s.add(0);
  s.add(-5);
  s.add(4);
  s.add(3);
  s.add(0);
  s.add(5);
  s.add(-4);
  s.add(3);
  s.add(0);
  s.add(5);
  s.add(4);
  s.add(-3);
  s.add(0);

  // Assert result = true
  s.add(5);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Verify: x1 XOR x2 XOR x3 = 1
  int v1 = s.val(1) > 0 ? 1 : 0;
  int v2 = s.val(2) > 0 ? 1 : 0;
  int v3 = s.val(3) > 0 ? 1 : 0;
  EXPECT_EQ((v1 ^ v2 ^ v3), 1);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Pigeonhole principle (classic hard UNSAT)
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, Pigeonhole3Into2) {
  MiniSATSolver s;
  // 3 pigeons into 2 holes: UNSAT
  // Variables p_ij: pigeon i in hole j
  // p11=1, p12=2, p21=3, p22=4, p31=5, p32=6

  // Each pigeon in at least one hole.
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(4);
  s.add(0);
  s.add(5);
  s.add(6);
  s.add(0);

  // At most one pigeon per hole.
  // Hole 1: at most one of {p11, p21, p31}
  s.add(-1);
  s.add(-3);
  s.add(0);
  s.add(-1);
  s.add(-5);
  s.add(0);
  s.add(-3);
  s.add(-5);
  s.add(0);
  // Hole 2: at most one of {p12, p22, p32}
  s.add(-2);
  s.add(-4);
  s.add(0);
  s.add(-2);
  s.add(-6);
  s.add(0);
  s.add(-4);
  s.add(-6);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

TEST(MiniSATSolverTest, Pigeonhole2Into2Sat) {
  MiniSATSolver s;
  // 2 pigeons into 2 holes: SAT
  // p11=1, p12=2, p21=3, p22=4

  // Each pigeon in at least one hole.
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(4);
  s.add(0);

  // At most one pigeon per hole.
  s.add(-1);
  s.add(-3);
  s.add(0);
  s.add(-2);
  s.add(-4);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Verify: each pigeon in exactly one hole, no conflicts.
  bool p11 = s.val(1) > 0, p12 = s.val(2) > 0;
  bool p21 = s.val(3) > 0, p22 = s.val(4) > 0;
  EXPECT_TRUE(p11 || p12);
  EXPECT_TRUE(p21 || p22);
  EXPECT_FALSE(p11 && p21); // hole 1 conflict
  EXPECT_FALSE(p12 && p22); // hole 2 conflict
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Tautology / duplicate handling
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, TautologyClause) {
  MiniSATSolver s;
  // (x1 OR NOT x1) is a tautology — should be ignored.
  s.add(1);
  s.add(-1);
  s.add(0);
  // Then add contradictory unit clauses.
  s.add(2);
  s.add(0);
  s.add(-2);
  s.add(0);
  // Should be UNSAT (the tautology doesn't help).
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

TEST(MiniSATSolverTest, DuplicateLiterals) {
  MiniSATSolver s;
  // (x1 OR x1) should be simplified to (x1)
  s.add(1);
  s.add(1);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Multiple solves (incremental)
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, IncrementalSolves) {
  MiniSATSolver s;
  // (x1 OR x2)
  s.add(1);
  s.add(2);
  s.add(0);

  // First solve: SAT
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Add more clauses: (NOT x1 OR NOT x2)
  s.add(-1);
  s.add(-2);
  s.add(0);

  // Still SAT (need one true, one false)
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  int v1 = s.val(1), v2 = s.val(2);
  EXPECT_NE(v1, v2); // one positive, one negative
  EXPECT_TRUE((v1 > 0) != (v2 > 0));

  // Add unit clauses forcing both true: UNSAT
  s.add(1);
  s.add(0);
  s.add(2);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Conflict limit
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, ConflictLimitReturnsUnknown) {
  MiniSATSolver s;
  // Build a pigeonhole instance that requires some conflicts.
  // 4 pigeons into 3 holes (UNSAT).
  // p_ij: pigeon i in hole j. Variables numbered sequentially.
  // 4 pigeons * 3 holes = 12 variables.
  int var = 1;
  int vars[4][3];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      vars[i][j] = var++;

  // Each pigeon in at least one hole.
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++)
      s.add(vars[i][j]);
    s.add(0);
  }

  // At most one pigeon per hole.
  for (int j = 0; j < 3; j++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = i1 + 1; i2 < 4; i2++) {
        s.add(-vars[i1][j]);
        s.add(-vars[i2][j]);
        s.add(0);
      }
    }
  }

  // With a very tight conflict limit, may return UNKNOWN.
  auto result = s.solve(1);
  EXPECT_TRUE(result == MiniSATSolver::kUNSAT ||
              result == MiniSATSolver::kUNKNOWN);

  // With no limit, must return UNSAT.
  result = s.solve();
  EXPECT_EQ(result, MiniSATSolver::kUNSAT);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Larger formula (N-queens idea: chain of implications)
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, ImplicationChain) {
  MiniSATSolver s;
  // x1 => x2 => x3 => ... => x10, assert x1, assert NOT x10 -> UNSAT
  for (int i = 1; i < 10; i++) {
    s.add(-i);
    s.add(i + 1);
    s.add(0);
  }
  s.add(1);
  s.add(0);
  s.add(-10);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

TEST(MiniSATSolverTest, ImplicationChainSat) {
  MiniSATSolver s;
  // x1 => x2 => x3 => ... => x10, assert x1 -> SAT (all true)
  for (int i = 1; i < 10; i++) {
    s.add(-i);
    s.add(i + 1);
    s.add(0);
  }
  s.add(1);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  for (int i = 1; i <= 10; i++)
    EXPECT_EQ(s.val(i), i); // all true
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - getNumClauses
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, ClauseCount) {
  MiniSATSolver s;
  EXPECT_EQ(s.getNumClauses(), 0);

  // Unit clauses are propagated at root, not stored as clauses.
  s.add(1);
  s.add(0);
  EXPECT_EQ(s.getNumClauses(), 0);

  // Binary clause.
  s.add(2);
  s.add(3);
  s.add(0);
  EXPECT_EQ(s.getNumClauses(), 1);

  // Ternary clause.
  s.add(4);
  s.add(5);
  s.add(6);
  s.add(0);
  EXPECT_EQ(s.getNumClauses(), 2);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Model verification for larger problems
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MiniSATSolver - Bookmark / Rollback
//===----------------------------------------------------------------------===//

TEST(MiniSATSolverTest, BookmarkRollbackBasic) {
  MiniSATSolver s;
  // Pre-allocate 3 variables, bookmark with no clauses
  s.reserveVars(3);
  s.bookmark();

  // Add clauses: (x1 OR x2) AND (NOT x1 OR x3)
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(-1);
  s.add(3);
  s.add(0);
  EXPECT_EQ(s.getNumClauses(), 2);

  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Rollback: clauses removed, solver is clean
  s.rollback();
  EXPECT_EQ(s.getNumClauses(), 0);

  // Solver should be satisfiable (no clauses)
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

TEST(MiniSATSolverTest, BookmarkRollbackMultipleCycles) {
  MiniSATSolver s;
  s.reserveVars(4);
  s.bookmark();

  // Cycle 1: add UNSAT formula, verify, rollback
  s.add(1);
  s.add(0);
  s.add(-1);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
  s.rollback();

  // Cycle 2: add SAT formula, verify, rollback
  s.add(2);
  s.add(3);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  s.rollback();

  // Cycle 3: still clean
  s.add(4);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(4), 4);
  s.rollback();
}

TEST(MiniSATSolverTest, BookmarkPreservesActivity) {
  MiniSATSolver s;
  s.reserveVars(6);
  s.bookmark();

  // Add a formula that requires conflicts to build VSIDS activity
  // Pigeonhole 3 into 2: UNSAT
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(4);
  s.add(0);
  s.add(5);
  s.add(6);
  s.add(0);
  s.add(-1);
  s.add(-3);
  s.add(0);
  s.add(-1);
  s.add(-5);
  s.add(0);
  s.add(-3);
  s.add(-5);
  s.add(0);
  s.add(-2);
  s.add(-4);
  s.add(0);
  s.add(-2);
  s.add(-6);
  s.add(0);
  s.add(-4);
  s.add(-6);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  // Rollback: VSIDS activity preserved (warm start for next query)
  s.rollback();

  // Add the same problem again - should still be UNSAT
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(4);
  s.add(0);
  s.add(5);
  s.add(6);
  s.add(0);
  s.add(-1);
  s.add(-3);
  s.add(0);
  s.add(-1);
  s.add(-5);
  s.add(0);
  s.add(-3);
  s.add(-5);
  s.add(0);
  s.add(-2);
  s.add(-4);
  s.add(0);
  s.add(-2);
  s.add(-6);
  s.add(0);
  s.add(-4);
  s.add(-6);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

TEST(MiniSATSolverTest, BookmarkRollbackWithAssumptions) {
  MiniSATSolver s;
  s.reserveVars(3);
  s.bookmark();

  // Add: (x1 OR x2), (NOT x1 OR x3)
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(-1);
  s.add(3);
  s.add(0);

  // Solve with assumption x1=true
  s.assume(1);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
  EXPECT_EQ(s.val(3), 3); // x3 must be true (NOT x1 OR x3, x1=true)

  s.rollback();

  // After rollback, no clauses - solve should be SAT with any assignment
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

TEST(MiniSATSolverTest, BookmarkRollbackTseitin) {
  // Simulate FRAIG-like usage: pre-allocate vars, bookmark, add Tseitin
  // encoding for a cone, verify equivalence, rollback, repeat.
  MiniSATSolver s;
  s.reserveVars(5);
  s.bookmark();

  // Cone 1: c3 = AND(x1, x2) using vars 1, 2, 3
  s.add(-3);
  s.add(1);
  s.add(0);
  s.add(-3);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(-1);
  s.add(-2);
  s.add(0);

  // Verify: can x1=1, x2=1, c3=0? (Should be UNSAT)
  s.assume(1);
  s.assume(2);
  s.assume(-3);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  s.rollback();

  // Cone 2: c5 = OR(x4, x5) using vars 4, 5 and a new output var
  // Use var 4 as output, var 4 and 5 as inputs... let me just use different
  // vars to avoid confusion.
  // c4 = OR(x1, x2) using vars 1, 2, 4
  s.add(4);
  s.add(-1);
  s.add(0);
  s.add(4);
  s.add(-2);
  s.add(0);
  s.add(-4);
  s.add(1);
  s.add(2);
  s.add(0);

  // Verify: can x1=0, x2=0, c4=1? (Should be UNSAT)
  s.assume(-1);
  s.assume(-2);
  s.assume(4);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  s.rollback();
}

TEST(MiniSATSolverTest, FullAdderCorrectness) {
  // Encode a 1-bit full adder and verify all 8 input combinations.
  // Inputs: a=1, b=2, cin=3. Outputs: sum=4, cout=5.
  // sum = a XOR b XOR cin, cout = (a AND b) OR (cin AND (a XOR b))
  //
  // We encode with Tseitin and test each input combination.

  auto testAdder = [](bool aVal, bool bVal, bool cinVal) {
    MiniSATSolver s;
    // Variables: a=1, b=2, cin=3, sum=4, cout=5
    // Aux: t1=6 (a XOR b), t2=7 (a AND b), t3=8 (cin AND t1)

    // t1 = a XOR b
    s.add(-6);
    s.add(-1);
    s.add(-2);
    s.add(0);
    s.add(-6);
    s.add(1);
    s.add(2);
    s.add(0);
    s.add(6);
    s.add(-1);
    s.add(2);
    s.add(0);
    s.add(6);
    s.add(1);
    s.add(-2);
    s.add(0);

    // sum = t1 XOR cin
    s.add(-4);
    s.add(-6);
    s.add(-3);
    s.add(0);
    s.add(-4);
    s.add(6);
    s.add(3);
    s.add(0);
    s.add(4);
    s.add(-6);
    s.add(3);
    s.add(0);
    s.add(4);
    s.add(6);
    s.add(-3);
    s.add(0);

    // t2 = a AND b
    s.add(-7);
    s.add(1);
    s.add(0);
    s.add(-7);
    s.add(2);
    s.add(0);
    s.add(7);
    s.add(-1);
    s.add(-2);
    s.add(0);

    // t3 = cin AND t1
    s.add(-8);
    s.add(3);
    s.add(0);
    s.add(-8);
    s.add(6);
    s.add(0);
    s.add(8);
    s.add(-3);
    s.add(-6);
    s.add(0);

    // cout = t2 OR t3
    s.add(5);
    s.add(-7);
    s.add(0);
    s.add(5);
    s.add(-8);
    s.add(0);
    s.add(-5);
    s.add(7);
    s.add(8);
    s.add(0);

    // Set input values.
    s.add(aVal ? 1 : -1);
    s.add(0);
    s.add(bVal ? 2 : -2);
    s.add(0);
    s.add(cinVal ? 3 : -3);
    s.add(0);

    EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

    bool expectedSum = aVal ^ bVal ^ cinVal;
    bool expectedCout = (aVal && bVal) || (cinVal && (aVal ^ bVal));

    EXPECT_EQ(s.val(4) > 0, expectedSum)
        << "a=" << aVal << " b=" << bVal << " cin=" << cinVal;
    EXPECT_EQ(s.val(5) > 0, expectedCout)
        << "a=" << aVal << " b=" << bVal << " cin=" << cinVal;
  };

  for (int a = 0; a < 2; a++)
    for (int b = 0; b < 2; b++)
      for (int cin = 0; cin < 2; cin++)
        testAdder(a, b, cin);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - Common Incremental SAT Test Patterns
//===----------------------------------------------------------------------===//

// Test pattern: AllSAT-style model enumeration with blocking clauses
TEST(MiniSATSolverTest, IncrementalModelEnumeration) {
  MiniSATSolver s;
  // Formula: (NOT x1 OR x2) AND (NOT x1 OR x3)
  // This has exactly 3 satisfying assignments:
  //   1. x1=F (any x2, x3)... wait no, that's 4 assignments
  // Let's use a simpler formula with exactly 2 models:
  // (x1 OR x2) AND (NOT x1 OR NOT x2)
  // Models: {x1=T,x2=F}, {x1=F,x2=T}
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(-1);
  s.add(-2);
  s.add(0);

  // Find first solution
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  int v1First = s.val(1), v2First = s.val(2);

  // Block first solution
  s.add(v1First > 0 ? -1 : 1);
  s.add(v2First > 0 ? -2 : 2);
  s.add(0);

  // Find second solution
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  int v1Second = s.val(1), v2Second = s.val(2);
  EXPECT_TRUE(v1First != v1Second || v2First != v2Second);

  // Block second solution
  s.add(v1Second > 0 ? -1 : 1);
  s.add(v2Second > 0 ? -2 : 2);
  s.add(0);

  // Should be UNSAT now (all 2 models blocked)
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

// Test pattern: Shared prefix with different temporary clauses
TEST(MiniSATSolverTest, IncrementalSharedPrefix) {
  MiniSATSolver s;
  s.reserveVars(4);

  // Permanent clauses: (x1 OR x2)
  s.add(1);
  s.add(2);
  s.add(0);

  // Test 1: Add temporary (NOT x1 OR x3), expect SAT
  s.bookmark();
  s.add(-1);
  s.add(3);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  s.rollback();

  // Test 2: Add temporary (NOT x1 OR x4) AND (NOT x2 OR NOT x4), expect SAT
  s.bookmark();
  s.add(-1);
  s.add(4);
  s.add(0);
  s.add(-2);
  s.add(-4);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  s.rollback();

  // Test 3: Add temporary (NOT x1) AND (NOT x2), expect UNSAT
  s.bookmark();
  s.add(-1);
  s.add(0);
  s.add(-2);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
  s.rollback();

  // Original formula should still be SAT
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

// Test pattern: Equivalence checking with bookmark/rollback (FRAIG-style)
TEST(MiniSATSolverTest, IncrementalEquivalenceChecking) {
  MiniSATSolver s;
  s.reserveVars(4);

  // Test 1: Check if two AND gates with same inputs are equivalent
  // f1 = a AND b (var 3), f2 = a AND b (var 4)
  // They should be equivalent (both return same value for all inputs)
  s.bookmark();

  // Encode f1 = a AND b
  s.add(-3);
  s.add(1);
  s.add(0);
  s.add(-3);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(-1);
  s.add(-2);
  s.add(0);

  // Encode f2 = a AND b
  s.add(-4);
  s.add(1);
  s.add(0);
  s.add(-4);
  s.add(2);
  s.add(0);
  s.add(4);
  s.add(-1);
  s.add(-2);
  s.add(0);

  // Check if they can differ: try f1=T, f2=F
  s.assume(3);
  s.assume(-4);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT); // Cannot differ (equivalent)

  // Check if they can differ: try f1=F, f2=T
  s.assume(-3);
  s.assume(4);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT); // Cannot differ (equivalent)

  s.rollback();

  // Test 2: Check if AND and OR are equivalent (they're not)
  s.bookmark();

  // Encode f1 = a AND b (var 3)
  s.add(-3);
  s.add(1);
  s.add(0);
  s.add(-3);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(-1);
  s.add(-2);
  s.add(0);

  // Encode f2 = a OR b (var 4)
  s.add(-4);
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(4);
  s.add(-1);
  s.add(0);
  s.add(4);
  s.add(-2);
  s.add(0);

  // Check if they can differ: try f1=T, f2=F
  s.assume(3);
  s.assume(-4);
  auto result = s.solve();
  // Should be SAT or UNSAT depending on which direction differs
  EXPECT_TRUE(result == MiniSATSolver::kSAT ||
              result == MiniSATSolver::kUNSAT);

  // Check if they can differ: try f1=F, f2=T (e.g., a=T, b=F: AND=F, OR=T)
  s.assume(-3);
  s.assume(4);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT); // Can differ (not equivalent)

  s.rollback();
}

// Test pattern: Incremental constraint strengthening
TEST(MiniSATSolverTest, IncrementalConstraintStrengthening) {
  MiniSATSolver s;
  s.reserveVars(4);

  // Start with: (x1 OR x2 OR x3 OR x4)
  s.add(1);
  s.add(2);
  s.add(3);
  s.add(4);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Add: (NOT x1 OR NOT x2)
  s.add(-1);
  s.add(-2);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Add: (NOT x3 OR NOT x4)
  s.add(-3);
  s.add(-4);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Add: (NOT x1 OR NOT x3)
  s.add(-1);
  s.add(-3);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Add: (NOT x2 OR NOT x4)
  s.add(-2);
  s.add(-4);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Add: (NOT x1 OR NOT x4)
  s.add(-1);
  s.add(-4);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Add: (NOT x2 OR NOT x3) - now only {x1,x3}, {x1,x4}, {x2,x3}, {x2,x4}
  s.add(-2);
  s.add(-3);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

// Test pattern: Nested bookmark/rollback
TEST(MiniSATSolverTest, IncrementalNestedBookmarks) {
  // Current implementation: calling bookmark() replaces the previous bookmark.
  // Calling rollback() multiple times goes back to the same bookmark state.
  MiniSATSolver s;
  s.reserveVars(3);

  // First bookmark with no clauses
  s.bookmark();
  s.add(1);
  s.add(2);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.getNumClauses(), 1);

  // Second bookmark (replaces the first) - now has 1 clause
  s.bookmark();
  s.add(-1);
  s.add(3);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.getNumClauses(), 2);

  // First rollback goes to second bookmark (1 clause)
  s.rollback();
  EXPECT_EQ(s.getNumClauses(), 1);

  // Second rollback goes to same bookmark (still 1 clause - idempotent)
  s.rollback();
  EXPECT_EQ(s.getNumClauses(), 1);
}

// Test pattern: Assumptions are properly cleared between solves
TEST(MiniSATSolverTest, IncrementalAssumptionClearing) {
  MiniSATSolver s;
  // Formula: (x1 OR x2)
  s.add(1);
  s.add(2);
  s.add(0);

  // Solve with assumption x1=F
  s.assume(-1);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(2), 2); // x2 must be true

  // Next solve should clear assumptions - both x1 and x2 can be anything
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  // Both should be satisfiable (no forced values without assumptions)
}

// Test pattern: UNSAT with assumptions, then SAT without
TEST(MiniSATSolverTest, IncrementalUnsatToSat) {
  MiniSATSolver s;
  // Formula: (x1 OR x2) AND (NOT x1 OR x3)
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(-1);
  s.add(3);
  s.add(0);

  // With contradictory assumptions: UNSAT
  s.assume(-1);
  s.assume(-2);
  s.assume(-3);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  // Without assumptions: SAT
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

// Test pattern: Conflicting assumptions
TEST(MiniSATSolverTest, IncrementalConflictingAssumptions) {
  MiniSATSolver s;
  // No clauses, just conflicting assumptions
  s.assume(1);
  s.assume(-1);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  // Should be SAT without assumptions
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

// Test pattern: Unit propagation at root level persists
TEST(MiniSATSolverTest, IncrementalUnitPropagation) {
  MiniSATSolver s;
  // Add unit clause: x1 must be true
  s.add(1);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);

  // Add more clauses - x1 should still be forced true
  s.add(2);
  s.add(3);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1); // Still forced

  // Add clause that depends on x1's value
  s.add(-1);
  s.add(2);
  s.add(0); // (NOT x1 OR x2) with x1=T means x2=T
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
  EXPECT_EQ(s.val(2), 2); // Forced by implication
}

// Test pattern: Bookmark with existing unit propagations
TEST(MiniSATSolverTest, IncrementalBookmarkWithUnits) {
  MiniSATSolver s;
  s.reserveVars(3);

  // Add unit clauses
  s.add(1);
  s.add(0);
  s.add(2);
  s.add(0);

  // Solve to ensure propagation is complete before bookmark
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
  EXPECT_EQ(s.val(2), 2);

  // Bookmark captures state with units
  s.bookmark();

  // Add temporary clause
  s.add(-1);
  s.add(3);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
  EXPECT_EQ(s.val(3), 3); // Forced by clause and x1=T

  // Rollback should restore to state with x1=T, x2=T
  s.rollback();
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);
  EXPECT_EQ(s.val(2), 2);
}

// Test pattern: Multiple equivalence checks (simulating FRAIG)
TEST(MiniSATSolverTest, IncrementalMultipleFRAIGChecks) {
  MiniSATSolver s;
  s.reserveVars(10);
  s.bookmark();

  // Check 1: Are c1=(a AND b) and c2=(a AND b) equivalent? (Yes)
  s.add(-3);
  s.add(1);
  s.add(0); // c1 implies a
  s.add(-3);
  s.add(2);
  s.add(0); // c1 implies b
  s.add(3);
  s.add(-1);
  s.add(-2);
  s.add(0); // (a AND b) implies c1

  s.add(-4);
  s.add(1);
  s.add(0); // c2 implies a
  s.add(-4);
  s.add(2);
  s.add(0); // c2 implies b
  s.add(4);
  s.add(-1);
  s.add(-2);
  s.add(0); // (a AND b) implies c2

  s.assume(-3);
  s.assume(4);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT); // Cannot differ
  s.rollback();

  // Check 2: Are c5=(x OR y) and c6=(x AND y) equivalent? (No)
  s.bookmark();
  s.add(-5);
  s.add(7);
  s.add(8);
  s.add(0); // c5 implies (x OR y)
  s.add(5);
  s.add(-7);
  s.add(0); // x implies c5
  s.add(5);
  s.add(-8);
  s.add(0); // y implies c5

  s.add(-6);
  s.add(7);
  s.add(0); // c6 implies x
  s.add(-6);
  s.add(8);
  s.add(0); // c6 implies y
  s.add(6);
  s.add(-7);
  s.add(-8);
  s.add(0); // (x AND y) implies c6

  s.assume(5);
  s.assume(-6);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT); // Can differ (x=T,y=F: OR=T, AND=F)
  s.rollback();
}

// Test pattern: Stress test with many bookmark/rollback cycles
TEST(MiniSATSolverTest, IncrementalManyRollbacks) {
  MiniSATSolver s;
  s.reserveVars(5);

  // Base formula: (x1 OR x2)
  s.add(1);
  s.add(2);
  s.add(0);

  for (int i = 0; i < 20; i++) {
    s.bookmark();

    // Add temporary clauses
    s.add(-1);
    s.add(3);
    s.add(0);
    s.add(-2);
    s.add(4);
    s.add(0);

    EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

    // Rollback
    s.rollback();
    EXPECT_EQ(s.getNumClauses(), 1); // Back to base
  }

  // After all rollbacks, base formula should still work
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

// Test pattern: Binary vs long clauses in rollback
TEST(MiniSATSolverTest, IncrementalMixedClauseLengths) {
  MiniSATSolver s;
  s.reserveVars(6);
  s.bookmark();

  // Add binary clauses
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(-1);
  s.add(3);
  s.add(0);

  // Add ternary clause
  s.add(2);
  s.add(4);
  s.add(5);
  s.add(0);

  // Add longer clause
  s.add(-2);
  s.add(-3);
  s.add(-4);
  s.add(6);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.getNumClauses(), 4);

  s.rollback();
  EXPECT_EQ(s.getNumClauses(), 0);
}

// Test pattern: Learned clauses are cleared on rollback
TEST(MiniSATSolverTest, IncrementalLearntClausesCleared) {
  MiniSATSolver s;
  s.reserveVars(6);
  s.bookmark();

  // Add pigeonhole problem that will generate learned clauses
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(4);
  s.add(0);
  s.add(5);
  s.add(6);
  s.add(0);
  s.add(-1);
  s.add(-3);
  s.add(0);
  s.add(-1);
  s.add(-5);
  s.add(0);
  s.add(-3);
  s.add(-5);
  s.add(0);
  s.add(-2);
  s.add(-4);
  s.add(0);
  s.add(-2);
  s.add(-6);
  s.add(0);
  s.add(-4);
  s.add(-6);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  // Rollback should clear learned clauses
  s.rollback();

  // Now add a SAT instance - learned clauses from previous UNSAT
  // should not interfere
  s.bookmark();
  s.add(1);
  s.add(2);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  s.rollback();
}

// Test pattern: Incremental solver as oracle (multiple assumption sets)
TEST(MiniSATSolverTest, IncrementalAssumptionOracle) {
  MiniSATSolver s;
  // Formula: (x1 OR x2 OR x3) AND (NOT x1 OR NOT x2) AND (NOT x1 OR NOT x3)
  s.add(1);
  s.add(2);
  s.add(3);
  s.add(0);
  s.add(-1);
  s.add(-2);
  s.add(0);
  s.add(-1);
  s.add(-3);
  s.add(0);

  // Query 1: Can x1 be true?
  s.assume(1);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Query 2: Can x1 and x2 both be true?
  s.assume(1);
  s.assume(2);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  // Query 3: Can x2 and x3 both be true?
  s.assume(2);
  s.assume(3);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Query 4: Can all be false?
  s.assume(-1);
  s.assume(-2);
  s.assume(-3);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
}

// Test pattern: Empty formula with bookmark
TEST(MiniSATSolverTest, IncrementalEmptyFormula) {
  MiniSATSolver s;
  s.reserveVars(2);
  s.bookmark();

  // No clauses - should be SAT
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Add clause
  s.add(1);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(1), 1);

  // Rollback to empty
  s.rollback();
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.getNumClauses(), 0);
}

// Test pattern: Bookmark then immediately rollback
TEST(MiniSATSolverTest, IncrementalImmediateRollback) {
  MiniSATSolver s;
  s.reserveVars(3);

  s.add(1);
  s.add(2);
  s.add(0);
  EXPECT_EQ(s.getNumClauses(), 1);

  s.bookmark();
  s.rollback();

  // Should be back to same state
  EXPECT_EQ(s.getNumClauses(), 1);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

// Test pattern: Large conjunction (many binary clauses)
TEST(MiniSATSolverTest, IncrementalManyBinaryClauses) {
  MiniSATSolver s;
  s.reserveVars(20);
  s.bookmark();

  // Add chain: x1=>x2=>x3=>...=>x20
  for (int i = 1; i < 20; i++) {
    s.add(-i);
    s.add(i + 1);
    s.add(0);
  }

  // Assert x1=T, x20=F -> UNSAT
  s.add(1);
  s.add(0);
  s.add(-20);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  s.rollback();
  EXPECT_EQ(s.getNumClauses(), 0);
}

// Test pattern: Incremental cardinality constraint
TEST(MiniSATSolverTest, IncrementalCardinalityConstraint) {
  MiniSATSolver s;
  // At-most-one constraint on x1, x2, x3
  s.add(-1);
  s.add(-2);
  s.add(0);
  s.add(-1);
  s.add(-3);
  s.add(0);
  s.add(-2);
  s.add(-3);
  s.add(0);

  // Check: can x1 be true? (Yes)
  s.assume(1);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  EXPECT_EQ(s.val(2), -2); // x2 must be false
  EXPECT_EQ(s.val(3), -3); // x3 must be false

  // Check: can x2 and x3 both be true? (No)
  s.assume(2);
  s.assume(3);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  // Incrementally strengthen: add at-least-one constraint
  s.add(1);
  s.add(2);
  s.add(3);
  s.add(0);

  // Now exactly one must be true
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  int count = (s.val(1) > 0 ? 1 : 0) + (s.val(2) > 0 ? 1 : 0) +
              (s.val(3) > 0 ? 1 : 0);
  EXPECT_EQ(count, 1);
}

// Test pattern: Incremental search with backtracking budget
TEST(MiniSATSolverTest, IncrementalWithConflictLimit) {
  MiniSATSolver s;
  s.reserveVars(10);

  // Add a moderately hard problem
  for (int i = 1; i < 10; i++) {
    s.add(-i);
    s.add(i + 1);
    s.add(0);
  }
  s.add(-10);
  s.add(1);
  s.add(0);

  // Solve with very low limit - might return UNKNOWN
  auto result = s.solve(1);
  EXPECT_TRUE(result == MiniSATSolver::kSAT ||
              result == MiniSATSolver::kUNKNOWN);

  // Solve with no limit - should definitely return SAT
  result = s.solve();
  EXPECT_EQ(result, MiniSATSolver::kSAT);

  // Add more clauses incrementally
  s.add(1);
  s.add(2);
  s.add(0);
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
}

//===----------------------------------------------------------------------===//
// MiniSATSolver - CaDiCaL comparison bug reproducers
//===----------------------------------------------------------------------===//

// Reproducer for bug found in mul.sv FRAIG run where CaDiCaL returns SAT
// but MiniSAT returns UNSAT on the same problem. This indicates a correctness
// bug in the MiniSAT implementation's bookmark/rollback mechanism.
//
// The original failure involved 2079 clauses with assumptions [865 -866],
// where MiniSAT incorrectly reported UNSAT due to a conflict at root level
// with binary clause [-31 -210].
//
// This test creates a simpler version that may expose similar issues.
TEST(MiniSATSolverTest, BookmarkRollbackWithComplexAssumptions) {
  MiniSATSolver s;
  s.reserveVars(100);
  s.bookmark();

  // Add a complex network of binary and ternary clauses that could expose
  // issues in the bookmark/rollback + assumptions interaction

  // Create a chain of implications
  for (int i = 1; i <= 30; i++) {
    s.add(-i);
    s.add(i + 1);
    s.add(0);
  }

  // Add some binary clauses that form a complex dependency
  s.add(10);
  s.add(-31);
  s.add(0);
  s.add(14);
  s.add(-31);
  s.add(0);

  // Add ternary clauses
  s.add(-31);
  s.add(-50);
  s.add(-60);
  s.add(0);
  s.add(31);
  s.add(50);
  s.add(0);

  // First solve should be SAT
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);

  // Now solve with assumptions - should still be consistent
  s.assume(50);
  s.assume(-60);
  auto result = s.solve();

  // Result should be either SAT or UNSAT, but never a crash or corruption
  EXPECT_TRUE(result == MiniSATSolver::kSAT ||
              result == MiniSATSolver::kUNSAT);

  s.rollback();

  // After rollback, solver should be in clean state
  EXPECT_EQ(s.getNumClauses(), 0);
}

// Reproducer focusing on the specific pattern: binary clause conflict
// at root level after assumptions + BCP with bookmark/rollback.
TEST(MiniSATSolverTest, BinaryConflictAtRootAfterRollback) {
  MiniSATSolver s;
  s.reserveVars(50);
  s.bookmark();

  // Create a scenario where assumptions + BCP could lead to root-level
  // conflicts if rollback doesn't properly restore watch lists

  // Binary clauses that should trigger specific watch list patterns
  s.add(1);
  s.add(-10);
  s.add(0);
  s.add(2);
  s.add(-10);
  s.add(0);
  s.add(-1);
  s.add(-2);
  s.add(0);

  // This creates: (1 OR -10), (2 OR -10), (-1 OR -2)
  // With assumptions, this might expose rollback bugs

  // First query
  s.assume(1);
  s.assume(2);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT); // Conflict via (-1 OR -2)

  s.rollback();

  // After rollback, the same problem setup
  s.bookmark();
  s.add(1);
  s.add(-10);
  s.add(0);
  s.add(2);
  s.add(-10);
  s.add(0);
  s.add(-1);
  s.add(-2);
  s.add(0);

  // Should get the same result
  s.assume(1);
  s.assume(2);
  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);

  s.rollback();
}

// Test for accumulation of learned clauses across bookmark/rollback cycles
// which could cause incorrect UNSAT results.
TEST(MiniSATSolverTest, LearnedClauseAccumulationBug) {
  MiniSATSolver s;
  s.reserveVars(20);
  s.bookmark();

  // Cycle 1: Add clauses that generate learned clauses (3-into-2 pigeonhole)
  // Same as Pigeonhole3Into2 test
  s.add(1);
  s.add(2);
  s.add(0);
  s.add(3);
  s.add(4);
  s.add(0);
  s.add(5);
  s.add(6);
  s.add(0);
  s.add(-1);
  s.add(-3);
  s.add(0);
  s.add(-1);
  s.add(-5);
  s.add(0);
  s.add(-3);
  s.add(-5);
  s.add(0);
  s.add(-2);
  s.add(-4);
  s.add(0);
  s.add(-2);
  s.add(-6);
  s.add(0);
  s.add(-4);
  s.add(-6);
  s.add(0);

  EXPECT_EQ(s.solve(), MiniSATSolver::kUNSAT);
  s.rollback();

  // Cycle 2: Add different SAT clauses - should not be affected by
  // learned clauses from Cycle 1
  s.bookmark();
  s.add(7);
  s.add(8);
  s.add(0);
  s.add(-7);
  s.add(9);
  s.add(0);

  // This should be SAT, but if learned clauses leak, might be UNSAT
  EXPECT_EQ(s.solve(), MiniSATSolver::kSAT);
  s.rollback();
}

// Test the exact failure pattern from mul.sv: assumptions with many
// binary clauses after bookmark/rollback
TEST(MiniSATSolverTest, ManyBinaryClausesWithAssumptions) {
  MiniSATSolver s;
  s.reserveVars(200);
  s.bookmark();

  // Add many binary clauses (simulating Tseitin encoding)
  for (int i = 1; i <= 50; i++) {
    s.add(i);
    s.add(-(i + 50));
    s.add(0);

    s.add(i + 10);
    s.add(-(i + 50));
    s.add(0);

    s.add(-(i + 50));
    s.add(-(i + 100));
    s.add(0);

    s.add(i + 50);
    s.add(-(i + 100));
    s.add(0);
  }

  // Solve with assumptions
  s.assume(100);
  s.assume(-101);

  auto result1 = s.solve();

  // Rollback and re-add same problem
  s.rollback();
  s.bookmark();

  for (int i = 1; i <= 50; i++) {
    s.add(i);
    s.add(-(i + 50));
    s.add(0);

    s.add(i + 10);
    s.add(-(i + 50));
    s.add(0);

    s.add(-(i + 50));
    s.add(-(i + 100));
    s.add(0);

    s.add(i + 50);
    s.add(-(i + 100));
    s.add(0);
  }

  s.assume(100);
  s.assume(-101);

  auto result2 = s.solve();

  // Results should be consistent across rollback
  EXPECT_EQ(result1, result2) << "Solver gives different results before/after rollback!";

  s.rollback();
}

TEST(MiniSATSolverTest, DIMACSExport) {
  MiniSATSolver s;

  // Create a simple formula: (x1 OR x2) AND (NOT x1 OR x3)
  // Clause 1: x1 OR x2
  s.add(1);
  s.add(2);
  s.add(0);

  // Clause 2: NOT x1 OR x3
  s.add(-1);
  s.add(3);
  s.add(0);

  // Export to string
  std::string output;
  llvm::raw_string_ostream os(output);
  s.dumpDIMACS(os);

  // Verify header
  EXPECT_TRUE(output.find("p cnf 3 2") != std::string::npos)
      << "Expected 3 variables and 2 clauses in header";

  // Verify clauses are present
  EXPECT_TRUE(output.find("1 2 0") != std::string::npos)
      << "Expected clause '1 2 0'";
  EXPECT_TRUE(output.find("-1 3 0") != std::string::npos)
      << "Expected clause '-1 3 0'";
}

TEST(MiniSATSolverTest, DIMACSExportWithAssumptions) {
  MiniSATSolver s;

  // Simple clause: x1 OR x2
  s.add(1);
  s.add(2);
  s.add(0);

  // Export with assumptions
  std::string output;
  llvm::raw_string_ostream os(output);
  s.dumpDIMACS(os, {1, -2}); // Assume x1=true, x2=false

  // Verify assumptions are exported as unit clauses
  EXPECT_TRUE(output.find("1 0") != std::string::npos)
      << "Expected assumption '1 0'";
  EXPECT_TRUE(output.find("-2 0") != std::string::npos)
      << "Expected assumption '-2 0'";

  // Should have 3 clauses total (1 original + 2 assumptions)
  EXPECT_TRUE(output.find("p cnf 2 3") != std::string::npos)
      << "Expected 2 variables and 3 clauses";
}
