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
