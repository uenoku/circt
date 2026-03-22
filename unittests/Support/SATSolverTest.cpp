//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SATSolver.h"

#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <sstream>
#include <cstdint>
#include <random>

using namespace circt;

namespace {

using Clause = llvm::SmallVector<int, 4>;

struct HeapNode {
  double score = 0.0;
};

struct HeapNodeScore {
  double operator()(const HeapNode &node) const { return node.score; }
};

static bool hasZ3Oracle() { return static_cast<bool>(createZ3SATSolver()); }

static IncrementalSATSolver::Result
solveWithZ3Oracle(llvm::ArrayRef<Clause> clauses,
                  llvm::ArrayRef<int> assumptions) {
  auto oracle = createZ3SATSolver();
  if (!oracle)
    return IncrementalSATSolver::kUNKNOWN;
  for (const Clause &clause : clauses)
    oracle->addClause(clause);
  return oracle->solve(assumptions);
}

static void expectModelSatisfies(IncrementalSATSolver &solver, unsigned numVars,
                                 llvm::ArrayRef<Clause> clauses,
                                 llvm::ArrayRef<int> assumptions) {
  llvm::SmallVector<bool> usedVars(numVars + 1, false);
  for (int lit : assumptions) {
    usedVars[std::abs(lit)] = true;
    int value = solver.val(std::abs(lit));
    ASSERT_NE(0, value);
    EXPECT_EQ(lit > 0, value > 0);
  }

  for (const Clause &clause : clauses) {
    bool satisfied = false;
    for (int lit : clause) {
      usedVars[std::abs(lit)] = true;
      int value = solver.val(std::abs(lit));
      ASSERT_NE(0, value);
      if ((lit > 0 && value > 0) || (lit < 0 && value < 0)) {
        satisfied = true;
        break;
      }
    }
    EXPECT_TRUE(satisfied);
  }

  for (unsigned var = 1; var <= numVars; ++var)
    if (usedVars[var])
      EXPECT_NE(0, solver.val(var));
}

static std::unique_ptr<IncrementalSATSolver> makeNativeSolver() {
  return createNativeSATSolver();
}

static std::unique_ptr<IncrementalSATSolver>
makeRestartHeavyNativeSolver() {
  return createNativeSATSolver();
}

static std::string formatClause(llvm::ArrayRef<int> clause) {
  std::ostringstream os;
  os << "{";
  for (size_t i = 0; i < clause.size(); ++i) {
    if (i)
      os << ", ";
    os << clause[i];
  }
  os << "}";
  return os.str();
}

static std::string formatClauses(llvm::ArrayRef<Clause> clauses) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < clauses.size(); ++i) {
    if (i)
      os << ", ";
    os << formatClause(clauses[i]);
  }
  os << "]";
  return os.str();
}

} // namespace

TEST(SatSolverTest, IndexedMaxHeapPopsInDescendingScoreOrder) {
  llvm::SmallVector<HeapNode, 4> nodes = {{1.0}, {5.0}, {3.0}, {4.0}};
  IndexedMaxHeap<HeapNode, HeapNodeScore> heap(nodes);

  for (unsigned i = 0; i < nodes.size(); ++i)
    heap.insert(i);

  EXPECT_EQ(1u, heap.pop());
  EXPECT_EQ(3u, heap.pop());
  EXPECT_EQ(2u, heap.pop());
  EXPECT_EQ(0u, heap.pop());
  EXPECT_TRUE(heap.empty());
}

TEST(SatSolverTest, IndexedMaxHeapIncreaseReordersExistingEntry) {
  llvm::SmallVector<HeapNode, 4> nodes = {{1.0}, {2.0}, {3.0}};
  IndexedMaxHeap<HeapNode, HeapNodeScore> heap(nodes);

  for (unsigned i = 0; i < nodes.size(); ++i)
    heap.insert(i);

  nodes[0].score = 10.0;
  heap.increase(0);

  EXPECT_EQ(0u, heap.pop());
  EXPECT_EQ(2u, heap.pop());
  EXPECT_EQ(1u, heap.pop());
}

TEST(SatSolverTest, IndexedMaxHeapAvoidsDuplicateInsertions) {
  llvm::SmallVector<HeapNode, 2> nodes = {{1.0}, {2.0}};
  IndexedMaxHeap<HeapNode, HeapNodeScore> heap(nodes);

  heap.insert(0);
  heap.insert(1);
  heap.insert(1);

  EXPECT_EQ(1u, heap.pop());
  EXPECT_EQ(0u, heap.pop());
  EXPECT_TRUE(heap.empty());
}

TEST(SatSolverTest, NativeUnitClauseAndAssumption) {
  auto solver = makeNativeSolver();

  solver->addClause({1});
  solver->addClause({-1, 2});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));

  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve({-1}));
}

TEST(SatSolverTest, NativeConflictBudgetProducesUnknown) {
  auto solver = makeNativeSolver();

  solver->addClause({1, 2});
  solver->addClause({-1, 2});
  solver->addClause({1, -2});
  solver->setConflictBudget(0);

  EXPECT_EQ(IncrementalSATSolver::kUNKNOWN, solver->solve());
}

TEST(SatSolverTest, NativeAssumptionsAreScopedToSolve) {
  auto solver = makeNativeSolver();

  solver->addClause({1});
  solver->addClause({2});

  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve({-1}));
  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));
}

TEST(SatSolverTest, NativeAddAndAssumeCompatibilityAPI) {
  auto solver = makeNativeSolver();

  solver->add(1);
  solver->add(0);
  solver->add(-1);
  solver->add(2);
  solver->add(0);

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());

  solver->assume(-2);
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, NativeTautologicalClauseIgnored) {
  auto solver = makeNativeSolver();

  solver->addClause({1, -1});
  solver->addClause({2});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(2, solver->val(2));
}

TEST(SatSolverTest, NativeDuplicateLiteralsAreDeduplicated) {
  auto solver = makeNativeSolver();

  solver->addClause({1, 1, 1});
  solver->addClause({-1, 2, 2});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));
}

TEST(SatSolverTest, NativeEmptyClauseMakesSolverUnsat) {
  auto solver = makeNativeSolver();

  solver->addClause({});

  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, NativeContradictoryUnitClausesRemainUnsat) {
  auto solver = makeNativeSolver();

  solver->addClause({1});
  solver->addClause({-1});

  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve({1}));
}

TEST(SatSolverTest, NativeBinaryPropagationChain) {
  auto solver = makeNativeSolver();

  solver->addClause({1});
  solver->addClause({-1, 2});
  solver->addClause({-2, 3});
  solver->addClause({-3, 4});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));
  EXPECT_EQ(3, solver->val(3));
  EXPECT_EQ(4, solver->val(4));
}

TEST(SatSolverTest, NativeTopLevelClauseAdditionCanMakeProblemUnsat) {
  auto solver = makeNativeSolver();

  solver->addClause({1, 2});
  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());

  solver->addClause({-1});
  solver->addClause({-2});
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, NativeInvalidVariableValueIsZero) {
  auto solver = makeNativeSolver();

  solver->addClause({1});
  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(0, solver->val(0));
  EXPECT_EQ(0, solver->val(2));
}

TEST(SatSolverTest, NativeConflictBudgetCanBeCleared) {
  auto solver = makeNativeSolver();

  solver->addClause({1, 2});
  solver->addClause({-1, 2});
  solver->addClause({1, -2});
  solver->setConflictBudget(0);
  EXPECT_EQ(IncrementalSATSolver::kUNKNOWN, solver->solve());

  solver->clearConflictBudget();
  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
}

TEST(SatSolverTest, NativeStatsTrackSolveCalls) {
  auto solver = makeNativeSolver();

  solver->addClause({1});
  EXPECT_EQ(0u, solver->stats().numSolveCalls);
  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1u, solver->stats().numSolveCalls);
  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(2u, solver->stats().numSolveCalls);
}

TEST(SatSolverTest, NativeContradictoryAssumptionsAreUnsat) {
  auto solver = makeNativeSolver();

  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve({1, -1}));
  EXPECT_EQ(0, solver->val(1));
}

TEST(SatSolverTest, NativeAssumptionsOnFreshVariablesAreScoped) {
  auto solver = makeNativeSolver();

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve({3}));
  EXPECT_EQ(3, solver->val(3));

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve({-3}));
  EXPECT_EQ(-3, solver->val(3));
}

TEST(SatSolverTest, NativeUnknownDoesNotPoisonFutureSolve) {
  auto solver = makeNativeSolver();

  solver->addClause({1, 2});
  solver->addClause({-1, 2});
  solver->addClause({1, -2});
  solver->addClause({-1, -2});

  solver->setConflictBudget(0);
  EXPECT_EQ(IncrementalSATSolver::kUNKNOWN, solver->solve());
  EXPECT_EQ(0, solver->val(1));
  EXPECT_EQ(0, solver->val(2));

  solver->clearConflictBudget();
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, NativeRootUnsatRemainsUnsatUnderAssumptions) {
  auto solver = makeNativeSolver();

  solver->addClause({1});
  solver->addClause({-1});

  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve({2}));
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve({-2, 3}));
  EXPECT_EQ(0, solver->val(1));
}

TEST(SatSolverTest, NativeDeterministicRandomDifferential) {
  if (!hasZ3Oracle())
    GTEST_SKIP() << "Z3 is not available in this build.";

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> clauseSizeDist(1, 3);
  std::uniform_int_distribution<int> signDist(0, 1);
  constexpr unsigned numVars = 4;

  for (unsigned testIndex = 0; testIndex < 100; ++testIndex) {
    auto solver = makeNativeSolver();
    llvm::SmallVector<Clause> clauses;

    unsigned numClauses = 1 + (rng() % 8);
    for (unsigned clauseIndex = 0; clauseIndex < numClauses; ++clauseIndex) {
      Clause clause;
      unsigned clauseSize = clauseSizeDist(rng);
      for (unsigned litIndex = 0; litIndex < clauseSize; ++litIndex) {
        int var = 1 + static_cast<int>(rng() % numVars);
        int lit = signDist(rng) ? var : -var;
        clause.push_back(lit);
      }
      clauses.push_back(clause);
      solver->addClause(clause);
    }

    unsigned numAssumptions = rng() % 4;
    llvm::SmallVector<int> assumptions;
    for (unsigned i = 0; i < numAssumptions; ++i) {
      int var = 1 + static_cast<int>(rng() % numVars);
      assumptions.push_back(signDist(rng) ? var : -var);
    }

    auto expected = solveWithZ3Oracle(clauses, assumptions);
    ASSERT_NE(IncrementalSATSolver::kUNKNOWN, expected)
        << "oracle returned unknown testIndex=" << testIndex;
    auto actual = solver->solve(assumptions);

    EXPECT_EQ(expected, actual) << "testIndex=" << testIndex;
    if (actual == IncrementalSATSolver::kSAT)
      expectModelSatisfies(*solver, numVars, clauses, assumptions);
  }
}

TEST(SatSolverTest, NativeRestartHeavyDeterministicRandomDifferential) {
  if (!hasZ3Oracle())
    GTEST_SKIP() << "Z3 is not available in this build.";

  std::mt19937 rng(67890);
  std::uniform_int_distribution<int> clauseSizeDist(1, 3);
  std::uniform_int_distribution<int> signDist(0, 1);
  constexpr unsigned numVars = 4;

  for (unsigned testIndex = 0; testIndex < 100; ++testIndex) {
    auto solver = makeRestartHeavyNativeSolver();
    llvm::SmallVector<Clause> clauses;

    unsigned numClauses = 1 + (rng() % 8);
    for (unsigned clauseIndex = 0; clauseIndex < numClauses; ++clauseIndex) {
      Clause clause;
      unsigned clauseSize = clauseSizeDist(rng);
      for (unsigned litIndex = 0; litIndex < clauseSize; ++litIndex) {
        int var = 1 + static_cast<int>(rng() % numVars);
        int lit = signDist(rng) ? var : -var;
        clause.push_back(lit);
      }
      clauses.push_back(clause);
      solver->addClause(clause);
    }

    unsigned numAssumptions = 1 + (rng() % 4);
    llvm::SmallVector<int> assumptions;
    for (unsigned i = 0; i < numAssumptions; ++i) {
      int var = 1 + static_cast<int>(rng() % numVars);
      assumptions.push_back(signDist(rng) ? var : -var);
    }

    auto expected = solveWithZ3Oracle(clauses, assumptions);
    ASSERT_NE(IncrementalSATSolver::kUNKNOWN, expected)
        << "oracle returned unknown testIndex=" << testIndex;
    auto actual = solver->solve(assumptions);

    EXPECT_EQ(expected, actual) << "testIndex=" << testIndex;
    if (actual == IncrementalSATSolver::kSAT)
      expectModelSatisfies(*solver, numVars, clauses, assumptions);
  }
}

TEST(SatSolverTest, NativeDeterministicIncrementalRandomDifferential) {
  if (!hasZ3Oracle())
    GTEST_SKIP() << "Z3 is not available in this build.";

  std::mt19937 rng(24680);
  std::uniform_int_distribution<int> clauseSizeDist(1, 3);
  std::uniform_int_distribution<int> signDist(0, 1);
  constexpr unsigned numVars = 4;

  for (unsigned testIndex = 0; testIndex < 50; ++testIndex) {
    auto solver = makeNativeSolver();
    llvm::SmallVector<Clause> clauses;

    for (unsigned step = 0; step < 12; ++step) {
      if ((rng() % 3) != 0 || clauses.empty()) {
        Clause clause;
        unsigned clauseSize = clauseSizeDist(rng);
        for (unsigned litIndex = 0; litIndex < clauseSize; ++litIndex) {
          int var = 1 + static_cast<int>(rng() % numVars);
          clause.push_back(signDist(rng) ? var : -var);
        }
        clauses.push_back(clause);
        solver->addClause(clause);
        continue;
      }

      unsigned numAssumptions = rng() % 4;
      llvm::SmallVector<int> assumptions;
      for (unsigned i = 0; i < numAssumptions; ++i) {
        int var = 1 + static_cast<int>(rng() % numVars);
        assumptions.push_back(signDist(rng) ? var : -var);
      }

      auto expected = solveWithZ3Oracle(clauses, assumptions);
      ASSERT_NE(IncrementalSATSolver::kUNKNOWN, expected)
          << "oracle returned unknown testIndex=" << testIndex
          << " step=" << step;
      auto actual = solver->solve(assumptions);

      EXPECT_EQ(expected, actual)
          << "testIndex=" << testIndex << " step=" << step
          << " clauses=" << formatClauses(clauses)
          << " assumptions=" << formatClause(assumptions);
      if (actual == IncrementalSATSolver::kSAT)
        expectModelSatisfies(*solver, numVars, clauses, assumptions);
    }

    auto expected = solveWithZ3Oracle(clauses, {});
    ASSERT_NE(IncrementalSATSolver::kUNKNOWN, expected)
        << "oracle returned unknown testIndex=" << testIndex << " final";
    auto actual = solver->solve();
    EXPECT_EQ(expected, actual) << "testIndex=" << testIndex << " final";
    if (actual == IncrementalSATSolver::kSAT)
      expectModelSatisfies(*solver, numVars, clauses, {});
  }
}

TEST(SatSolverTest, NativeTseitinAndGate) {
  auto solver = makeNativeSolver();

  // c = a AND b, with a=1, b=2, c=3.
  solver->addClause({-3, 1});
  solver->addClause({-3, 2});
  solver->addClause({3, -1, -2});
  solver->addClause({3});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));
  EXPECT_EQ(3, solver->val(3));
}

TEST(SatSolverTest, NativeTseitinOrGate) {
  auto solver = makeNativeSolver();

  // c = a OR b, with a=1, b=2, c=3.
  solver->addClause({3, -1});
  solver->addClause({3, -2});
  solver->addClause({-3, 1, 2});
  solver->addClause({-3});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(-1, solver->val(1));
  EXPECT_EQ(-2, solver->val(2));
  EXPECT_EQ(-3, solver->val(3));
}

TEST(SatSolverTest, NativeTseitinXorChain) {
  auto solver = makeNativeSolver();

  // t = x1 XOR x2 (var 4), result = t XOR x3 (var 5), assert result = true.
  solver->addClause({-4, -1, -2});
  solver->addClause({-4, 1, 2});
  solver->addClause({4, -1, 2});
  solver->addClause({4, 1, -2});

  solver->addClause({-5, -4, -3});
  solver->addClause({-5, 4, 3});
  solver->addClause({5, -4, 3});
  solver->addClause({5, 4, -3});
  solver->addClause({5});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());

  int v1 = solver->val(1) > 0 ? 1 : 0;
  int v2 = solver->val(2) > 0 ? 1 : 0;
  int v3 = solver->val(3) > 0 ? 1 : 0;
  EXPECT_EQ(v1 ^ v2 ^ v3, 1);
}

TEST(SatSolverTest, NativePigeonholeThreeIntoTwoIsUnsat) {
  auto solver = makeNativeSolver();

  // 3 pigeons into 2 holes. Variables:
  // p11=1, p12=2, p21=3, p22=4, p31=5, p32=6.
  solver->addClause({1, 2});
  solver->addClause({3, 4});
  solver->addClause({5, 6});

  solver->addClause({-1, -3});
  solver->addClause({-1, -5});
  solver->addClause({-3, -5});

  solver->addClause({-2, -4});
  solver->addClause({-2, -6});
  solver->addClause({-4, -6});

  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, NativePigeonholeTwoIntoTwoIsSat) {
  auto solver = makeNativeSolver();

  // 2 pigeons into 2 holes. Variables:
  // p11=1, p12=2, p21=3, p22=4.
  solver->addClause({1, 2});
  solver->addClause({3, 4});
  solver->addClause({-1, -3});
  solver->addClause({-2, -4});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());

  bool p11 = solver->val(1) > 0;
  bool p12 = solver->val(2) > 0;
  bool p21 = solver->val(3) > 0;
  bool p22 = solver->val(4) > 0;
  EXPECT_TRUE(p11 || p12);
  EXPECT_TRUE(p21 || p22);
  EXPECT_FALSE(p11 && p21);
  EXPECT_FALSE(p12 && p22);
}

TEST(SatSolverTest, NativeIncrementalSolvesWithAddedClauses) {
  auto solver = makeNativeSolver();

  solver->addClause({1, 2});
  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());

  solver->addClause({-1, -2});
  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_NE(solver->val(1) > 0, solver->val(2) > 0);

  solver->addClause({1});
  solver->addClause({2});
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, NativeConflictLimitMayReturnUnknownOnHarderUnsat) {
  auto solver = makeNativeSolver();

  int vars[4][3];
  int nextVar = 1;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      vars[i][j] = nextVar++;

  for (int i = 0; i < 4; ++i)
    solver->addClause({vars[i][0], vars[i][1], vars[i][2]});

  for (int j = 0; j < 3; ++j)
    for (int i1 = 0; i1 < 4; ++i1)
      for (int i2 = i1 + 1; i2 < 4; ++i2)
        solver->addClause({-vars[i1][j], -vars[i2][j]});

  solver->setConflictBudget(1);
  auto limited = solver->solve();
  EXPECT_TRUE(limited == IncrementalSATSolver::kUNKNOWN ||
              limited == IncrementalSATSolver::kUNSAT);

  solver->clearConflictBudget();
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, NativeImplicationChainUnsat) {
  auto solver = makeNativeSolver();

  for (int i = 1; i < 10; ++i)
    solver->addClause({-i, i + 1});
  solver->addClause({1});
  solver->addClause({-10});

  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, NativeImplicationChainSat) {
  auto solver = makeNativeSolver();

  for (int i = 1; i < 10; ++i)
    solver->addClause({-i, i + 1});
  solver->addClause({1});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  for (int i = 1; i <= 10; ++i)
    EXPECT_EQ(i, solver->val(i));
}

TEST(SatSolverTest, UnitClauseAndAssumption) {
  // -DLLVM_ENABLE_Z3_SOLVER=ON is required to run this test.
  auto solver = createZ3SATSolver();
  if (!solver)
    GTEST_SKIP() << "Z3 is not available in this build.";

  // (x1) AND (!x1 OR x2)
  // This should be satisfiable with x1=true, x2=true.
  // But if we assume !x1, it should become unsatisfiable.
  solver->addClause({1});
  solver->addClause({-1, 2});

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));

  solver->assume(-1);
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());
}

TEST(SatSolverTest, AssumptionsAreScopedToSolve) {
  // -DLLVM_ENABLE_Z3_SOLVER=On is required to run this test.
  auto solver = createZ3SATSolver();
  if (!solver)
    GTEST_SKIP() << "Z3 is not available in this build.";

  solver->addClause({1});
  solver->addClause({2});

  solver->assume(-1);
  EXPECT_EQ(IncrementalSATSolver::kUNSAT, solver->solve());

  EXPECT_EQ(IncrementalSATSolver::kSAT, solver->solve());
  EXPECT_EQ(1, solver->val(1));
  EXPECT_EQ(2, solver->val(2));
}
