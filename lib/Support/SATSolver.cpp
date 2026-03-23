//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines incremental SAT solvers with an IPASIR-style interface.
//
// Native solver design:
// - The core search loop follows a MiniSat-style CDCL solver intended for low
//   latency.
// - References:
//   * Moskewicz et al., "Chaff: Engineering an Efficient SAT Solver" (DAC
//     2001)
//   * Een and Sorensson, "An Extensible SAT-solver" (SAT 2003)
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SATSolver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/SMTAPI.h"
#include "llvm/Support/TrailingObjects.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace circt {

namespace {

static const SATSolverStats kEmptyStats;
constexpr double kActivityRescaleThreshold = 1e100;
constexpr double kActivityRescaleFactor = 1e-100;

class Lit {
public:
  constexpr Lit() = default;

  static constexpr Lit fromRaw(int raw) {
    assert(raw != 0 && "literal must be non-zero");
    Lit lit;
    lit.rawValue = raw;
    return lit;
  }

  constexpr int raw() const { return rawValue; }
  constexpr int var() const { return std::abs(rawValue); }
  constexpr bool isNegated() const { return rawValue < 0; }
  constexpr Lit negated() const { return Lit::fromRaw(-rawValue); }
  constexpr int watchIndex() const {
    int v = var();
    return isNegated() ? 2 * (v - 1) + 1 : 2 * (v - 1);
  }

  friend constexpr bool operator==(Lit lhs, Lit rhs) {
    return lhs.rawValue == rhs.rawValue;
  }
  friend constexpr bool operator<(Lit lhs, Lit rhs) {
    return lhs.rawValue < rhs.rawValue;
  }

private:
  int rawValue = 0;
};

static_assert(sizeof(Lit) == sizeof(int), "Lit should stay register-passable");
static_assert(alignof(Lit) == alignof(int),
              "Lit should preserve int alignment");
static_assert(std::is_trivially_copyable_v<Lit>,
              "Lit should remain a trivial wrapper");

enum class Assignment : int8_t {
  Unassigned = 0,
  True = 1,
  False = -1,
};

static_assert(sizeof(Assignment) == sizeof(int8_t),
              "Assignment should remain byte-sized");

inline static Assignment assignmentFor(Lit lit) {
  return lit.isNegated() ? Assignment::False : Assignment::True;
}

inline static int assignmentToInt(Assignment assignment) {
  return static_cast<int>(assignment);
}

struct Clause final : private llvm::TrailingObjects<Clause, Lit> {
  friend TrailingObjects;

  unsigned numLits = 0;
  bool learnt = false;

  Clause(unsigned numLits, bool learnt);

  static Clause *create(llvm::BumpPtrAllocator &allocator,
                        llvm::ArrayRef<Lit> lits, bool learnt);

  llvm::MutableArrayRef<Lit> getLits();
  llvm::ArrayRef<Lit> getLits() const;
};

Clause::Clause(unsigned numLits, bool learnt)
    : numLits(numLits), learnt(learnt) {}

Clause *Clause::create(llvm::BumpPtrAllocator &allocator,
                       llvm::ArrayRef<Lit> lits, bool learnt) {
  void *mem =
      allocator.Allocate(totalSizeToAlloc<Lit>(lits.size()), alignof(Clause));
  Clause *clause = new (mem) Clause(lits.size(), learnt);
  std::uninitialized_copy(lits.begin(), lits.end(),
                          clause->getTrailingObjects());
  return clause;
}

llvm::MutableArrayRef<Lit> Clause::getLits() {
  return {getTrailingObjects(), numLits};
}

llvm::ArrayRef<Lit> Clause::getLits() const {
  return {getTrailingObjects(), numLits};
}

struct Watcher {
  // The clause currently watching one literal in a watch list keyed by the
  // other watched literal becoming false.
  Clause *clause = nullptr;

  // Cached "other watched literal". If it is already true, propagation can
  // skip touching the full clause.
  Lit blocker;
};

struct VariableState {
  // VSIDS activity score used by the branching heap.
  double activity = 0.0;

  // Null for decisions/root units, otherwise the clause that implied the value.
  Clause *reason = nullptr;

  // Decision level at which the current assignment was made.
  int32_t level = 0;

  // Current truth value in the partial assignment.
  Assignment assignment = Assignment::Unassigned;

  // Saved preferred branch polarity.
  int8_t polarity = 1;

  // Temporary mark bit used during conflict analysis.
  uint8_t seen = 0;

  // NOTE: VariableState layout expected to remain cache-friendly for the VSIDS
  // heap. So be careful of adding new fields here that could cause cache lines
  // to split.
};

struct VariableActivityScore {
  double operator()(const VariableState &state) const { return state.activity; }
};

inline static llvm::SmallVector<Lit, 8> toLits(llvm::ArrayRef<int> lits) {
  llvm::SmallVector<Lit, 8> result;
  result.reserve(lits.size());
  for (int lit : lits) {
    assert(lit != 0 && "clause/assumption literals must be non-zero");
    result.push_back(Lit::fromRaw(lit));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Native Incremental SAT Solver
//===----------------------------------------------------------------------===//

class NativeSATSolver : public IncrementalSATSolver {
public:
  explicit NativeSATSolver(const NativeSATSolverOptions &options);

  //===--------------------------------------------------------------------===//
  // IncrementalSATSolver Interface
  //===--------------------------------------------------------------------===//

  void add(int lit) override;
  void assume(int lit) override;

  // Returns kSAT / kUNSAT / kUNKNOWN. kUNKNOWN is used when search stops
  // early (for example due to conflict budget).
  Result solve() override;

  // Same result contract as solve(). `assumptions` are temporary for this call.
  Result solve(llvm::ArrayRef<int> assumptions) override;

  // Valid only after kSAT. Returns +v or -v for the model value, otherwise 0.
  int val(int v) const override;
  void reserveVars(int maxVar) override;

private:
  void addClause(llvm::ArrayRef<int> lits) override;
  void addClauseInternal(llvm::ArrayRef<Lit> lits);
  void setConflictBudget(uint64_t conflicts) override;
  void clearConflictBudget() override;
  const SATSolverStats &stats() const override;

  enum class SearchAction { Continue, Unsat, Sat };

  //===--------------------------------------------------------------------===//
  // Clause Handling
  //===--------------------------------------------------------------------===//

  // Clause front-end helpers for the compatibility add()/assume() APIs and
  // for canonicalizing new permanent clauses before they reach the watch lists.
  void flushBufferedClause();

  // Returns false if the clause is a tautology and should be discarded.
  bool canonicalizeClause(llvm::SmallVectorImpl<Lit> &clause);

  // Drop literals falsified at level 0 and discard clauses already satisfied
  // by the permanent root assignment.
  bool reduceClauseAtRoot(llvm::SmallVectorImpl<Lit> &clause);

  void insertTopLevelClause(llvm::ArrayRef<Lit> clause);
  Clause *allocateClause(llvm::ArrayRef<Lit> lits, bool learnt);
  void attachClause(Clause &clause);

  //===--------------------------------------------------------------------===//
  // Solve Lifecycle
  //===--------------------------------------------------------------------===//

  // Per-solve setup before entering the CDCL loop.
  // Returns std::nullopt when search should continue normally.
  // Returns a terminal result (currently kUNSAT) when root-level propagation
  // already proves the query inconsistent.
  std::optional<Result> prepareSolve(llvm::ArrayRef<Lit> assumptions);

  // Finish an UNSAT result, optionally marking the permanent clause database as
  // inconsistent.
  Result finishUnsat(bool makePermanentUnsat);

  // Reset temporary state from the last solve, so the solver is ready for a new
  // search. This is called at the start of add()/assume() and at the end of
  // solve().
  void resetLastSolveState();
  Result solveInternal(llvm::ArrayRef<Lit> assumptions);

  //===--------------------------------------------------------------------===//
  // Search
  //===--------------------------------------------------------------------===//

  // Search loop helpers. These implement the MiniSat-style
  // propagate/analyze/backtrack/branch cycle.
  // Returns false if the learned clause immediately causes a contradiction.
  bool recordLearnedClause(llvm::ArrayRef<Lit> learntClause);

  // Rebuild the temporary assumption decision prefix one unmet assumption at a
  // time.
  std::optional<SearchAction>
  tryEnqueueNextAssumption(llvm::ArrayRef<Lit> assumptions);

  // Chooses whether to continue search, report UNSAT under assumptions, or SAT.
  SearchAction scheduleNextAssignment(llvm::ArrayRef<Lit> assumptions);

  // Returns the current value of `lit` under the current partial assignment.
  Assignment currentAssignment(Lit lit) const;
  inline VariableState &stateFor(Lit lit) { return variables[lit.var()]; }
  inline const VariableState &stateFor(Lit lit) const {
    return variables[lit.var()];
  }

  // Returns false iff `lit` conflicts with the current assignment.
  bool enqueue(Lit lit, Clause *reason);

  // Returns a conflicting clause, or nullptr when propagation reaches a fixed
  // point with no conflict.
  Clause *propagate();
  void analyze(Clause *conflict, llvm::SmallVectorImpl<Lit> &learntClause,
               int &backtrackLevel);

  // Returns the level to backtrack to for a learnt clause. This is the maximum
  // decision level among the literals in `learntClause` other than the
  // asserting literal at index 0.
  int computeBacktrackLevel(llvm::SmallVectorImpl<Lit> &learntClause) const;

  //===--------------------------------------------------------------------===//
  // Branching And Backtracking
  //===--------------------------------------------------------------------===//

  // Branching heuristic and decision-level maintenance.
  // Increase VSIDS activity for one variable and keep the order heap updated.
  void bumpVariableActivity(int var);
  // Age all activity scores implicitly by scaling the increment factor.
  void decayVariableActivity();
  // Returns 0 iff no unassigned decision variable remains.
  std::optional<Lit> pickBranchLiteral();
  void newDecisionLevel();
  int decisionLevel() const;
  void cancelUntil(int level);

  //===--------------------------------------------------------------------===//
  // Solver State
  //===--------------------------------------------------------------------===//

  // Solver configuration and externally visible accounting.
  NativeSATSolverOptions options;
  SATSolverStats solverStats;

  // Clause database. Clauses are monotonic for now, so bump allocation keeps
  // storage compact and avoids per-clause frees.
  llvm::BumpPtrAllocator clauseAllocator;
  llvm::SmallVector<Clause *, 0> learntClauses;
  llvm::SmallVector<llvm::SmallVector<Watcher, 4>, 0> watches;

  // Per-variable state and branching order. The variable table is 1-indexed so
  // external SAT variable ids can be used directly as indices.
  llvm::SmallVector<VariableState, 0> variables;
  IndexedMaxHeap<VariableState, VariableActivityScore> variableOrder;

  // Standard CDCL trail and per-level cut points.
  // Invariant: `trailLimits[d]` is the trail size at the start of decision
  // level `d + 1`, so `trailLimits.size() == decisionLevel()`.
  llvm::SmallVector<Lit> trail;
  llvm::SmallVector<size_t> trailLimits;

  // Temporary front-end state for IPASIR-style add()/assume() usage.
  llvm::SmallVector<Lit> clauseBuffer, pendingAssumptions;

  // Search cursors and scalar state.
  // Invariant: `nextPropagateIndex <= trail.size()`.
  // Entries in `trail[0..nextPropagateIndex)` have already been propagated.
  size_t nextPropagateIndex = 0;

  int maxVariable = 0;
  double variableActivityIncrement = 1.0;
  Result lastResult = kUNKNOWN;
  bool permanentUnsat = false;
  std::optional<uint64_t> conflictBudget;
};

NativeSATSolver::NativeSATSolver(const NativeSATSolverOptions &options)
    : options(options), variableOrder(variables) {}

void NativeSATSolver::add(int lit) {
  resetLastSolveState();
  if (lit == 0) {
    // The compatibility API terminates one clause with 0, so flush the
    // buffered literals as a permanent clause at that point.
    addClauseInternal(clauseBuffer);
    clauseBuffer.clear();
    return;
  }
  reserveVars(std::abs(lit));
  clauseBuffer.push_back(Lit::fromRaw(lit));
}

void NativeSATSolver::assume(int lit) {
  resetLastSolveState();
  if (lit == 0)
    return;
  // Assumptions are stored only until the next solve() call. They never enter
  // the permanent clause database.
  reserveVars(std::abs(lit));
  pendingAssumptions.push_back(Lit::fromRaw(lit));
}

IncrementalSATSolver::Result NativeSATSolver::solve() {
  auto assumptions = pendingAssumptions;
  pendingAssumptions.clear();
  return solveInternal(assumptions);
}

IncrementalSATSolver::Result
NativeSATSolver::solve(llvm::ArrayRef<int> assumptions) {
  return solveInternal(toLits(assumptions));
}

IncrementalSATSolver::Result
NativeSATSolver::solveInternal(llvm::ArrayRef<Lit> assumptions) {
  resetLastSolveState();
  ++solverStats.numSolveCalls;

  // Finish any in-progress add()/add(0) clause first so the search sees a
  // stable permanent clause database.
  flushBufferedClause();
  if (permanentUnsat)
    return lastResult = kUNSAT;

  // Assumptions can mention variables that have not appeared in any permanent
  // clause yet, so the solver must still allocate state for them.
  for (Lit lit : assumptions)
    reserveVars(lit.var());

  // Assumptions are modeled as a temporary decision prefix on top of the
  // permanent root assignment. All learned clauses and VSIDS state are reused
  // across calls, but assumption decisions are rebuilt from scratch for each
  // solve.
  if (auto result = prepareSolve(assumptions))
    return *result;

  uint64_t solveConflictBudget =
      conflictBudget ? *conflictBudget : std::numeric_limits<uint64_t>::max();
  uint64_t conflictsAtStart = solverStats.numConflicts;
  while (true) {
    // Keep propagating until either:
    // 1) a conflict is found, or
    // 2) there are no more forced assignments to make under the current
    //    decisions/assumptions.
    if (Clause *conflict = propagate()) {
      ++solverStats.numConflicts;
      if (decisionLevel() == 0)
        return finishUnsat(assumptions.empty());

      llvm::SmallVector<Lit, 8> learntClause;
      int backtrackLevel = 0;
      // Conflict analysis computes one asserting clause, then search
      // backtracks so that clause immediately becomes unit.
      analyze(conflict, learntClause, backtrackLevel);
      cancelUntil(backtrackLevel);

      if (!recordLearnedClause(learntClause))
        return finishUnsat(assumptions.empty());

      decayVariableActivity();
      continue;
    }

    if (solverStats.numConflicts - conflictsAtStart >= solveConflictBudget) {
      cancelUntil(0);
      return lastResult = kUNKNOWN;
    }

    // No conflict and no propagation left: either re-apply the next unmet
    // assumption, branch on a fresh variable, or conclude SAT if every
    // variable relevant to the clause database is assigned consistently.
    switch (scheduleNextAssignment(assumptions)) {
    case SearchAction::Continue:
      continue;
    case SearchAction::Unsat:
      return lastResult;
    case SearchAction::Sat:
      return lastResult = kSAT;
    }
  }
}

int NativeSATSolver::val(int v) const {
  // Models are only meaningful after SAT. UNSAT/UNKNOWN leave the previous
  // trail state intentionally inaccessible through the public API.
  if (lastResult != kSAT || v <= 0 || v > maxVariable)
    return 0;

  switch (variables[v].assignment) {
  // When unassigned, the caller can't distinguish that from an
  // assigned variable with the value false.
  case Assignment::Unassigned:
    return 0;
  case Assignment::True:
    return v;
  case Assignment::False:
    return -v;
  }
  llvm_unreachable("invalid variable assignment");
}

void NativeSATSolver::reserveVars(int maxVar) {
  if (maxVar <= maxVariable)
    return;
  // Variable ids are exposed directly to clients, so growing the solver means
  // extending every per-variable/per-literal side structure in lockstep.
  size_t newSize = maxVar + 1;
  variables.resize(newSize);
  watches.resize(static_cast<size_t>(maxVar) * 2);
  variableOrder.resize(maxVar + 1);
  for (int var = maxVariable + 1; var <= maxVar; ++var)
    variableOrder.insert(var);
  maxVariable = maxVar;
}

void NativeSATSolver::addClause(llvm::ArrayRef<int> lits) {
  addClauseInternal(toLits(lits));
}

void NativeSATSolver::addClauseInternal(llvm::ArrayRef<Lit> lits) {
  resetLastSolveState();
  cancelUntil(0);
  if (permanentUnsat)
    return;

  // Permanent clauses are normalized in a scratch buffer first so tautologies,
  // duplicates, and root-satisfied literals never enter the watch structure.
  llvm::SmallVector<Lit, 8> clause(lits.begin(), lits.end());
  if (!canonicalizeClause(clause) || !reduceClauseAtRoot(clause))
    return;
  insertTopLevelClause(clause);
}

void NativeSATSolver::setConflictBudget(uint64_t conflicts) {
  conflictBudget = conflicts;
}

void NativeSATSolver::clearConflictBudget() { conflictBudget.reset(); }

const SATSolverStats &NativeSATSolver::stats() const { return solverStats; }

void NativeSATSolver::flushBufferedClause() {
  if (clauseBuffer.empty())
    return;
  addClauseInternal(clauseBuffer);
  clauseBuffer.clear();
}

Clause *NativeSATSolver::allocateClause(llvm::ArrayRef<Lit> lits, bool learnt) {
  // Clause memory is monotonic today, so a bump allocator is enough and keeps
  // clause headers plus literals contiguous.
  return Clause::create(clauseAllocator, lits, learnt);
}

bool NativeSATSolver::canonicalizeClause(llvm::SmallVectorImpl<Lit> &clause) {
  // All literals must refer to existing variables before sorting or root
  // simplification can inspect assignments.
  for (Lit lit : clause)
    reserveVars(lit.var());

  if (clause.empty()) {
    permanentUnsat = true;
    return false;
  }

  // Canonical order groups equal/opposite literals together so one linear
  // pass can remove duplicates and detect tautologies.
  llvm::sort(clause, [](Lit lhs, Lit rhs) {
    int lhsVar = lhs.var();
    int rhsVar = rhs.var();
    if (lhsVar != rhsVar)
      return lhsVar < rhsVar;
    return lhs < rhs;
  });

  size_t writeIndex = 0;
  for (size_t i = 0, e = clause.size(); i < e; ++i) {
    // Remove duplicate literals
    if (i && clause[i] == clause[i - 1])
      continue;
    // A tautology is detected when a literal and its negation are adjacent.
    if (i && clause[i] == clause[i - 1].negated())
      return false;
    clause[writeIndex++] = clause[i];
  }
  clause.resize(writeIndex);
  if (clause.empty()) {
    permanentUnsat = true;
    return false;
  }
  return true;
}

bool NativeSATSolver::reduceClauseAtRoot(llvm::SmallVectorImpl<Lit> &clause) {
  // New permanent clauses are simplified against the current level-0
  // assignment so propagation never has to revisit literals already fixed at
  // the root.
  size_t out = 0;
  for (Lit lit : clause) {
    Assignment value = currentAssignment(lit);
    if (value == Assignment::True)
      return false;
    if (value == Assignment::False)
      continue;
    clause[out++] = lit;
  }
  clause.resize(out);
  if (!clause.empty())
    return true;
  permanentUnsat = true;
  return false;
}

void NativeSATSolver::insertTopLevelClause(llvm::ArrayRef<Lit> clause) {
  // Root-level units are enqueued immediately; longer clauses are attached to
  // the watch lists and then propagated once to detect any immediate conflict.
  if (clause.size() == 1) {
    if (!enqueue(clause.front(), nullptr) || propagate())
      permanentUnsat = true;
    return;
  }

  Clause *storedClause = allocateClause(clause, /*learnt=*/false);
  attachClause(*storedClause);
  if (propagate())
    permanentUnsat = true;
}

std::optional<IncrementalSATSolver::Result>
NativeSATSolver::prepareSolve(llvm::ArrayRef<Lit> assumptions) {
  // Every solve starts from the root state. Learned clauses and activities
  // survive, but temporary assumption decisions do not.
  cancelUntil(0);
  if (!propagate())
    return std::nullopt;
  return finishUnsat(assumptions.empty());
}

IncrementalSATSolver::Result
NativeSATSolver::finishUnsat(bool makePermanentUnsat) {
  if (makePermanentUnsat)
    permanentUnsat = true;
  cancelUntil(0);
  return lastResult = kUNSAT;
}

bool NativeSATSolver::recordLearnedClause(llvm::ArrayRef<Lit> learntClause) {
  // A first-UIP analysis may produce a unit clause. In that case there is no
  // need to allocate a watched clause object; the implied literal can be
  // enqueued directly after backtracking.
  if (learntClause.size() == 1)
    return enqueue(learntClause.front(), nullptr);

  Clause *learnt = allocateClause(learntClause, /*learnt=*/true);
  learntClauses.push_back(learnt);
  attachClause(*learnt);
  ++solverStats.numLearnedClauses;
  return enqueue(learnt->getLits().front(), learnt);
}

std::optional<NativeSATSolver::SearchAction>
NativeSATSolver::tryEnqueueNextAssumption(llvm::ArrayRef<Lit> assumptions) {
  for (Lit assumption : assumptions) {
    Assignment value = currentAssignment(assumption);
    if (value == Assignment::True)
      continue;
    if (value == Assignment::False) {
      // The current trail already forces the opposite value, so the query is
      // UNSAT under the assumption prefix seen so far.
      finishUnsat(/*makePermanentUnsat=*/false);
      return SearchAction::Unsat;
    }
    // Assumptions are modeled as decisions layered above the root. As soon as
    // one unmet assumption is found, it becomes the next active decision and
    // propagation restarts from there.
    newDecisionLevel();
    (void)enqueue(assumption, nullptr);
    return SearchAction::Continue;
  }
  return std::nullopt;
}

NativeSATSolver::SearchAction
NativeSATSolver::scheduleNextAssignment(llvm::ArrayRef<Lit> assumptions) {
  // Assumptions have priority over free branching: the solver must first
  // rebuild the temporary assumption prefix for this solve call.
  if (auto assumptionAction = tryEnqueueNextAssumption(assumptions))
    return *assumptionAction;

  auto decision = pickBranchLiteral();
  if (!decision)
    // No unmet assumptions and no unassigned decision variable remain, so the
    // current assignment is a satisfying model for the permanent clauses and
    // the active assumption frame.
    return SearchAction::Sat;

  // Otherwise continue ordinary CDCL search with a heuristic branch decision.
  ++solverStats.numDecisions;
  newDecisionLevel();
  (void)enqueue(*decision, nullptr);
  return SearchAction::Continue;
}

void NativeSATSolver::attachClause(Clause &clause) {
  auto lits = clause.getLits();
  assert(lits.size() >= 2 && "cannot watch a unit clause");
  // The first two literals are kept as the watched pair. Propagation may swap
  // these positions later, but attach always starts from the canonical prefix.
  watches[lits[0].watchIndex()].push_back({&clause, lits[1]});
  watches[lits[1].watchIndex()].push_back({&clause, lits[0]});
}

Assignment NativeSATSolver::currentAssignment(Lit lit) const {
  // Literal truth is derived from the underlying variable assignment and then
  // adjusted for sign, so +1 means satisfied and -1 means falsified.
  Assignment value = stateFor(lit).assignment;
  switch (value) {
  case Assignment::Unassigned:
    return Assignment::Unassigned;
  case Assignment::True:
    return lit.isNegated() ? Assignment::False : Assignment::True;
  case Assignment::False:
    return lit.isNegated() ? Assignment::True : Assignment::False;
  }
}

bool NativeSATSolver::enqueue(Lit lit, Clause *reason) {
  int var = lit.var();
  Assignment wanted = assignmentFor(lit);
  auto &state = variables[var];
  Assignment current = state.assignment;
  if (current == wanted)
    return true;
  if (current != Assignment::Unassigned)
    return false;
  // The trail is the single source of truth for assignment order. Backtracking
  // and propagation both replay state from it rather than keeping copies.
  state.assignment = wanted;
  state.polarity = assignmentToInt(wanted);
  state.level = decisionLevel();
  state.reason = reason;
  trail.push_back(lit);
  return true;
}

Clause *NativeSATSolver::propagate() {
  while (nextPropagateIndex < trail.size()) {
    // The trail entry at nextPropagateIndex just became true, so only clauses
    // watching its
    // negation can have changed status.
    Lit lit = trail[nextPropagateIndex++];
    ++solverStats.numPropagations;
    auto &watchList = watches[lit.negated().watchIndex()];
    size_t write = 0;
    for (size_t read = 0; read < watchList.size(); ++read) {
      Watcher watcher = watchList[read];
      Clause *clause = watcher.clause;
      if (currentAssignment(watcher.blocker) == Assignment::True) {
        watchList[write++] = watcher;
        continue;
      }

      auto lits = clause->getLits();
      if (lits[0] == lit.negated())
        std::swap(lits[0], lits[1]);
      assert(lits[1] == lit.negated() &&
             "watched literal must be in second position");

      Lit first = lits[0];
      if (currentAssignment(first) == Assignment::True) {
        watchList[write++] = {clause, first};
        continue;
      }

      bool foundWatch = false;
      for (size_t idx = 2, e = lits.size(); idx != e; ++idx) {
        Lit candidate = lits[idx];
        if (currentAssignment(candidate) == Assignment::False)
          continue;
        // Move the watch away from the falsified literal. The old second watch
        // is swapped into the candidate slot so the first two entries remain
        // the watched pair.
        lits[1] = candidate;
        lits[idx] = lit.negated();
        watches[candidate.watchIndex()].push_back({clause, first});
        foundWatch = true;
        break;
      }
      if (foundWatch)
        continue;

      watchList[write++] = {clause, first};
      if (currentAssignment(first) == Assignment::False) {
        watchList.resize(write);
        return clause;
      }
      if (!enqueue(first, clause)) {
        watchList.resize(write);
        return clause;
      }
    }
    watchList.resize(write);
  }
  return nullptr;
}

void NativeSATSolver::analyze(Clause *conflict,
                              llvm::SmallVectorImpl<Lit> &learntClause,
                              int &backtrackLevel) {
  assert(conflict && "analyze requires a non-null conflict clause");
  // `unresolvedAtCurrentLevel` counts literals from the current decision level
  // that still need resolution before reaching the first UIP.
  learntClause.clear();
  learntClause.push_back(Lit()); // Placeholder for the final asserting literal.
  int unresolvedAtCurrentLevel = 0;
  Lit uipLit;
  size_t trailIndex = trail.size();
  const int currentLevel = decisionLevel();

  // Resolve conflict clauses until we isolate the first UIP literal.
  // Each iteration picks one seen literal from the trail (latest first),
  // replaces the current conflict with that literal's reason clause, and
  // repeats until only one current-level literal remains unresolved.
  while (true) {
    // Collect literals from the current conflict clause:
    // - literals from the current level increase the unresolved count,
    // - literals from lower levels become part of the learned clause.
    for (Lit lit : conflict->getLits()) {
      int var = lit.var();
      auto &state = variables[var];
      if (state.seen || state.level == 0)
        continue;
      state.seen = true;
      bumpVariableActivity(var);
      if (state.level == currentLevel)
        ++unresolvedAtCurrentLevel;
      else
        learntClause.push_back(lit);
    }

    // Walk backward on the trail to find the most recent literal that was
    // marked `seen`. This is the next literal to resolve away.
    do {
      assert(trailIndex > 0 && "expected a seen literal on the trail");
      uipLit = trail[--trailIndex];
    } while (!stateFor(uipLit).seen);

    auto &uipState = stateFor(uipLit);
    uipState.seen = false;
    // Continue resolution through the implication reason of `uipLit`.
    conflict = uipState.reason;
    // Stop at the first UIP (no unresolved current-level literals left), or if
    // there is no reason clause left to resolve through.
    if (--unresolvedAtCurrentLevel <= 0 || !conflict)
      break;
  }

  learntClause[0] = uipLit.negated();
  backtrackLevel = computeBacktrackLevel(learntClause);

  for (Lit lit : learntClause)
    stateFor(lit).seen = false;
}

int NativeSATSolver::computeBacktrackLevel(
    llvm::SmallVectorImpl<Lit> &learntClause) const {
  // The learned clause is arranged so element 0 is the asserting literal and
  // element 1 is the highest remaining decision level; backtracking there
  // makes the clause unit immediately.
  if (learntClause.size() == 1)
    return 0;

  size_t bestIndex = 1;
  for (size_t i = 2; i < learntClause.size(); ++i)
    if (stateFor(learntClause[i]).level >
        stateFor(learntClause[bestIndex]).level)
      bestIndex = i;
  // Maintain the learned-clause layout invariant:
  // - learntClause[0] is the asserting literal.
  // - learntClause[1] is the highest-level remaining literal.
  // Backtracking to level(learntClause[1]) then makes learntClause[0] unit
  // immediately, and `attachClause` starts with this canonical watched pair.
  std::swap(learntClause[1], learntClause[bestIndex]);
  return stateFor(learntClause[1]).level;
}

void NativeSATSolver::bumpVariableActivity(int var) {
  variables[var].activity += variableActivityIncrement;
  if (variables[var].activity > kActivityRescaleThreshold) {
    for (int idx = 1; idx <= maxVariable; ++idx)
      variables[idx].activity *= kActivityRescaleFactor;
    variableActivityIncrement *= kActivityRescaleFactor;
  }

  // Every variable is decayed uniformly, and only `var` is increased, so it is
  // sufficient to restore heap order for that one entry.
  variableOrder.increase(var);
}

void NativeSATSolver::decayVariableActivity() {
  variableActivityIncrement /= options.variableDecay;
}

std::optional<Lit> NativeSATSolver::pickBranchLiteral() {
  // The heap can contain stale entries for variables that became assigned
  // after insertion. Pop until an unassigned variable is found, then branch
  // using the saved polarity from the last time that variable was forced or
  // decided.
  while (!variableOrder.empty()) {
    auto var = variableOrder.pop();
    if (variables[var].assignment != Assignment::Unassigned)
      continue;
    return variables[var].polarity >= 0 ? Lit::fromRaw(var)
                                        : Lit::fromRaw(-var);
  }

  // Defensive fallback: the heap should normally contain every unassigned
  // variable, but a linear scan keeps search robust if that invariant is
  // temporarily violated.
  for (int var = 1; var <= maxVariable; ++var)
    if (variables[var].assignment == Assignment::Unassigned)
      return variables[var].polarity >= 0 ? Lit::fromRaw(var)
                                          : Lit::fromRaw(-var);
  return std::nullopt;
}

void NativeSATSolver::newDecisionLevel() {
  trailLimits.push_back(trail.size());
}

int NativeSATSolver::decisionLevel() const {
  return static_cast<int>(trailLimits.size());
}

void NativeSATSolver::cancelUntil(int level) {
  if (decisionLevel() <= level)
    return;
  size_t newTrailSize = trailLimits[level];
  // Unassigned variables are reinserted into the branching heap lazily here
  // instead of rebuilding the heap from scratch after every conflict.
  for (size_t i = trail.size(); i > newTrailSize; --i) {
    int var = trail[i - 1].var();
    auto &state = variables[var];
    state.assignment = Assignment::Unassigned;
    state.level = 0;
    state.reason = nullptr;
    // TODO: Consider batch insertions into the heap if this becomes a
    // bottleneck.
    variableOrder.insert(var);
  }
  trail.resize(newTrailSize);
  trailLimits.resize(level);
  nextPropagateIndex = trail.size();
}

void NativeSATSolver::resetLastSolveState() {
  // Any API action that mutates query state invalidates the previous SAT
  // result and its derived model.
  lastResult = kUNKNOWN;
}

//===----------------------------------------------------------------------===//
// Z3 Backend
//===----------------------------------------------------------------------===//

#if LLVM_WITH_Z3

class Z3SATSolver : public IncrementalSATSolver {
public:
  Z3SATSolver();
  ~Z3SATSolver() override;

  void add(int lit) override;
  void assume(int lit) override;
  Result solve() override;
  Result solve(llvm::ArrayRef<int> assumptions) override;
  int val(int v) const override;
  void reserveVars(int maxVar) override;
  const SATSolverStats &stats() const override;

private:
  void clearSolveScope();
  int newVariable();
  llvm::SMTExprRef literalToExpr(int lit);
  void addClauseInternal(llvm::ArrayRef<int> lits);

  llvm::SMTSolverRef solver;
  llvm::SmallVector<llvm::SMTExprRef> variables;
  llvm::SmallVector<int> assumptions;
  llvm::SmallVector<int> clauseBuffer;
  SATSolverStats solverStats;
  int maxVariable = 0;
  Result lastResult = kUNKNOWN;
  bool solveScopeActive = false;
};

Z3SATSolver::Z3SATSolver() : solver(llvm::CreateZ3Solver()) {}

Z3SATSolver::~Z3SATSolver() { clearSolveScope(); }

void Z3SATSolver::add(int lit) {
  clearSolveScope();
  if (lit == 0) {
    addClauseInternal(clauseBuffer);
    clauseBuffer.clear();
    return;
  }

  reserveVars(std::abs(lit));
  clauseBuffer.push_back(lit);
}

void Z3SATSolver::assume(int lit) {
  clearSolveScope();
  if (lit == 0)
    return;
  assumptions.push_back(lit);
}

IncrementalSATSolver::Result Z3SATSolver::solve() {
  auto localAssumptions = assumptions;
  assumptions.clear();
  return solve(localAssumptions);
}

IncrementalSATSolver::Result
Z3SATSolver::solve(llvm::ArrayRef<int> assumptions) {
  ++solverStats.numSolveCalls;
  clearSolveScope();
  solver->push();
  solveScopeActive = true;
  for (int lit : assumptions)
    solver->addConstraint(literalToExpr(lit));
  auto result = solver->check();
  if (!result)
    return lastResult = kUNKNOWN;
  if (*result)
    return lastResult = kSAT;
  return lastResult = kUNSAT;
}

int Z3SATSolver::val(int v) const {
  if (lastResult != kSAT || v <= 0 || v > maxVariable)
    return 0;
  llvm::APSInt value(llvm::APInt(1, 0), true);
  // Z3 returns an interpretation for all variables, even those not involved
  // in the problem. If the variable is not involved, return 0 to indicate
  // "undefined" rather than a potentially misleading true/false value.
  if (!solver->getInterpretation(variables[v - 1], value))
    return 0;
  return value != 0 ? v : -v;
}

void Z3SATSolver::reserveVars(int maxVar) {
  if (maxVar <= maxVariable)
    return;
  while (static_cast<int>(variables.size()) < maxVar)
    newVariable();
  maxVariable = maxVar;
}

const SATSolverStats &Z3SATSolver::stats() const { return solverStats; }

void Z3SATSolver::clearSolveScope() {
  if (!solveScopeActive)
    return;
  solver->pop();
  solveScopeActive = false;
  lastResult = kUNKNOWN;
}

int Z3SATSolver::newVariable() {
  int varIndex = static_cast<int>(variables.size()) + 1;
  std::string name = "v" + std::to_string(varIndex);
  variables.push_back(solver->mkSymbol(name.c_str(), solver->getBoolSort()));
  return varIndex;
}

llvm::SMTExprRef Z3SATSolver::literalToExpr(int lit) {
  int absLit = std::abs(lit);
  // Ensure variable exists for this literal.
  reserveVars(absLit);
  auto *variable = variables[absLit - 1];
  return lit > 0 ? variable : solver->mkNot(variable);
}

void Z3SATSolver::addClauseInternal(llvm::ArrayRef<int> lits) {
  if (lits.empty()) {
    solver->addConstraint(solver->mkBoolean(false));
    return;
  }

  llvm::SMTExprRef clause = nullptr;
  for (int lit : lits) {
    if (lit == 0)
      continue;
    auto *expr = literalToExpr(lit);
    clause = clause ? solver->mkOr(clause, expr) : expr;
  }

  if (!clause) {
    solver->addConstraint(solver->mkBoolean(false));
    return;
  }
  solver->addConstraint(clause);
}

#endif // LLVM_WITH_Z3

} // namespace

const SATSolverStats &IncrementalSATSolver::stats() const {
  return kEmptyStats;
}

std::unique_ptr<IncrementalSATSolver>
createNativeSATSolver(const NativeSATSolverOptions &options) {
  return std::make_unique<NativeSATSolver>(options);
}

std::unique_ptr<IncrementalSATSolver> createZ3SATSolver() {
#if LLVM_WITH_Z3
  return std::make_unique<Z3SATSolver>();
#else
  return {};
#endif
}

} // namespace circt
