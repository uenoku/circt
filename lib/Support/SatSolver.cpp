//===- SatSolver.cpp - Lightweight CDCL SAT Solver ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SatSolver.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>

using namespace circt;

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void MiniSATSolver::add(int lit) {
  if (lit == 0) {
    commitClause();
    clauseBuf.clear();
  } else {
    ensureVar(std::abs(lit));
    clauseBuf.push_back(lit);
  }
}

void MiniSATSolver::assume(int lit) {
  assert(lit != 0 && "assumption literal must be non-zero");
  ensureVar(std::abs(lit));
  assumptionBuf.push_back(lit);
}

MiniSATSolver::Result MiniSATSolver::solve(int64_t confLimit) {
  Result result = solveImpl(confLimit);
  assumptionBuf.clear();
  return result;
}

int MiniSATSolver::val(int v) const {
  assert(v > 0 && v <= numVars && "invalid variable index");
  return vars[v - 1].modelVal == kTrue ? v : -v;
}

void MiniSATSolver::bookmark() {
  // Verify preconditions
  assert(decisionLevel() == 0 && "bookmark requires decision level 0");
  assert(propagHead == static_cast<int>(trail.size()) &&
         "bookmark requires completed propagation");

  // Save current state
  Bookmark bm;
  bm.numVars = numVars;
  bm.numProblemClauses = static_cast<int>(problemClauses.size());
  bm.trailSize = static_cast<int>(trail.size());
  bm.propagHead = propagHead;
  bm.varActInc = varActInc;
  bm.claActInc = claActInc;

  // Save watch list sizes for all literals
  bm.watchListSizes.resize(2 * numVars);
  for (int i = 0; i < 2 * numVars; i++) {
    bm.watchListSizes[i] = watchLists[i].size();
  }

  bookmark_ = std::move(bm);
}

void MiniSATSolver::rollback() {
  assert(bookmark_.has_value() && "rollback requires active bookmark");

  const Bookmark &bm = *bookmark_;

  // 1. Remove temporary variables
  numVars = bm.numVars;
  vars.resize(bm.numVars);
  watchLists.resize(2 * bm.numVars);

  // 2. Remove temporary problem clauses
  problemClauses.resize(bm.numProblemClauses);
  numClauses = bm.numProblemClauses; // Binary clauses still counted elsewhere

  // 3. Clean learned clauses that reference removed variables
  for (auto &clause : learntClauses) {
    if (clause.size == 0)
      continue; // Already deleted
    for (auto lit : clause.lits) {
      if (litVar(lit) >= bm.numVars) {
        clause.size = 0; // Mark as deleted
        break;
      }
    }
  }

  // 4. Restore watch lists
  for (int lit = 0; lit < 2 * bm.numVars; lit++) {
    auto &wlist = watchLists[lit];

    // Truncate to saved size first (removes watches added after bookmark)
    if (wlist.size() > bm.watchListSizes[lit]) {
      wlist.resize(bm.watchListSizes[lit]);
    }

    // Remove watches to deleted clauses or removed variables
    llvm::erase_if(wlist, [&](const WatchEntry &w) {
      if (w.isBinary()) {
        // Check if other literal references removed variable
        int otherLit = encodeLit(w.otherLit());
        return litVar(otherLit) >= bm.numVars;
      }
      uint32_t idx = w.clauseIdx();
      if (idx & kLearntTag) {
        // Check learned clause
        return getClause(idx).size == 0;
      }
      // Check problem clause
      return idx >= static_cast<uint32_t>(bm.numProblemClauses);
    });
  }

  // 5. Clear trail and reset propagation
  trail.resize(bm.trailSize);
  trailLimits.clear();
  propagHead = bm.propagHead;
  rootLevel = 0;

  // 6. Clear variable assignments for all variables
  for (int i = 0; i < bm.numVars; i++) {
    vars[i].assign = kUndef;
    vars[i].level = 0;
    vars[i].reason = kNoReason;
    // Keep polarity (phase saving helps)
    // Keep activity (VSIDS warmth!)
    vars[i].seen = 0;
  }

  // 7. Restore activity decay state
  varActInc = bm.varActInc;
  claActInc = bm.claActInc;

  // 8. Rebuild VSIDS heap with remaining variables
  vsidsHeap.clear();
  vsidsHeap.grow(bm.numVars);
  auto sf = [this](int x) { return vars[x].activity; };
  for (int v = 0; v < bm.numVars; v++) {
    vsidsHeap.insert(v, sf);
  }

  // Keep bookmark active for potential future rollbacks
}

//===----------------------------------------------------------------------===//
// Variable management
//===----------------------------------------------------------------------===//

void MiniSATSolver::ensureVar(int v) {
  assert(v > 0 && "variables are 1-indexed");
  if (v <= numVars)
    return;
  int old = numVars;
  numVars = v;
  vars.resize(numVars);
  watchLists.resize(numVars * 2);
  auto sf = [this](int x) { return vars[x].activity; };
  vsidsHeap.grow(numVars);
  for (int i = old; i < numVars; i++)
    vsidsHeap.insert(i, sf);
}

//===----------------------------------------------------------------------===//
// VSIDS activity
//===----------------------------------------------------------------------===//

void MiniSATSolver::bumpVarActivity(int v) {
  vars[v].activity += varActInc;
  if (vars[v].activity > kVarActRescale) [[unlikely]] {
    for (auto &var : vars)
      var.activity *= kVarActRescaleInv;
    varActInc *= kVarActRescaleInv;
  }
  auto sf = [this](int x) { return vars[x].activity; };
  vsidsHeap.increased(v, sf);
}

void MiniSATSolver::decayVarActivity() { varActInc *= kVarActDecayFactor; }

void MiniSATSolver::bumpClauseActivity(Clause &c) {
  c.activity += static_cast<float>(claActInc);
  if (c.activity > static_cast<float>(kClaActRescale)) [[unlikely]] {
    for (auto &cl : learntClauses)
      cl.activity *= static_cast<float>(kClaActRescaleInv);
    claActInc *= kClaActRescaleInv;
  }
}

void MiniSATSolver::decayClauseActivity() { claActInc *= kClaActDecayFactor; }

//===----------------------------------------------------------------------===//
// Trail / Assignment
//===----------------------------------------------------------------------===//

bool MiniSATSolver::enqueue(int lit, int reason) {
  int v = litVar(lit);
  auto &var = vars[v];
  if (var.assign != kUndef) {
    // Check if current assignment is consistent with the literal being enqueued
    Assign expectedAssign = (lit & 1) ? kFalse : kTrue;
    return var.assign == expectedAssign;
  }
  var.assign = (lit & 1) ? kFalse : kTrue;
  var.level = decisionLevel();
  var.reason = reason;
  trail.push_back(lit);
  return true;
}

void MiniSATSolver::newDecisionLevel() {
  trailLimits.push_back(static_cast<int>(trail.size()));
}

void MiniSATSolver::backtrack(int level) {
  if (decisionLevel() <= level)
    return;
  auto sf = [this](int x) { return vars[x].activity; };
  for (int i = static_cast<int>(trail.size()) - 1; i >= trailLimits[level];
       i--) {
    int v = litVar(trail[i]);
    auto &var = vars[v];
    var.assign = kUndef;
    var.reason = kNoReason;
    var.polarity = (trail[i] & 1) ? 1 : 0;
    if (!vsidsHeap.contains(v))
      vsidsHeap.insert(v, sf);
  }
  trail.resize(trailLimits[level]);
  trailLimits.resize(level);
  propagHead = static_cast<int>(trail.size());
}

//===----------------------------------------------------------------------===//
// Boolean Constraint Propagation
//===----------------------------------------------------------------------===//

MiniSATSolver::Conflict MiniSATSolver::propagate() {
  while (propagHead < static_cast<int>(trail.size())) {
    int propagatedLit = trail[propagHead++];
    int falsifiedLit = negLit(propagatedLit);

    // Iterate over all clauses watching falsifiedLit. Use two-pointer technique
    // to compact the watch list while processing: watches that stay are copied
    // from readIdx to writeIdx, while watches that move elsewhere are skipped.
    auto &watchList = watchLists[falsifiedLit];
    size_t readIdx = 0, writeIdx = 0;

    while (readIdx < watchList.size()) {
      // Use reference to avoid copy
      const WatchEntry &entry = watchList[readIdx];

      if (entry.isBinary()) [[unlikely]] {
        int otherLit = encodeLit(entry.otherLit());
        if (!enqueue(otherLit, kNoReason)) [[unlikely]] {
          // Both literals false: conflict
          watchList[writeIdx++] = watchList[readIdx++];
          std::copy(watchList.begin() + readIdx, watchList.end(),
                    watchList.begin() + writeIdx);
          watchList.resize(writeIdx + (watchList.size() - readIdx));
          return Conflict{Conflict::kBinary, {falsifiedLit, otherLit}};
        }
        watchList[writeIdx++] = watchList[readIdx++];
        continue;
      }

      // Non-binary clause: try to find a new non-false literal to watch
      uint32_t clauseIdx = entry.clauseIdx();
      Clause &clause = getClause(clauseIdx);
      int *lits = clause.lits.data();
      int size = static_cast<int>(clause.size);

      // Ensure falsifiedLit is at position 1 (invariant: lits[0] and lits[1]
      // are the watched literals)
      if (lits[0] == falsifiedLit)
        std::swap(lits[0], lits[1]);
      assert(lits[1] == falsifiedLit && "watch list invariant violated");

      // If the other watched literal is true, clause is satisfied
      if (evalLit(lits[0]) == kTrue) [[likely]] {
        watchList[writeIdx++] = watchList[readIdx++];
        continue;
      }

      // Search for a new literal to watch (must not be false)
      bool foundReplacement = false;
      for (int k = 2; k < size; ++k) {
        if (evalLit(lits[k]) != kFalse) {
          std::swap(lits[1], lits[k]);
          watchLists[lits[1]].push_back(WatchEntry(clauseIdx));
          foundReplacement = true;
          ++readIdx; // Don't copy to writeIdx: watch moved elsewhere
          break;
        }
      }

      if (foundReplacement) [[likely]]
        continue;

      // No replacement found: clause is unit or conflicting
      watchList[writeIdx++] = watchList[readIdx++];
      if (!enqueue(lits[0], static_cast<int>(clauseIdx))) [[unlikely]] {
        // Conflict: restore remaining watches and return
        std::copy(watchList.begin() + readIdx, watchList.end(),
                  watchList.begin() + writeIdx);
        watchList.resize(writeIdx + (watchList.size() - readIdx));
        return Conflict{static_cast<int>(clauseIdx), {}};
      }
    }

    watchList.resize(writeIdx);
  }

  return Conflict{};
}

//===----------------------------------------------------------------------===//
// Conflict Analysis (1UIP with self-subsumption minimization)
//===----------------------------------------------------------------------===//

void MiniSATSolver::analyze(const Conflict &confl,
                            llvm::SmallVectorImpl<int> &outLearnt,
                            int &outBackLevel) {
  int pathCount = 0; // Literals at current decision level still to resolve
  int assertingLit = -1;
  int trailIdx = static_cast<int>(trail.size()) - 1;
  outLearnt.clear();
  outLearnt.push_back(-1); // Placeholder for 1UIP asserting literal

  auto processLit = [&](int lit) {
    int v = litVar(lit);
    auto &var = vars[v];
    if (!var.seen && var.level > 0) {
      var.seen = 1;
      bumpVarActivity(v);
      if (var.level == decisionLevel())
        pathCount++;
      else
        outLearnt.push_back(lit);
    }
  };

  // Seed with conflict clause
  if (confl.isBinary()) {
    processLit(confl.binLits[0]);
    processLit(confl.binLits[1]);
  } else {
    Clause &c = getClause(confl.index);
    if (c.learnt)
      bumpClauseActivity(c);
    for (unsigned k = 0; k < c.size; k++)
      processLit(c.lits[k]);
  }

  // Walk trail backwards to find first UIP
  while (pathCount > 0) {
    while (!vars[litVar(trail[trailIdx])].seen)
      trailIdx--;
    assertingLit = trail[trailIdx--];
    vars[litVar(assertingLit)].seen = 0;
    pathCount--;

    if (pathCount > 0) {
      int reason = vars[litVar(assertingLit)].reason;
      if (reason >= 0) {
        Clause &c = getClause(reason);
        if (c.learnt)
          bumpClauseActivity(c);
        for (unsigned k = 0; k < c.size; k++)
          if (c.lits[k] != assertingLit)
            processLit(c.lits[k]);
      }
    }
  }

  outLearnt[0] = negLit(assertingLit);

  // Self-subsumption minimization: remove literals whose reason clauses
  // are subsumed by other literals in the learnt clause
  unsigned levelMask = 0;
  for (size_t i = 1; i < outLearnt.size(); i++)
    levelMask |= 1u << (vars[litVar(outLearnt[i])].level & 31);

  // Reuse workspace vectors to avoid allocations
  analyzeStack.clear();
  analyzeToClear.clear();
  size_t writePos = 1;
  for (size_t i = 1; i < outLearnt.size(); i++) {
    int v = litVar(outLearnt[i]);
    if (vars[v].reason < 0 ||
        !isRedundant(v, levelMask, analyzeStack, analyzeToClear))
      outLearnt[writePos++] = outLearnt[i];
  }
  outLearnt.resize(writePos);

  for (auto lit : outLearnt)
    vars[litVar(lit)].seen = 0;

  // Backtrack level is second-highest decision level in learnt clause
  if (outLearnt.size() == 1) {
    outBackLevel = 0;
  } else {
    int maxIdx = 1;
    int maxLev = vars[litVar(outLearnt[1])].level;
    for (size_t i = 2; i < outLearnt.size(); i++) {
      if (vars[litVar(outLearnt[i])].level > maxLev) {
        maxLev = vars[litVar(outLearnt[i])].level;
        maxIdx = static_cast<int>(i);
      }
    }
    std::swap(outLearnt[1], outLearnt[maxIdx]);
    outBackLevel = maxLev;
  }
}

bool MiniSATSolver::isRedundant(int v, unsigned levelMask,
                                llvm::SmallVectorImpl<int> &stack,
                                llvm::SmallVectorImpl<int> &toClear) {
  stack.clear();
  stack.push_back(v);
  int top = static_cast<int>(toClear.size());

  while (!stack.empty()) {
    int x = stack.pop_back_val();
    int reason = vars[x].reason;
    if (reason < 0) {
      cleanupRedundancyCheck(top, toClear);
      return false;
    }

    Clause &c = getClause(reason);
    for (unsigned k = 0; k < c.size; k++) {
      int lv = litVar(c.lits[k]);
      auto &var = vars[lv];
      if (lv == x || var.seen || var.level == 0)
        continue;
      if (var.reason >= 0 && ((1u << (var.level & 31)) & levelMask)) {
        var.seen = 1;
        stack.push_back(lv);
        toClear.push_back(lv);
      } else {
        cleanupRedundancyCheck(top, toClear);
        return false;
      }
    }
  }
  return true;
}

void MiniSATSolver::cleanupRedundancyCheck(
    int top, llvm::SmallVectorImpl<int> &toClear) {
  for (int i = top; i < static_cast<int>(toClear.size()); i++)
    vars[toClear[i]].seen = 0;
  toClear.resize(top);
}

//===----------------------------------------------------------------------===//
// Clause construction
//===----------------------------------------------------------------------===//

void MiniSATSolver::addWatchPair(int lit0, int lit1, uint32_t clauseIdx) {
  watchLists[lit0].push_back(WatchEntry(clauseIdx));
  watchLists[lit1].push_back(WatchEntry(clauseIdx));
}

void MiniSATSolver::addBinaryWatch(int lit0, int lit1) {
  watchLists[lit0].push_back(
      WatchEntry(kBinaryTag | static_cast<uint32_t>(lit1)));
  watchLists[lit1].push_back(
      WatchEntry(kBinaryTag | static_cast<uint32_t>(lit0)));
}

void MiniSATSolver::commitClause() {
  if (!ok)
    return;

  // Encode to internal representation, sort, and remove duplicates
  llvm::SmallVector<int> lits;
  for (int lit : clauseBuf)
    lits.push_back(encodeLit(lit));
  llvm::sort(lits);

  size_t writePos = 0;
  for (size_t i = 0; i < lits.size(); i++) {
    if (writePos > 0 && lits[i] == lits[writePos - 1])
      continue; // Duplicate
    if (writePos > 0 && lits[i] == negLit(lits[writePos - 1]))
      return; // Tautology
    if (decisionLevel() == 0 && evalLit(lits[i]) == kTrue)
      return; // Already satisfied
    if (decisionLevel() == 0 && evalLit(lits[i]) == kFalse)
      continue; // Remove falsified literal
    lits[writePos++] = lits[i];
  }
  lits.resize(writePos);

  if (lits.empty()) {
    ok = false;
    return;
  }
  if (lits.size() == 1) {
    ok = enqueue(lits[0], kNoReason);
    return;
  }

  numClauses++;

  if (lits.size() == 2) {
    addBinaryWatch(lits[0], lits[1]);
    return;
  }

  auto clauseIdx = static_cast<uint32_t>(problemClauses.size());
  problemClauses.push_back(
      {static_cast<uint32_t>(lits.size()), /*learnt=*/false, 0.0f,
       llvm::SmallVector<int, 4>(lits.begin(), lits.end())});
  addWatchPair(lits[0], lits[1], clauseIdx);
}

void MiniSATSolver::recordLearnt(llvm::SmallVectorImpl<int> &lits) {
  if (lits.size() == 1) {
    enqueue(lits[0], kNoReason);
    return;
  }
  if (lits.size() == 2) {
    addBinaryWatch(lits[0], lits[1]);
    enqueue(lits[0], kNoReason);
    return;
  }

  auto ci = static_cast<uint32_t>(learntClauses.size()) | kLearntTag;
  learntClauses.push_back(
      {static_cast<uint32_t>(lits.size()), /*learnt=*/true, 0.0f,
       llvm::SmallVector<int, 4>(lits.begin(), lits.end())});
  addWatchPair(lits[0], lits[1], ci);
  enqueue(lits[0], static_cast<int>(ci));
}

//===----------------------------------------------------------------------===//
// Learned clause reduction
//===----------------------------------------------------------------------===//

void MiniSATSolver::reduceLearnts() {
  // Sort indices by activity (ascending) to identify low-activity clauses
  llvm::SmallVector<int> order(learntClauses.size());
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
      continue; // Keep binary clauses
    // Keep if clause is the reason for a current assignment
    int v = litVar(c.lits[0]);
    if (vars[v].assign != kUndef &&
        vars[v].reason == static_cast<int>(idx | kLearntTag))
      continue;
    c.size = 0; // Mark as deleted
    removed++;
  }

  // Remove watches pointing to deleted clauses
  for (auto &ws : watchLists) {
    llvm::erase_if(ws, [&](WatchEntry &w) {
      return !w.isBinary() && getClause(w.clauseIdx()).size == 0;
    });
  }
}

//===----------------------------------------------------------------------===//
// Decision
//===----------------------------------------------------------------------===//

int MiniSATSolver::pickBranchVar() {
  auto sf = [this](int x) { return vars[x].activity; };
  while (!vsidsHeap.empty()) {
    int v = vsidsHeap.pop(sf);
    if (vars[v].assign == kUndef)
      return v;
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// Luby restart sequence
//===----------------------------------------------------------------------===//

double MiniSATSolver::luby(double y, int i) {
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

//===----------------------------------------------------------------------===//
// Main search loop
//===----------------------------------------------------------------------===//

MiniSATSolver::Result MiniSATSolver::search(int64_t confBudget) {
  int64_t conflicts = 0;
  llvm::SmallVector<int> learntClause;

  for (;;) {
    Conflict confl = propagate();
    if (!confl.isNone()) [[unlikely]] {
      // Handle conflict
      conflicts++;
      if (decisionLevel() == rootLevel) [[unlikely]]
        return kUNSAT; // Conflict at assumption level
      int backLevel;
      analyze(confl, learntClause, backLevel);
      backtrack(std::max(backLevel, rootLevel));
      recordLearnt(learntClause);
      decayVarActivity();
      decayClauseActivity();
    } else {
      // No conflict: check budget, reduce clauses, or make decision
      if (confBudget >= 0 && conflicts >= confBudget) [[unlikely]] {
        backtrack(rootLevel);
        return kUNKNOWN;
      }
      if (static_cast<int>(learntClauses.size()) >
          numVars * kLearntsMultiplier + kLearntsOffset) [[unlikely]]
        reduceLearnts();

      int next = pickBranchVar();
      if (next == -1) [[unlikely]] {
        // All variables assigned: SAT
        for (auto &var : vars)
          var.modelVal = var.assign;
        backtrack(rootLevel);
        return kSAT;
      }
      newDecisionLevel();
      enqueue(vars[next].polarity ? (2 * next + 1) : (2 * next),
              kDecisionReason);
    }
  }
}

MiniSATSolver::Result MiniSATSolver::solveImpl(int64_t confLimit) {
  if (!ok)
    return kUNSAT;

  if (!propagate().isNone()) {
    ok = false;
    return kUNSAT;
  }

  // Enqueue assumptions as decisions with propagation after each
  rootLevel = 0;
  for (int extLit : assumptionBuf) {
    int enc = encodeLit(extLit);
    newDecisionLevel();
    rootLevel = decisionLevel();
    if (!enqueue(enc, kDecisionReason) || !propagate().isNone()) {
      backtrack(0);
      rootLevel = 0;
      return kUNSAT;
    }
  }

  // Main search with Luby restarts
  Result result = kUNKNOWN;
  for (int iter = 0; result == kUNKNOWN; iter++) {
    int64_t budget = static_cast<int64_t>(kLubyBase * luby(kLubyFactor, iter));
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
