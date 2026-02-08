//===- SatSolver.cpp - Lightweight CDCL SAT Solver --------------*- C++ -*-===//
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
// MiniSATSolver - Public API
//===----------------------------------------------------------------------===//

void MiniSATSolver::add(int lit) {
  if (lit == 0) {
    commitClause();
    clauseBuf.clear();
  } else {
    assert(lit != 0 && "Literal must be non-zero");
    ensureVar(std::abs(lit));
    clauseBuf.push_back(lit);
  }
}

void MiniSATSolver::assume(int lit) {
  assert(lit != 0 && "Assumption literal must be non-zero");
  ensureVar(std::abs(lit));
  assumptionBuf.push_back(lit);
}

MiniSATSolver::Result MiniSATSolver::solve(int64_t confLimit) {
  Result result = solveImpl(confLimit);
  assumptionBuf.clear();
  return result;
}

int MiniSATSolver::val(int v) const {
  assert(v > 0 && v <= numVars);
  return vars[v - 1].modelVal == kTrue ? v : -v;
}

//===----------------------------------------------------------------------===//
// Clause access
//===----------------------------------------------------------------------===//

MiniSATSolver::Clause &MiniSATSolver::getClause(uint32_t idx) {
  return (idx & kLearntTag) ? learntClauses[idx & kLearntMask]
                            : problemClauses[idx];
}

const MiniSATSolver::Clause &MiniSATSolver::getClause(uint32_t idx) const {
  return (idx & kLearntTag) ? learntClauses[idx & kLearntMask]
                            : problemClauses[idx];
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
  if (vars[v].activity > kVarActRescale) {
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
  if (c.activity > static_cast<float>(kClaActRescale)) {
    for (auto &cl : learntClauses)
      cl.activity *= static_cast<float>(kClaActRescaleInv);
    claActInc *= kClaActRescaleInv;
  }
}

void MiniSATSolver::decayClauseActivity() { claActInc *= kClaActDecayFactor; }

//===----------------------------------------------------------------------===//
// Trail / Assignment
//===----------------------------------------------------------------------===//

MiniSATSolver::Assign MiniSATSolver::evalLit(int enc) const {
  Assign a = vars[litVar(enc)].assign;
  if (a == kUndef)
    return kUndef;
  return static_cast<Assign>((enc & 1) ? -a : a);
}

int MiniSATSolver::decisionLevel() const {
  return static_cast<int>(trailLimits.size());
}

bool MiniSATSolver::enqueue(int p, int reason) {
  int v = litVar(p);
  auto &var = vars[v];
  if (var.assign != kUndef) {
    // Check if current assignment is consistent with the literal we're trying
    // to enqueue
    Assign expectedAssign = (p & 1) ? kFalse : kTrue;
    return var.assign == expectedAssign;
  }
  var.assign = (p & 1) ? kFalse : kTrue;
  var.level = decisionLevel();
  var.reason = reason;
  trail.push_back(p);
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
    int p = trail[propagHead++];
    int falseLit = negLit(p);

    auto &ws = watchLists[falseLit];
    size_t i = 0, j = 0;
    while (i < ws.size()) {
      WatchEntry w = ws[i];

      if (w.isBinary()) {
        int otherEnc = encodeLit(w.otherLit());
        if (!enqueue(otherEnc, kNoReason)) {
          Conflict confl;
          confl.index = Conflict::kBinary;
          confl.binLits[0] = falseLit;
          confl.binLits[1] = otherEnc;
          // Copy remaining watches.
          ws[j++] = ws[i++];
          while (i < ws.size())
            ws[j++] = ws[i++];
          ws.resize(j);
          return confl;
        }
        ws[j++] = ws[i++];
        continue;
      }

      uint32_t ci = w.clauseIdx();
      Clause &c = getClause(ci);
      int *lits = c.lits.data();
      int sz = static_cast<int>(c.size);

      // Put the false literal at position 1 (watched literals are at positions
      // 0 and 1).
      if (lits[0] == falseLit)
        std::swap(lits[0], lits[1]);
      assert(lits[1] == falseLit && "False literal must be at position 1");

      // Clause already satisfied by lits[0]?
      bool clauseSatisfied = (evalLit(lits[0]) == kTrue);
      if (clauseSatisfied) {
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
        return Conflict{static_cast<int>(ci), {}};
      }
    }
    ws.resize(j);
  }
  return Conflict{}; // no conflict
}

//===----------------------------------------------------------------------===//
// Conflict Analysis (1UIP)
//===----------------------------------------------------------------------===//

void MiniSATSolver::analyze(const Conflict &confl,
                            llvm::SmallVectorImpl<int> &outLearnt,
                            int &outBackLevel) {
  int pathCount = 0;
  int p = -1;
  int trailIdx = static_cast<int>(trail.size()) - 1;
  outLearnt.clear();
  outLearnt.push_back(-1); // placeholder for asserting literal

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

  // Seed with the conflict clause literals.
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

  // Walk trail backwards to find 1UIP.
  while (pathCount > 0) {
    while (!vars[litVar(trail[trailIdx])].seen)
      trailIdx--;
    p = trail[trailIdx--];
    vars[litVar(p)].seen = 0;
    pathCount--;

    if (pathCount > 0) {
      int reason = vars[litVar(p)].reason;
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
    levelMask |= 1u << (vars[litVar(outLearnt[i])].level & 31);

  llvm::SmallVector<int> stack;
  llvm::SmallVector<int> toClear;
  size_t writePos = 1;
  for (size_t i = 1; i < outLearnt.size(); i++) {
    int v = litVar(outLearnt[i]);
    if (vars[v].reason < 0 || !isRedundant(v, levelMask, stack, toClear))
      outLearnt[writePos++] = outLearnt[i];
  }
  outLearnt.resize(writePos);

  for (auto lit : outLearnt)
    vars[litVar(lit)].seen = 0;

  // Find backtrack level = second-highest decision level in learnt clause.
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

void MiniSATSolver::addWatchPair(int lit0, int lit1, uint32_t ci) {
  watchLists[lit0].push_back(WatchEntry(ci));
  watchLists[lit1].push_back(WatchEntry(ci));
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

  // Encode, sort, deduplicate.
  llvm::SmallVector<int> lits;
  for (int lit : clauseBuf)
    lits.push_back(encodeLit(lit));
  llvm::sort(lits);

  size_t w = 0;
  for (size_t i = 0; i < lits.size(); i++) {
    if (w > 0 && lits[i] == lits[w - 1])
      continue;
    if (w > 0 && lits[i] == negLit(lits[w - 1]))
      return; // tautology
    if (decisionLevel() == 0 && evalLit(lits[i]) == kTrue)
      return; // satisfied at root
    if (decisionLevel() == 0 && evalLit(lits[i]) == kFalse)
      continue; // falsified at root
    lits[w++] = lits[i];
  }
  lits.resize(w);

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

  auto ci = static_cast<uint32_t>(problemClauses.size());
  problemClauses.push_back(
      {static_cast<uint32_t>(lits.size()), /*learnt=*/false, 0.0f,
       llvm::SmallVector<int, 4>(lits.begin(), lits.end())});
  addWatchPair(lits[0], lits[1], ci);
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
  // Sort by activity ascending; remove the less active half.
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
      continue;
    // Don't remove if clause is the reason for a current assignment.
    int v = litVar(c.lits[0]);
    if (vars[v].assign != kUndef &&
        vars[v].reason == static_cast<int>(idx | kLearntTag))
      continue;
    c.size = 0; // mark deleted
    removed++;
  }

  // Purge dangling watch references.
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
    if (!confl.isNone()) {
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

  // Push assumptions as decision levels with BCP after each.
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
