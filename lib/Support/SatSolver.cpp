//===- SatSolver.cpp - Lightweight CDCL SAT Solver ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SatSolver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>

#define DEBUG_TYPE "sat-solver"

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

void MiniSATSolver::dumpDIMACS(llvm::raw_ostream &os,
                                llvm::ArrayRef<int> assumptions) const {
  // Count total clauses: problem + learnt + binary + assumptions
  size_t totalClauses = problemClauses.size() + learntClauses.size() +
                        binaryClauses.size() + learntBinaryClauses.size() +
                        assumptions.size();

  // DIMACS header
  os << "p cnf " << numVars << " " << totalClauses << "\n";

  // Problem clauses (non-binary)
  for (const auto &clause : problemClauses) {
    for (int lit : clause.lits)
      os << decodeLit(lit) << " ";
    os << "0\n";
  }

  // Binary problem clauses
  for (const auto &[lit0, lit1] : binaryClauses) {
    os << decodeLit(lit0) << " " << decodeLit(lit1) << " 0\n";
  }

  // Learned clauses (optional - helps reproducibility)
  for (const auto &clause : learntClauses) {
    for (int lit : clause.lits)
      os << decodeLit(lit) << " ";
    os << "0\n";
  }

  // Binary learned clauses
  for (const auto &[lit0, lit1] : learntBinaryClauses) {
    os << decodeLit(lit0) << " " << decodeLit(lit1) << " 0\n";
  }

  // Assumptions as unit clauses
  for (int lit : assumptions) {
    os << lit << " 0\n";
  }
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

  // Determine where to clear from:
  // - For level > 0: clear from trailLimits[level] (start of level+1)
  // - For level = 0: clear EVERYTHING (including unit propagations at level 0)
  int clearFrom;
  if (level == 0) {
    clearFrom = 0; // Clear entire trail when backtracking to level 0
  } else if (level < static_cast<int>(trailLimits.size())) {
    clearFrom = trailLimits[level];
  } else {
    clearFrom = static_cast<int>(trail.size()); // Nothing to clear
  }

  llvm::errs() << "    backtrack(" << level << "): decisionLevel="
               << decisionLevel() << " trail.size=" << trail.size()
               << " clearFrom=" << clearFrom << "\n";

  auto sf = [this](int x) { return vars[x].activity; };
  for (int i = static_cast<int>(trail.size()) - 1; i >= clearFrom; i--) {
    int v = litVar(trail[i]);
    auto &var = vars[v];
    var.assign = kUndef;
    var.reason = kNoReason;
    var.polarity = (trail[i] & 1) ? 1 : 0;
    if (!vsidsHeap.contains(v))
      vsidsHeap.insert(v, sf);
  }
  trail.resize(clearFrom);
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
        // Encode binary clause reason: use the other literal for conflict resolution
        // Store falsifiedLit as the reason (the literal that forced this unit propagation)
        int binaryReason = -(falsifiedLit + 3);
        if (!enqueue(otherLit, binaryReason)) [[unlikely]] {
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
      assert(lits[1] == falsifiedLit &&
             "watch list invariant violated");

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
        // Regular clause reason
        Clause &c = getClause(reason);
        if (c.learnt)
          bumpClauseActivity(c);
        for (unsigned k = 0; k < c.size; k++)
          if (c.lits[k] != assertingLit)
            processLit(c.lits[k]);
      } else if (reason <= -3) {
        // Binary clause reason: encoded as -(falsifiedLit + 3)
        int otherLit = -(reason + 3);
        processLit(otherLit);
      }
      // reason == -1 (kNoReason) or -2 (kDecisionReason): skip
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
    if (vars[v].reason < 0 || !isRedundant(v, levelMask, analyzeStack, analyzeToClear))
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

  // Optimize for small clauses (common in Tseitin encodings): use insertion
  // sort for <=6 literals to avoid qsort overhead. Benchmarking shows most
  // FRAIG clauses are binary/ternary, making this a significant win.
  if (lits.size() <= 6) {
    for (size_t i = 1; i < lits.size(); ++i) {
      int key = lits[i];
      size_t j = i;
      while (j > 0 && lits[j - 1] > key) {
        lits[j] = lits[j - 1];
        --j;
      }
      lits[j] = key;
    }
  } else {
    llvm::sort(lits);
  }

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
    binaryClauses.push_back({lits[0], lits[1]});
    addBinaryWatch(lits[0], lits[1]);
    // Debug: log binary clauses involving vars ~140 or ~57 (half of 280/115)
    int v0 = litVar(lits[0]), v1 = litVar(lits[1]);
    if ((v0 >= 135 && v0 <= 145) || (v1 >= 135 && v1 <= 145) ||
        (v0 >= 52 && v0 <= 62) || (v1 >= 52 && v1 <= 62)) {
      llvm::errs() << "Binary clause added: " << decodeLit(lits[0]) << " "
                   << decodeLit(lits[1]) << "\n";
    }
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
    learntBinaryClauses.push_back({lits[0], lits[1]});
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
  // Lazy heap rebuild after rollback (optimization: only rebuild when needed)
  if (heapNeedsRebuild) {
    vsidsHeap.grow(numVars);
    auto sf = [this](int x) { return vars[x].activity; };
    for (int v = 0; v < numVars; v++)
      vsidsHeap.insert(v, sf);
    heapNeedsRebuild = false;
  }

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
      if (decisionLevel() == rootLevel) [[unlikely]] {
        llvm::errs() << "    UNSAT: Conflict at root level! rootLevel="
                     << rootLevel << " decisionLevel=" << decisionLevel()
                     << " conflictType="
                     << (confl.isBinary() ? "binary" : "clause")
                     << " conflictIdx=" << confl.index << "\n";
        if (confl.isBinary()) {
          int lit0 = confl.binLits[0];
          int lit1 = confl.binLits[1];
          int v0 = litVar(lit0);
          int v1 = litVar(lit1);
          llvm::errs() << "      Binary conflict: " << decodeLit(lit0)
                       << " " << decodeLit(lit1) << "\n";
          llvm::errs() << "      Assignments: var" << v0 << "="
                       << static_cast<int>(vars[v0].assign)
                       << " (level=" << vars[v0].level << " reason=" << vars[v0].reason;
          if (vars[v0].reason >= 0) {
            Clause &c0 = getClause(vars[v0].reason);
            llvm::errs() << " clauseSize=" << c0.size;
          }
          llvm::errs() << "), var" << v1 << "="
                       << static_cast<int>(vars[v1].assign)
                       << " (level=" << vars[v1].level << " reason=" << vars[v1].reason;
          if (vars[v1].reason >= 0) {
            Clause &c1 = getClause(vars[v1].reason);
            llvm::errs() << " clauseSize=" << c1.size;
          }
          llvm::errs() << ")\n";
          // Check if the binary clause exists in our clause lists
          bool found = false;
          for (auto &[l0, l1] : binaryClauses) {
            if ((l0 == lit0 && l1 == lit1) || (l0 == lit1 && l1 == lit0)) {
              llvm::errs() << "      Found in binaryClauses (problem)\n";
              found = true;
              break;
            }
          }
          if (!found) {
            for (auto &[l0, l1] : learntBinaryClauses) {
              if ((l0 == lit0 && l1 == lit1) || (l0 == lit1 && l1 == lit0)) {
                llvm::errs() << "      Found in learntBinaryClauses\n";
                found = true;
                break;
              }
            }
          }
          if (!found) {
            llvm::errs() << "      WARNING: Binary clause NOT found in storage!\n";
          }
        } else if (confl.index >= 0) {
          auto &c = getClause(confl.index);
          llvm::errs() << "      Clause conflict: size=" << c.size
                       << " learnt=" << c.learnt << " lits=";
          for (size_t i = 0; i < c.size && i < 5; i++)
            llvm::errs() << decodeLit(c.lits[i]) << " ";
          if (c.size > 5)
            llvm::errs() << "...";
          llvm::errs() << "\n";
        }
        return kUNSAT; // Conflict at assumption level
      }
      int backLevel;
      analyze(confl, learntClause, backLevel);
      backtrack(std::max(backLevel, rootLevel));
      // Only record learned clauses if they don't involve assumption levels.
      // Learned clauses containing literals from assumption levels (1..rootLevel)
      // are contaminated and invalid for other assumption sets.
      if (backLevel < rootLevel) {
        // Don't learn - clause is tainted by assumptions
        LLVM_DEBUG(llvm::dbgs() << "LEARN: Skipping tainted clause (backLevel="
                                << backLevel << " < rootLevel=" << rootLevel
                                << ")\n");
      } else {
        recordLearnt(learntClause);
      }
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
  size_t totalWatches = 0;
  for (const auto &wl : watchLists)
    totalWatches += wl.size();
  llvm::errs() << "SOLVE: ok=" << ok << " numClauses=" << numClauses
               << " assumptions=" << assumptionBuf.size()
               << " trail=" << trail.size() << " watches=" << totalWatches
               << " problemClauses=" << problemClauses.size()
               << " binaryClauses=" << binaryClauses.size()
               << " learntClauses=" << learntClauses.size()
               << " learntBinaryClauses=" << learntBinaryClauses.size()
               << "\n";
  llvm::errs() << "  Assumption buffer: [";
  for (int lit : assumptionBuf)
    llvm::errs() << lit << " ";
  llvm::errs() << "]\n";
  if (!ok) {
    llvm::errs() << "SOLVE: Returning UNSAT because ok=false\n";
    return kUNSAT;
  }

  // Initial propagation at decision level 0 (before assumptions).
  // A conflict here means the formula is UNSAT, but this is a PER-SOLVE UNSAT,
  // not a permanent state - don't set ok=false which would break future solves!
  Conflict initialConfl = propagate();
  llvm::errs() << "  After initial propagation: trail.size=" << trail.size() << "\n";
  if (!initialConfl.isNone()) {
    llvm::errs() << "SOLVE: Initial propagation conflict!\n";
    // Note: NOT setting ok=false - that's only for addClause failures
    return kUNSAT;
  }

  // Enqueue assumptions as decisions with propagation after each
  rootLevel = 0;
  for (int extLit : assumptionBuf) {
    int enc = encodeLit(extLit);
    int v = litVar(enc);
    llvm::errs() << "  Assuming: " << extLit << " (enc=" << enc << " var=" << v
                 << " currentAssign=" << static_cast<int>(vars[v].assign)
                 << ")\n";
    newDecisionLevel();
    rootLevel = decisionLevel();
    bool enqueueOk = enqueue(enc, kDecisionReason);
    if (!enqueueOk) {
      llvm::errs() << "SOLVE: Enqueue failed! var=" << v
                   << " assign=" << static_cast<int>(vars[v].assign)
                   << " wantedLit=" << enc << "\n";
    }
    llvm::errs() << "  After enqueue: trail.size=" << trail.size() << "\n";
    Conflict assumptionConfl = propagate();
    llvm::errs() << "  After propagate: trail.size=" << trail.size() << "\n";
    if (!enqueueOk || !assumptionConfl.isNone()) {
      llvm::errs() << "SOLVE: Assumption conflict! enqueueOk=" << enqueueOk
                   << " conflictNone=" << assumptionConfl.isNone() << "\n";
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
    llvm::errs() << "  Search iter=" << iter << " budget=" << budget << "\n";
    result = search(budget);
    llvm::errs() << "  Search iter=" << iter << " result="
                 << (result == kSAT ? "SAT"
                                    : (result == kUNSAT ? "UNSAT" : "UNKNOWN"))
                 << "\n";
    if (confLimit >= 0) {
      confLimit -= budget;
      if (confLimit <= 0 && result == kUNKNOWN)
        break;
    }
  }

  llvm::errs() << "  Before final backtrack: decisionLevel=" << decisionLevel()
               << " trail.size=" << trail.size() << "\n";
  backtrack(0);
  rootLevel = 0;
  llvm::errs() << "  After final backtrack: decisionLevel=" << decisionLevel()
               << " trail.size=" << trail.size() << "\n";
  llvm::errs() << "  Final result: "
               << (result == kSAT ? "SAT"
                                  : (result == kUNSAT ? "UNSAT" : "UNKNOWN"))
               << "\n";
  return result;
}

//===----------------------------------------------------------------------===//
// Bookmark / Rollback (ABC-style incremental solving)
//===----------------------------------------------------------------------===//

void MiniSATSolver::bookmark() {
  assert(decisionLevel() == 0 && "bookmark requires decision level 0");
  assert(propagHead == static_cast<int>(trail.size()) &&
         "bookmark requires completed propagation");

  Bookmark bm;
  bm.numVars = numVars;
  bm.numClauses = numClauses;
  bm.numProblemClauses = static_cast<int>(problemClauses.size());
  bm.numBinaryClauses = static_cast<int>(binaryClauses.size());
  bm.numLearntClauses = static_cast<int>(learntClauses.size());
  bm.numLearntBinaries = static_cast<int>(learntBinaryClauses.size());
  bm.trailSize = static_cast<int>(trail.size());
  bm.propagHead = propagHead;
  bm.varActInc = varActInc;
  bm.claActInc = claActInc;
  bookmark_ = bm;
}

void MiniSATSolver::rollback() {
  assert(bookmark_.has_value() && "rollback requires active bookmark");
  const Bookmark &bm = *bookmark_;

  llvm::errs() << "ROLLBACK: numVars " << numVars << " -> " << bm.numVars
               << "\n";
  llvm::errs() << "  problemClauses " << problemClauses.size() << " -> "
               << bm.numProblemClauses << "\n";
  llvm::errs() << "  binaryClauses " << binaryClauses.size() << " -> "
               << bm.numBinaryClauses << "\n";
  llvm::errs() << "  learntClauses " << learntClauses.size() << " -> "
               << bm.numLearntClauses << "\n";
  llvm::errs() << "  learntBinaryClauses " << learntBinaryClauses.size()
               << " -> " << bm.numLearntBinaries << "\n";

  // 1. Truncate variables to bookmarked count
  numVars = bm.numVars;
  vars.resize(bm.numVars);
  watchLists.resize(2 * bm.numVars);

  // 2. Truncate problem clauses and binary clauses
  problemClauses.resize(bm.numProblemClauses);
  binaryClauses.resize(bm.numBinaryClauses);
  numClauses = bm.numClauses;

  // 3. Remove ALL learned clauses. Learned clauses may be unsound when combined
  // with different assumptions, leading to false UNSAT results. Always clear them
  // on rollback to ensure correctness.
  llvm::errs() << "  Truncating learntClauses from " << learntClauses.size()
               << " to 0\n";
  learntClauses.clear();
  learntBinaryClauses.clear();

  // 4. Clear trail and reset all variable assignments
  //    (preserve activity and polarity for VSIDS warm start)
  trail.clear();
  trailLimits.clear();
  propagHead = 0;
  rootLevel = 0;
  for (int i = 0; i < bm.numVars; i++) {
    vars[i].assign = kUndef;
    vars[i].level = 0;
    vars[i].reason = kNoReason;
    vars[i].seen = 0;
  }

  // 5. Rebuild all watch lists from surviving clause storage
  //    (ABC rebuilds by iterating its arena; we iterate our vectors)
  rebuildWatchLists();

  llvm::errs() << "ROLLBACK: After rebuild, total watches: ";
  size_t totalWatches = 0;
  for (const auto &wl : watchLists)
    totalWatches += wl.size();
  llvm::errs() << totalWatches << "\n";

  // 6. Rebuild VSIDS heap immediately (must rebuild before any backtrack()
  //    calls during solve, otherwise heap state will be inconsistent)
  vsidsHeap.clear();
  vsidsHeap.grow(bm.numVars);
  auto sf = [this](int x) { return vars[x].activity; };
  for (int v = 0; v < bm.numVars; v++)
    vsidsHeap.insert(v, sf);
  heapNeedsRebuild = false;

  // 7. Restore activity decay state
  varActInc = bm.varActInc;
  claActInc = bm.claActInc;

  // 8. Reset ok flag (UNSAT-causing clauses may have been removed)
  ok = true;

  // 9. Clear input buffers
  clauseBuf.clear();
  assumptionBuf.clear();

  // Bookmark remains active for repeated rollbacks
}

void MiniSATSolver::rebuildWatchLists() {
  for (auto &wl : watchLists)
    wl.clear();

  // Problem binary clauses
  for (auto &[lit0, lit1] : binaryClauses)
    addBinaryWatch(lit0, lit1);

  // Learned binary clauses
  for (auto &[lit0, lit1] : learntBinaryClauses)
    addBinaryWatch(lit0, lit1);

  // Problem clauses (size >= 3)
  for (uint32_t i = 0; i < problemClauses.size(); i++) {
    auto &c = problemClauses[i];
    if (c.size >= 2)
      addWatchPair(c.lits[0], c.lits[1], i);
  }

  // Learned clauses (size >= 3, skip deleted)
  for (uint32_t i = 0; i < learntClauses.size(); i++) {
    auto &c = learntClauses[i];
    if (c.size >= 2)
      addWatchPair(c.lits[0], c.lits[1], i | kLearntTag);
  }
}
