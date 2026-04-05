//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass exploits the asymmetric care set of DIG nodes by resubstituting
// only the effective `y` input of ternary `synth.dig.dot_inv` roots. For
// `D(x, y, z)`, the `y` input only matters under care set `x & ~z`, so a
// cheaper candidate `y'` is legal whenever
//   x & ~z -> (D(x, y, z) == D(x, y', z))
// is valid.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/SATSolver.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <functional>
#include <random>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_DIGYCARERESUB
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

namespace {

struct CandidateExpr {
  enum class Kind { Value, Constant };

  Kind kind = Kind::Constant;
  Value value;
  bool inverted = false;
  bool constant = false;

  static CandidateExpr getValue(Value value, bool inverted = false) {
    CandidateExpr expr;
    expr.kind = Kind::Value;
    expr.value = value;
    expr.inverted = inverted;
    return expr;
  }

  static CandidateExpr getConstant(bool constant) {
    CandidateExpr expr;
    expr.kind = Kind::Constant;
    expr.constant = constant;
    return expr;
  }

  bool operator==(const CandidateExpr &rhs) const {
    if (kind != rhs.kind)
      return false;
    if (kind == Kind::Constant)
      return constant == rhs.constant;
    return value == rhs.value && inverted == rhs.inverted;
  }
};

static CandidateExpr normalizeExpr(Value value, bool inverted = false) {
  APInt constantValue;
  if (matchPattern(value, mlir::m_ConstantInt(&constantValue)))
    return CandidateExpr::getConstant(constantValue[0] ^ inverted);
  return CandidateExpr::getValue(value, inverted);
}

static CandidateExpr complementExpr(CandidateExpr expr) {
  if (expr.kind == CandidateExpr::Kind::Constant)
    expr.constant = !expr.constant;
  else
    expr.inverted = !expr.inverted;
  return expr;
}

static bool isSupportedSimulatableOp(Operation *op) {
  return isa<hw::WireOp, aig::AndInverterOp, mig::MajorityInverterOp,
             dig::DotInverterOp, comb::AndOp, comb::OrOp, comb::XorOp>(op);
}

class SimulationEngine {
public:
  SimulationEngine(unsigned numPatterns, unsigned seed)
      : numPatterns(numPatterns), numWords(numPatterns / 64), rng(seed) {}

  APInt simulateExpr(CandidateExpr expr) {
    if (expr.kind == CandidateExpr::Kind::Constant)
      return expr.constant ? APInt::getAllOnes(numPatterns)
                           : APInt::getZero(numPatterns);
    APInt value = simulateValue(expr.value);
    return expr.inverted ? ~value : value;
  }

private:
  APInt simulateValue(Value value) {
    if (auto it = cache.find(value); it != cache.end())
      return it->second;

    APInt constantValue;
    if (matchPattern(value, mlir::m_ConstantInt(&constantValue)))
      return cache.try_emplace(value, numPatterns, constantValue[0]).first->second;

    if (isa<BlockArgument>(value))
      return cache.try_emplace(value, makeRandomPattern()).first->second;

    Operation *op = value.getDefiningOp();
    if (!op || !isSupportedSimulatableOp(op))
      return cache.try_emplace(value, makeRandomPattern()).first->second;

    APInt result = TypeSwitch<Operation *, APInt>(op)
                       .Case<hw::WireOp>([&](auto wireOp) {
                         return simulateValue(wireOp.getInput());
                       })
                       .Case<dig::DotInverterOp>([&](auto digOp) {
                         SmallVector<APInt, 3> inputs;
                         inputs.reserve(digOp.getNumOperands());
                         for (Value operand : digOp.getInputs())
                           inputs.push_back(simulateValue(operand));
                         return digOp.evaluate(inputs);
                       })
                       .Case<aig::AndInverterOp>([&](auto andOp) {
                         SmallVector<APInt, 4> inputs;
                         inputs.reserve(andOp.getNumOperands());
                         for (Value operand : andOp.getInputs())
                           inputs.push_back(simulateValue(operand));
                         return andOp.evaluate(inputs);
                       })
                       .Case<mig::MajorityInverterOp>([&](auto majOp) {
                         SmallVector<APInt, 5> inputs;
                         inputs.reserve(majOp.getNumOperands());
                         for (Value operand : majOp.getInputs())
                           inputs.push_back(simulateValue(operand));
                         return majOp.evaluate(inputs);
                       })
                       .Case<comb::AndOp>([&](auto andOp) {
                         APInt result = APInt::getAllOnes(numPatterns);
                         for (Value operand : andOp.getInputs())
                           result &= simulateValue(operand);
                         return result;
                       })
                       .Case<comb::OrOp>([&](auto orOp) {
                         APInt result = APInt::getZero(numPatterns);
                         for (Value operand : orOp.getInputs())
                           result |= simulateValue(operand);
                         return result;
                       })
                       .Case<comb::XorOp>([&](auto xorOp) {
                         APInt result = APInt::getZero(numPatterns);
                         for (Value operand : xorOp.getInputs())
                           result ^= simulateValue(operand);
                         return result;
                       })
                       .Default([&](Operation *) { return makeRandomPattern(); });
    return cache.try_emplace(value, std::move(result)).first->second;
  }

  APInt makeRandomPattern() {
    SmallVector<uint64_t> words(numWords);
    for (uint64_t &word : words)
      word = rng();
    return APInt(numPatterns, words);
  }

  unsigned numPatterns;
  unsigned numWords;
  std::mt19937_64 rng;
  DenseMap<Value, APInt> cache;
};

class DigCareSetSATBuilder {
public:
  DigCareSetSATBuilder(IncrementalSATSolver &solver) : solver(solver) {}

  bool proveEquivalent(dig::DotInverterOp root, CandidateExpr xExpr,
                       CandidateExpr yExpr, CandidateExpr zExpr) {
    encodeValue(root.getResult());
    int oldLit = getOrCreateVar(root.getResult());
    int xLit = getExprLiteral(xExpr);
    int yLit = getExprLiteral(yExpr);
    int zLit = getExprLiteral(zExpr);
    int careLit = buildAnd(xLit, -zLit);
    int newLit = buildDig(xLit, yLit, zLit);
    int diffLit = buildXor(oldLit, newLit);
    int badLit = buildAnd(careLit, diffLit);
    return solver.solve({badLit}) == IncrementalSATSolver::kUNSAT;
  }

private:
  int getOrCreateVar(Value value) {
    auto [it, inserted] = satVars.try_emplace(value, 0);
    if (inserted) {
      it->second = ++nextFreshVar;
      solver.reserveVars(nextFreshVar);
    }
    return it->second;
  }

  int createAuxVar() {
    int freshVar = ++nextFreshVar;
    solver.reserveVars(freshVar);
    return freshVar;
  }

  int getConstLiteral(bool value) {
    int &var = value ? trueVar : falseVar;
    if (var != 0)
      return var;
    var = createAuxVar();
    solver.addClause({value ? var : -var});
    return var;
  }

  int getLiteral(Value value, bool inverted = false) {
    int lit = getOrCreateVar(value);
    return inverted ? -lit : lit;
  }

  int getExprLiteral(CandidateExpr expr) {
    if (expr.kind == CandidateExpr::Kind::Constant)
      return getConstLiteral(expr.constant);
    encodeValue(expr.value);
    return getLiteral(expr.value, expr.inverted);
  }

  void addEquivalenceClauses(int outVar, int inputLit) {
    solver.addClause({-outVar, inputLit});
    solver.addClause({outVar, -inputLit});
  }

  void addAndClauses(int outVar, ArrayRef<int> inputLits) {
    for (int lit : inputLits)
      solver.addClause({-outVar, lit});
    SmallVector<int> clause;
    clause.reserve(inputLits.size() + 1);
    for (int lit : inputLits)
      clause.push_back(-lit);
    clause.push_back(outVar);
    solver.addClause(clause);
  }

  void addOrClauses(int outVar, ArrayRef<int> inputLits) {
    for (int lit : inputLits)
      solver.addClause({-lit, outVar});
    SmallVector<int> clause;
    clause.reserve(inputLits.size() + 1);
    clause.push_back(-outVar);
    clause.append(inputLits.begin(), inputLits.end());
    solver.addClause(clause);
  }

  void addXorClauses(int outVar, int lhsLit, int rhsLit) {
    solver.addClause({-lhsLit, -rhsLit, -outVar});
    solver.addClause({lhsLit, rhsLit, -outVar});
    solver.addClause({lhsLit, -rhsLit, outVar});
    solver.addClause({-lhsLit, rhsLit, outVar});
  }

  void addParityClauses(int outVar, ArrayRef<int> inputLits) {
    assert(!inputLits.empty() && "parity requires at least one input");
    if (inputLits.size() == 1) {
      addEquivalenceClauses(outVar, inputLits.front());
      return;
    }

    int accumulatedLit = inputLits.front();
    for (auto [index, lit] : llvm::enumerate(inputLits.drop_front())) {
      bool isLast = index + 2 == inputLits.size();
      int outLit = isLast ? outVar : createAuxVar();
      addXorClauses(outLit, accumulatedLit, lit);
      accumulatedLit = outLit;
    }
  }

  void addMajorityClauses(int outVar, ArrayRef<int> inputLits) {
    size_t threshold = inputLits.size() / 2 + 1;
    SmallVector<int> subset;
    subset.reserve(threshold);
    std::function<void(size_t, size_t)> visit = [&](size_t start,
                                                    size_t remaining) {
      if (remaining == 0) {
        SmallVector<int> positiveClause;
        positiveClause.reserve(subset.size() + 1);
        for (int lit : subset)
          positiveClause.push_back(-lit);
        positiveClause.push_back(outVar);
        solver.addClause(positiveClause);

        SmallVector<int> negativeClause;
        negativeClause.reserve(subset.size() + 1);
        negativeClause.push_back(-outVar);
        negativeClause.append(subset.begin(), subset.end());
        solver.addClause(negativeClause);
        return;
      }
      for (size_t i = start; i + remaining <= inputLits.size(); ++i) {
        subset.push_back(inputLits[i]);
        visit(i + 1, remaining - 1);
        subset.pop_back();
      }
    };
    visit(0, threshold);
  }

  int buildAnd(int lhsLit, int rhsLit) {
    int outVar = createAuxVar();
    addAndClauses(outVar, {lhsLit, rhsLit});
    return outVar;
  }

  int buildOr(int lhsLit, int rhsLit) {
    int outVar = createAuxVar();
    addOrClauses(outVar, {lhsLit, rhsLit});
    return outVar;
  }

  int buildXor(int lhsLit, int rhsLit) {
    int outVar = createAuxVar();
    addXorClauses(outVar, lhsLit, rhsLit);
    return outVar;
  }

  int buildDig(int xLit, int yLit, int zLit) {
    int xyLit = buildAnd(xLit, yLit);
    int orLit = buildOr(zLit, xyLit);
    return buildXor(xLit, orLit);
  }

  void encodeValue(Value value) {
    SmallVector<std::pair<Value, bool>> worklist;
    worklist.push_back({value, false});

    while (!worklist.empty()) {
      auto [current, readyToEncode] = worklist.pop_back_val();
      if (encodedValues.contains(current))
        continue;

      APInt constantValue;
      if (matchPattern(current, mlir::m_ConstantInt(&constantValue))) {
        encodedValues.insert(current);
        solver.addClause({constantValue[0] ? getOrCreateVar(current)
                                           : -getOrCreateVar(current)});
        continue;
      }

      Operation *op = current.getDefiningOp();
      if (!op) {
        encodedValues.insert(current);
        getOrCreateVar(current);
        continue;
      }

      if (!isSupportedSimulatableOp(op)) {
        encodedValues.insert(current);
        getOrCreateVar(current);
        continue;
      }

      if (!readyToEncode) {
        worklist.push_back({current, true});
        for (Value operand : op->getOperands())
          worklist.push_back({operand, false});
        continue;
      }

      encodedValues.insert(current);
      int outVar = getOrCreateVar(current);
      TypeSwitch<Operation *>(op)
          .Case<hw::WireOp>([&](auto wireOp) {
            addEquivalenceClauses(outVar, getLiteral(wireOp.getInput()));
          })
          .Case<dig::DotInverterOp>([&](auto digOp) {
            if (digOp.getNumOperands() == 1) {
              addEquivalenceClauses(
                  outVar, getLiteral(digOp.getOperand(0), digOp.isInverted(0)));
              return;
            }
            int xLit = getLiteral(digOp.getOperand(0), digOp.isInverted(0));
            int yLit = getLiteral(digOp.getOperand(1), digOp.isInverted(1));
            int zLit = getLiteral(digOp.getOperand(2), digOp.isInverted(2));
            int newLit = buildDig(xLit, yLit, zLit);
            addEquivalenceClauses(outVar, newLit);
          })
          .Case<aig::AndInverterOp>([&](auto andOp) {
            SmallVector<int> inputLits;
            inputLits.reserve(andOp.getNumOperands());
            for (auto [operand, inverted] :
                 llvm::zip(andOp.getInputs(), andOp.getInverted()))
              inputLits.push_back(getLiteral(operand, inverted));
            addAndClauses(outVar, inputLits);
          })
          .Case<mig::MajorityInverterOp>([&](auto majOp) {
            SmallVector<int> inputLits;
            inputLits.reserve(majOp.getNumOperands());
            for (auto [operand, inverted] :
                 llvm::zip(majOp.getInputs(), majOp.getInverted()))
              inputLits.push_back(getLiteral(operand, inverted));
            addMajorityClauses(outVar, inputLits);
          })
          .Case<comb::AndOp>([&](auto andOp) {
            SmallVector<int> inputLits;
            inputLits.reserve(andOp.getNumOperands());
            for (Value operand : andOp.getInputs())
              inputLits.push_back(getLiteral(operand));
            addAndClauses(outVar, inputLits);
          })
          .Case<comb::OrOp>([&](auto orOp) {
            SmallVector<int> inputLits;
            inputLits.reserve(orOp.getNumOperands());
            for (Value operand : orOp.getInputs())
              inputLits.push_back(getLiteral(operand));
            addOrClauses(outVar, inputLits);
          })
          .Case<comb::XorOp>([&](auto xorOp) {
            SmallVector<int> inputLits;
            inputLits.reserve(xorOp.getNumOperands());
            for (Value operand : xorOp.getInputs())
              inputLits.push_back(getLiteral(operand));
            addParityClauses(outVar, inputLits);
          })
          .Default([](Operation *) { llvm_unreachable("unexpected op"); });
    }
  }

  IncrementalSATSolver &solver;
  DenseMap<Value, int> satVars;
  DenseSet<Value> encodedValues;
  int nextFreshVar = 0;
  int trueVar = 0;
  int falseVar = 0;
};

static bool isEligibleRoot(dig::DotInverterOp op) {
  return op.getNumOperands() == 3 && op.getResult().getType().isInteger(1);
}

static void appendCandidate(SmallVectorImpl<CandidateExpr> &candidates,
                            CandidateExpr candidate,
                            CandidateExpr currentYExpr,
                            unsigned maxCandidates) {
  if (candidate == currentYExpr)
    return;
  if (llvm::is_contained(candidates, candidate))
    return;
  if (candidates.size() >= maxCandidates)
    return;
  candidates.push_back(candidate);
}

static SmallVector<CandidateExpr>
buildCandidates(dig::DotInverterOp root, unsigned maxCandidates) {
  SmallVector<CandidateExpr> candidates;
  if (maxCandidates == 0)
    return candidates;

  CandidateExpr xExpr = normalizeExpr(root.getOperand(0), root.isInverted(0));
  CandidateExpr currentYExpr =
      normalizeExpr(root.getOperand(1), root.isInverted(1));
  CandidateExpr zExpr = normalizeExpr(root.getOperand(2), root.isInverted(2));

  appendCandidate(candidates, CandidateExpr::getConstant(false), currentYExpr,
                  maxCandidates);
  appendCandidate(candidates, CandidateExpr::getConstant(true), currentYExpr,
                  maxCandidates);
  appendCandidate(candidates, xExpr, currentYExpr, maxCandidates);
  appendCandidate(candidates, complementExpr(xExpr), currentYExpr,
                  maxCandidates);
  appendCandidate(candidates, zExpr, currentYExpr, maxCandidates);
  appendCandidate(candidates, complementExpr(zExpr), currentYExpr,
                  maxCandidates);

  auto rawYProducer = root.getOperand(1).getDefiningOp<dig::DotInverterOp>();
  if (!rawYProducer)
    return candidates;

  if (rawYProducer.getNumOperands() == 1) {
    CandidateExpr inputExpr =
        normalizeExpr(rawYProducer.getOperand(0), rawYProducer.isInverted(0));
    appendCandidate(candidates, inputExpr, currentYExpr, maxCandidates);
    appendCandidate(candidates, complementExpr(inputExpr), currentYExpr,
                    maxCandidates);
    return candidates;
  }

  for (auto [index, operand] : llvm::enumerate(rawYProducer.getInputs())) {
    CandidateExpr inputExpr =
        normalizeExpr(operand, rawYProducer.isInverted(index));
    appendCandidate(candidates, inputExpr, currentYExpr, maxCandidates);
    appendCandidate(candidates, complementExpr(inputExpr), currentYExpr,
                    maxCandidates);
  }

  return candidates;
}

static bool rejectsBySimulation(dig::DotInverterOp root, CandidateExpr candidate,
                                unsigned numPatterns, unsigned seed) {
  if (numPatterns == 0)
    return false;

  SimulationEngine simulator(numPatterns, seed);
  CandidateExpr xExpr = normalizeExpr(root.getOperand(0), root.isInverted(0));
  CandidateExpr currentYExpr =
      normalizeExpr(root.getOperand(1), root.isInverted(1));
  CandidateExpr zExpr = normalizeExpr(root.getOperand(2), root.isInverted(2));

  APInt x = simulator.simulateExpr(xExpr);
  APInt z = simulator.simulateExpr(zExpr);
  APInt y = simulator.simulateExpr(currentYExpr);
  APInt yPrime = simulator.simulateExpr(candidate);
  APInt care = x & ~z;
  APInt oldValue = x ^ (z | (x & y));
  APInt newValue = x ^ (z | (x & yPrime));
  return (care & (oldValue ^ newValue)).getBoolValue();
}

static bool provesCandidate(dig::DotInverterOp root, CandidateExpr candidate,
                            StringRef satBackend, int conflictLimit) {
  auto solver = createIncrementalSATSolver(satBackend);
  assert(solver && "solver availability must be checked by the pass");
  solver->setConflictLimit(conflictLimit);

  CandidateExpr xExpr = normalizeExpr(root.getOperand(0), root.isInverted(0));
  CandidateExpr zExpr = normalizeExpr(root.getOperand(2), root.isInverted(2));

  DigCareSetSATBuilder builder(*solver);
  return builder.proveEquivalent(root, xExpr, candidate, zExpr);
}

static void collectReachableYConeOps(Value value, Value protectedValue,
                                     SmallVectorImpl<Operation *> &reachable,
                                     llvm::SmallPtrSetImpl<Operation *> &seen) {
  if (protectedValue && value == protectedValue)
    return;
  auto digOp = value.getDefiningOp<dig::DotInverterOp>();
  if (!digOp || !seen.insert(digOp).second)
    return;
  reachable.push_back(digOp);
  for (Value operand : digOp.getInputs())
    collectReachableYConeOps(operand, protectedValue, reachable, seen);
}

static SmallVector<Operation *> computeDoomedYCone(dig::DotInverterOp root,
                                                   CandidateExpr candidate) {
  SmallVector<Operation *> reachable;
  llvm::SmallPtrSet<Operation *, 16> seen;
  Value protectedValue =
      candidate.kind == CandidateExpr::Kind::Value ? candidate.value : Value();
  collectReachableYConeOps(root.getOperand(1), protectedValue, reachable, seen);

  llvm::SmallPtrSet<Operation *, 16> doomed;
  doomed.insert(root);
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : reachable) {
      if (doomed.contains(op))
        continue;
      Value result = op->getResult(0);
      bool removable = llvm::all_of(result.getUsers(), [&](Operation *user) {
        return doomed.contains(user);
      });
      if (!removable)
        continue;
      doomed.insert(op);
      changed = true;
    }
  }

  SmallVector<Operation *> doomedOps;
  for (Operation *op : reachable)
    if (doomed.contains(op))
      doomedOps.push_back(op);
  return doomedOps;
}

static Value materializeExpr(OpBuilder &builder, Location loc, CandidateExpr expr,
                             SmallVectorImpl<bool> &inverted) {
  if (expr.kind == CandidateExpr::Kind::Constant) {
    inverted.push_back(false);
    return hw::ConstantOp::create(builder, loc, APInt(1, expr.constant));
  }
  inverted.push_back(expr.inverted);
  return expr.value;
}

static bool rewriteRoot(dig::DotInverterOp root, CandidateExpr candidate,
                        ArrayRef<Operation *> doomedYConeOps) {
  CandidateExpr xExpr = normalizeExpr(root.getOperand(0), root.isInverted(0));
  CandidateExpr zExpr = normalizeExpr(root.getOperand(2), root.isInverted(2));

  OpBuilder builder(root);
  SmallVector<Value, 3> operands;
  SmallVector<bool, 3> inverted;
  operands.push_back(materializeExpr(builder, root.getLoc(), xExpr, inverted));
  operands.push_back(materializeExpr(builder, root.getLoc(), candidate, inverted));
  operands.push_back(materializeExpr(builder, root.getLoc(), zExpr, inverted));

  auto replacement =
      dig::DotInverterOp::create(builder, root.getLoc(), operands, inverted);
  if (auto namehint = root->getAttr("sv.namehint"))
    replacement->setAttr("sv.namehint", namehint);

  root.getResult().replaceAllUsesWith(replacement.getResult());
  root.erase();

  SmallVector<Operation *> eraseOrder;
  Block *block = replacement->getBlock();
  for (Operation &op : llvm::reverse(*block))
    if (llvm::is_contained(doomedYConeOps, &op))
      eraseOrder.push_back(&op);
  for (Operation *op : eraseOrder)
    op->erase();

  return true;
}

struct DIGYCareResubPass
    : public circt::synth::impl::DIGYCareResubBase<DIGYCareResubPass> {
  using DIGYCareResubBase::DIGYCareResubBase;

  void runOnOperation() override {
    auto module = getOperation();

    if (maxCandidatesPerRoot < 0) {
      module.emitError()
          << "'max-candidates-per-root' must be greater than or equal to 0";
      return signalPassFailure();
    }
    if (numRandomPatterns != 0 && (numRandomPatterns & 63U) != 0) {
      module.emitError()
          << "'num-random-patterns' must be 0 or a positive multiple of 64";
      return signalPassFailure();
    }
    if (conflictLimit < -1) {
      module.emitError()
          << "'conflict-limit' must be greater than or equal to -1";
      return signalPassFailure();
    }

    bool hasEligibleRoots = false;
    for (Operation &op : *module.getBodyBlock()) {
      auto digOp = dyn_cast<dig::DotInverterOp>(&op);
      if (digOp && isEligibleRoot(digOp)) {
        hasEligibleRoots = true;
        break;
      }
    }

    if (!hasEligibleRoots) {
      markAllAnalysesPreserved();
      return;
    }

    if (!createIncrementalSATSolver(satSolver)) {
      module.emitError() << "unsupported or unavailable SAT solver '"
                         << satSolver
                         << "' (expected auto, z3, or cadical)";
      return signalPassFailure();
    }

    bool changed = false;
    unsigned rootSeed = 0;
    for (Operation &op : llvm::make_early_inc_range(*module.getBodyBlock())) {
      auto root = dyn_cast<dig::DotInverterOp>(&op);
      if (!root || !isEligibleRoot(root))
        continue;

      ++numRootsVisited;
      SmallVector<CandidateExpr> candidates =
          buildCandidates(root, maxCandidatesPerRoot);
      if (candidates.empty()) {
        ++rootSeed;
        continue;
      }

      for (CandidateExpr candidate : candidates) {
        ++numCandidatesTried;
        if (rejectsBySimulation(root, candidate, numRandomPatterns,
                                0x9e3779b9U + rootSeed)) {
          ++numSimulationRejected;
          continue;
        }
        if (!provesCandidate(root, candidate, satSolver, conflictLimit)) {
          ++numSatRejected;
          continue;
        }

        SmallVector<Operation *> doomedYConeOps =
            computeDoomedYCone(root, candidate);
        unsigned oldArea = 1 + doomedYConeOps.size();
        unsigned newArea = 1;
        if (newArea >= oldArea)
          continue;

        rewriteRoot(root, candidate, doomedYConeOps);
        ++numRewrites;
        changed = true;
        break;
      }

      ++rootSeed;
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
