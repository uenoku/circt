//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the value-based greedy speculative cut rewriting
// helpers and driver.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/CutRewriter.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <array>
#include <optional>
#include <utility>

#define DEBUG_TYPE "synth-greedy-cut-rewrite"

using namespace circt;
using namespace circt::synth;

namespace {

constexpr unsigned greedyMaxExpansionDepth = 4;

struct StructuralKey {
  OperationName opName;
  SmallVector<uint64_t, 3> operandSignals;

  StructuralKey() = delete;
  StructuralKey(OperationName opName, SmallVector<uint64_t, 3> operandSignals)
      : opName(opName), operandSignals(std::move(operandSignals)) {}

  bool operator==(const StructuralKey &other) const {
    return opName == other.opName && operandSignals == other.operandSignals;
  }
};

} // namespace

template <>
struct llvm::DenseMapInfo<StructuralKey> {
  static StructuralKey getEmptyKey() {
    return StructuralKey(llvm::DenseMapInfo<OperationName>::getEmptyKey(), {});
  }

  static StructuralKey getTombstoneKey() {
    return StructuralKey(llvm::DenseMapInfo<OperationName>::getTombstoneKey(),
                         {});
  }

  static unsigned getHashValue(const StructuralKey &key) {
    return static_cast<unsigned>(llvm::hash_combine(
        key.opName, llvm::hash_combine_range(key.operandSignals.begin(),
                                             key.operandSignals.end())));
  }

  static bool isEqual(const StructuralKey &lhs, const StructuralKey &rhs) {
    return lhs == rhs;
  }
};

namespace {

struct ProbeResult {
  unsigned newArea = 0;
  uint64_t rootSignal = 0;
  SmallVector<Operation *, 8> reusedOps;
};

class GreedyStructuralIndex;
static bool isFreeInverterOp(Operation *op);
static uint64_t getGreedyStructuralOrder(const GreedyStructuralIndex &index,
                                         Value value);

struct LocalFrontierLeaf {
  Value value;
  unsigned depth = 0;
};

static NPNClass
computeNPNClassFromTruthTable(const BinaryTruthTable &truthTable,
                              const CutRewriterOptions &options) {
  NPNClass canonicalForm;
  if (!options.npnTable || !options.npnTable->lookup(truthTable, canonicalForm))
    canonicalForm = NPNClass::computeNPNCanonicalForm(truthTable);
  return canonicalForm;
}

static MatchBinding getIdentityBinding(unsigned numInputs) {
  MatchBinding binding;
  binding.inputPermutation.reserve(numInputs);
  for (unsigned i = 0; i != numInputs; ++i)
    binding.inputPermutation.push_back(i);
  return binding;
}

static const BinaryTruthTable &
getLocalCutTruthTable(const LocalCut &cut, const CutRewriterOptions &options) {
  (void)cut.getNPNClass(options);
  assert(cut.truthTable && "local cut truth table should be available");
  return *cut.truthTable;
}

static FailureOr<BinaryTruthTable>
evaluateGreedyPatternTruthTable(const GreedyPatternBlock &pattern,
                                const MatchBinding &binding,
                                unsigned numInputs) {
  if (!pattern.body || !pattern.output)
    return failure();
  DenseMap<Value, llvm::APInt> seedValues;
  for (BlockArgument blockArg : pattern.body->getArguments()) {
    unsigned inputIndex = blockArg.getArgNumber();
    if (inputIndex >= binding.inputPermutation.size())
      return failure();
    llvm::APInt value =
        circt::createVarMask(numInputs, binding.inputPermutation[inputIndex],
                             true);
    if ((binding.inputNegationMask >> inputIndex) & 1u)
      value.flipAllBits();
    seedValues[blockArg] = std::move(value);
  }

  auto truthTable =
      getTruthTable(ValueRange(pattern.output), pattern.body, numInputs, seedValues);
  if (failed(truthTable))
    return failure();
  if (binding.outputNegated)
    truthTable->table.flipAllBits();
  return *truthTable;
}

static unsigned computePatternBlockDepth(const GreedyPatternBlock &pattern) {
  if (!pattern.body || !pattern.output)
    return 0;

  DenseMap<Value, unsigned> depthByValue;
  for (BlockArgument arg : pattern.body->getArguments())
    depthByValue[arg] = 0;

  for (Operation &op : pattern.body->without_terminator()) {
    unsigned depth = 0;
    if (isa<hw::ConstantOp>(op)) {
      depth = 0;
    } else if (auto wireOp = dyn_cast<hw::WireOp>(op)) {
      depth = depthByValue.lookup(wireOp.getInput());
    } else if (isFreeInverterOp(&op)) {
      depth = depthByValue.lookup(op.getOperand(0));
    } else {
      for (Value operand : op.getOperands())
        depth = std::max(depth, depthByValue.lookup(operand));
      depth += 1;
    }
    depthByValue[op.getResult(0)] = depth;
  }

  return depthByValue.lookup(pattern.output);
}

static MatchBinding getBindingForPattern(const LocalCut &cut,
                                         const NPNClass *patternNPN,
                                         const CutRewriterOptions &options) {
  MatchBinding binding = getIdentityBinding(cut.getInputSize());
  if (!patternNPN)
    return binding;

  const auto &cutNPN = cut.getNPNClass(options);
  cutNPN.getInputPermutation(*patternNPN, binding.inputPermutation);
  binding.inputNegationMask =
      static_cast<uint8_t>(cutNPN.inputNegation ^ patternNPN->inputNegation);
  binding.outputNegated = static_cast<bool>(
      (cutNPN.outputNegation ^ patternNPN->outputNegation) & 1u);
  return binding;
}

static bool isGreedyLeafValue(Value value) {
  if (!value)
    return true;
  if (isa<BlockArgument>(value))
    return true;
  if (isa<hw::ConstantOp, synth::ChoiceOp>(value.getDefiningOp()))
    return true;
  return false;
}

static bool isGreedyConstantValue(Value value) {
  return value && isa_and_present<hw::ConstantOp>(value.getDefiningOp());
}

static bool isGreedyExpandableValue(Value value) {
  if (!value || isGreedyLeafValue(value))
    return false;
  auto *op = value.getDefiningOp();
  if (!op || !value.getType().isInteger(1))
    return false;
  if (isa<hw::WireOp>(op))
    return true;
  if (auto andOp = dyn_cast<aig::AndInverterOp>(op))
    return andOp.getNumOperands() == 1 || andOp.getNumOperands() == 2;
  if (auto xorOp = dyn_cast<comb::XorOp>(op))
    return xorOp.getNumOperands() == 2;
  if (auto majOp = dyn_cast<mig::MajorityInverterOp>(op))
    return majOp.getNumOperands() == 1 || majOp.getNumOperands() == 3;
  if (auto dotOp = dyn_cast<dig::DotInverterOp>(op))
    return dotOp.getNumOperands() == 1 || dotOp.getNumOperands() == 3;
  return false;
}

static void getExpandableGreedyOperands(Value value,
                                        SmallVectorImpl<Value> &operands) {
  operands.clear();
  if (!isGreedyExpandableValue(value))
    return;

  auto *op = value.getDefiningOp();
  if (auto wireOp = dyn_cast<hw::WireOp>(op)) {
    operands.push_back(wireOp.getInput());
    return;
  }
  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    llvm::append_range(operands, andOp.getInputs());
    return;
  }
  if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
    llvm::append_range(operands, xorOp.getOperands());
    return;
  }
  if (auto majOp = dyn_cast<mig::MajorityInverterOp>(op)) {
    llvm::append_range(operands, majOp.getOperands());
    return;
  }
  if (auto dotOp = dyn_cast<dig::DotInverterOp>(op)) {
    llvm::append_range(operands, dotOp.getOperands());
    return;
  }
}

static void
dedupNormalizedLocalFrontier(SmallVectorImpl<LocalFrontierLeaf> &frontier) {
  unsigned write = 0;
  for (unsigned read = 0; read != frontier.size(); ++read) {
    if (!frontier[read].value)
      continue;

    if (write != 0 && frontier[write - 1].value == frontier[read].value) {
      frontier[write - 1].depth =
          std::min(frontier[write - 1].depth, frontier[read].depth);
      continue;
    }
    frontier[write++] = frontier[read];
  }
  frontier.resize(write);
}

static void
normalizeAndDedupLocalFrontier(SmallVectorImpl<LocalFrontierLeaf> &frontier,
                               const GreedyStructuralIndex &structuralIndex) {
  llvm::sort(frontier,
             [&](const LocalFrontierLeaf &lhs, const LocalFrontierLeaf &rhs) {
               return getGreedyStructuralOrder(structuralIndex, lhs.value) <
                      getGreedyStructuralOrder(structuralIndex, rhs.value);
             });
  dedupNormalizedLocalFrontier(frontier);
}

static void appendExpandedGreedyLeaf(Value value, unsigned nextDepth,
                                     SmallVectorImpl<LocalFrontierLeaf> &out) {
  SmallVector<Value, 3> operands;
  getExpandableGreedyOperands(value, operands);
  for (Value operand : operands)
    out.push_back({operand, nextDepth});
}

static bool hasEquivalentGreedyCut(ArrayRef<LocalCut> cuts,
                                   const LocalCut &candidate) {
  return llvm::any_of(cuts, [&](const LocalCut &cut) {
    return cut.root == candidate.root && cut.leaves == candidate.leaves;
  });
}

static void
enumerateGreedyCutsForRoot(Value root, unsigned maxInputs,
                           unsigned maxExpansionDepth,
                           const GreedyStructuralIndex &structuralIndex,
                           SmallVectorImpl<LocalCut> &cuts) {
  cuts.clear();
  if (!isGreedyExpandableValue(root))
    return;

  SmallVector<LocalFrontierLeaf, 8> initialFrontier;
  appendExpandedGreedyLeaf(root, 1, initialFrontier);
  normalizeAndDedupLocalFrontier(initialFrontier, structuralIndex);
  if (initialFrontier.size() > maxInputs)
    return;

  auto addCut = [&](ArrayRef<LocalFrontierLeaf> frontier) {
    LocalCut cut;
    cut.root = root;
    cut.leaves.reserve(frontier.size());
    cut.depth = 0;
    for (const auto &leaf : frontier)
      if (!isGreedyConstantValue(leaf.value))
        cut.leaves.push_back(leaf.value);
    for (const auto &leaf : frontier)
      cut.depth = std::max(cut.depth, leaf.depth);
    if (!cut.leaves.empty() && !hasEquivalentGreedyCut(cuts, cut))
      cuts.push_back(std::move(cut));
  };

  auto recurse =
      [&](auto &&self,
          const SmallVector<LocalFrontierLeaf, 8> &frontier) -> void {
    addCut(frontier);

    for (unsigned leafIndex = 0; leafIndex != frontier.size(); ++leafIndex) {
      const auto &leaf = frontier[leafIndex];
      if (leaf.depth >= maxExpansionDepth ||
          !isGreedyExpandableValue(leaf.value))
        continue;

      SmallVector<LocalFrontierLeaf, 8> expanded;
      expanded.reserve(frontier.size() + 2);
      for (unsigned i = 0; i != frontier.size(); ++i) {
        if (i == leafIndex)
          continue;
        expanded.push_back(frontier[i]);
      }
      appendExpandedGreedyLeaf(leaf.value, leaf.depth + 1, expanded);
      normalizeAndDedupLocalFrontier(expanded, structuralIndex);
      if (expanded.empty() || expanded.size() > maxInputs)
        continue;
      self(self, expanded);
    }
  };

  recurse(recurse, initialFrontier);
}

static bool isFreeInverterOp(Operation *op) {
  return (isa<aig::AndInverterOp, mig::MajorityInverterOp, dig::DotInverterOp>(
              op) &&
          op->getNumOperands() == 1);
}

static bool isAreaCountedLogicOp(Operation *op) {
  return isa<comb::XorOp>(op) ||
         ((isa<aig::AndInverterOp, mig::MajorityInverterOp, dig::DotInverterOp>(
              op)) &&
          op->getNumOperands() != 1);
}

static bool isGreedyDoomedTrackableOp(Operation *op) {
  return isAreaCountedLogicOp(op) || isFreeInverterOp(op) ||
         isa<hw::WireOp, synth::ChoiceOp>(op);
}

static unsigned getUseCount(Value value) {
  return static_cast<unsigned>(
      std::distance(value.use_begin(), value.use_end()));
}

static OperationName getGreedyInverterOperationName(MLIRContext *context,
                                                    GreedyPatternInverterKind kind) {
  switch (kind) {
  case GreedyPatternInverterKind::Aig:
    return OperationName(aig::AndInverterOp::getOperationName(), context);
  case GreedyPatternInverterKind::Mig:
    return OperationName(mig::MajorityInverterOp::getOperationName(), context);
  case GreedyPatternInverterKind::Dig:
    return OperationName(dig::DotInverterOp::getOperationName(), context);
  }
  llvm_unreachable("unsupported greedy pattern inverter kind");
}

static bool isStructurallyCommutativeOp(Operation *op) {
  return isa<comb::XorOp>(op) ||
         (isa<aig::AndInverterOp>(op) && op->getNumOperands() == 2) ||
         (isa<mig::MajorityInverterOp>(op) && op->getNumOperands() == 3);
}

static void canonicalizeStructuralOperands(Operation *op,
                                           SmallVectorImpl<uint64_t> &signals) {
  if (!isStructurallyCommutativeOp(op))
    return;
  llvm::sort(signals);
}

static std::optional<StructuralKey>
computeStructuralKey(Operation *op,
                     llvm::function_ref<uint64_t(Value)> getSignalId) {
  SmallVector<uint64_t, 3> operandSignals;
  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    operandSignals.reserve(andOp.getNumOperands());
    for (auto [i, operand] : llvm::enumerate(andOp.getInputs()))
      operandSignals.push_back((getSignalId(operand) << 1) |
                               static_cast<uint64_t>(andOp.isInverted(i)));
    canonicalizeStructuralOperands(op, operandSignals);
    return StructuralKey(op->getName(), std::move(operandSignals));
  }
  if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
    operandSignals.reserve(xorOp.getNumOperands());
    for (Value operand : xorOp.getOperands())
      operandSignals.push_back(getSignalId(operand) << 1);
    canonicalizeStructuralOperands(op, operandSignals);
    return StructuralKey(op->getName(), std::move(operandSignals));
  }
  if (auto majOp = dyn_cast<mig::MajorityInverterOp>(op)) {
    operandSignals.reserve(majOp.getNumOperands());
    for (auto [i, operand] : llvm::enumerate(majOp.getOperands()))
      operandSignals.push_back((getSignalId(operand) << 1) |
                               static_cast<uint64_t>(majOp.isInverted(i)));
    canonicalizeStructuralOperands(op, operandSignals);
    return StructuralKey(op->getName(), std::move(operandSignals));
  }
  if (auto dotOp = dyn_cast<dig::DotInverterOp>(op)) {
    operandSignals.reserve(dotOp.getNumOperands());
    for (auto [i, operand] : llvm::enumerate(dotOp.getOperands()))
      operandSignals.push_back((getSignalId(operand) << 1) |
                               static_cast<uint64_t>(dotOp.isInverted(i)));
    canonicalizeStructuralOperands(op, operandSignals);
    return StructuralKey(op->getName(), std::move(operandSignals));
  }
  return std::nullopt;
}

class GreedyStructuralIndex {
public:
  void initialize(Block *block) {
    this->block = block;
    signalIds.clear();
    valueTable.clear();
    opToKey.clear();
    nextSignalId = 1;
    for (Value arg : block->getArguments())
      getOrCreateSignalId(arg);
    for (Operation &op : *block)
      noteInserted(&op);
  }

  uint64_t getOrCreateSignalId(Value value) {
    auto [it, inserted] = signalIds.try_emplace(value, nextSignalId);
    if (inserted)
      ++nextSignalId;
    return it->second;
  }

  uint64_t getSignalId(Value value) const {
    auto it = signalIds.find(value);
    assert(it != signalIds.end() && "missing structural signal id");
    return it->second;
  }

  uint64_t getNextSignalId() const { return nextSignalId; }

  Value lookupValue(OperationName opName,
                    ArrayRef<uint64_t> operandSignals) const {
    auto it = valueTable.find(
        StructuralKey(opName, SmallVector<uint64_t, 3>(operandSignals.begin(),
                                                       operandSignals.end())));
    if (it == valueTable.end() || it->second.empty())
      return {};
    return it->second.front();
  }

  void noteInserted(Operation *op) {
    if (!op || !op->getNumResults() || !op->getResult(0).getType().isInteger(1))
      return;

    Value result = op->getResult(0);
    getOrCreateSignalId(result);
    if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
      if (constant.getType().isInteger(1))
        constants[constant.getValue().isOne()].push_back(result);
      return;
    }

    auto key = computeStructuralKey(op, [&](Value value) {
      return getSignalId(value);
    });
    if (!key)
      return;

    valueTable[*key].push_back(result);
    opToKey.try_emplace(op, std::move(*key));
  }

  void noteErased(Operation *op) {
    if (!op)
      return;
    if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
      if (constant.getType().isInteger(1))
        llvm::erase(constants[constant.getValue().isOne()], op->getResult(0));
      return;
    }
    auto it = opToKey.find(op);
    if (it == opToKey.end())
      return;

    Value result = op->getResult(0);
    auto tableIt = valueTable.find(it->second);
    if (tableIt != valueTable.end()) {
      llvm::erase(tableIt->second, result);
      if (tableIt->second.empty())
        valueTable.erase(tableIt);
    }
    opToKey.erase(it);
  }

  Value lookupConstant(bool one) const {
    for (Value constant : constants[one])
      if (constant && constant.getDefiningOp() && constant.getDefiningOp()->getBlock())
        return constant;
    return {};
  }

private:
  Block *block = nullptr;
  DenseMap<Value, uint64_t> signalIds;
  DenseMap<StructuralKey, SmallVector<Value, 1>> valueTable;
  DenseMap<Operation *, StructuralKey> opToKey;
  SmallVector<Value, 2> constants[2];
  uint64_t nextSignalId = 1;
};

static uint64_t getGreedyStructuralOrder(const GreedyStructuralIndex &index,
                                         Value value) {
  return index.getSignalId(value);
}

class GreedyStructuralIndexListener : public mlir::RewriterBase::Listener {
public:
  explicit GreedyStructuralIndexListener(GreedyStructuralIndex &index)
      : index(index) {}

  void notifyOperationInserted(Operation *op,
                               mlir::OpBuilder::InsertPoint) override {
    index.noteInserted(op);
  }

  void notifyOperationReplaced(Operation *op, ValueRange replacement) override {
    index.noteErased(op);
  }

  void notifyOperationErased(Operation *op) override { index.noteErased(op); }

private:
  GreedyStructuralIndex &index;
};

static Value materializePatternInverter(OpBuilder &builder, Location loc,
                                        GreedyPatternInverterKind kind,
                                        Value input) {
  std::array<Value, 1> operands = {input};
  std::array<bool, 1> inverted = {true};
  switch (kind) {
  case GreedyPatternInverterKind::Aig:
    return aig::AndInverterOp::create(builder, loc, operands, inverted);
  case GreedyPatternInverterKind::Mig:
    return mig::MajorityInverterOp::create(builder, loc, operands, inverted);
  case GreedyPatternInverterKind::Dig:
    return dig::DotInverterOp::create(builder, loc, operands, inverted);
  }
  llvm_unreachable("unsupported greedy pattern inverter kind");
}

static void collectReachablePatternOps(const GreedyPatternBlock &pattern,
                                       SmallVectorImpl<Operation *> &ops) {
  ops.clear();
  if (!pattern.body || !pattern.output)
    return;

  SmallPtrSet<Operation *, 16> reachable;
  auto visitValue = [&](auto &&self, Value value) -> void {
    auto *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != pattern.body || isa<hw::OutputOp>(defOp))
      return;
    if (!reachable.insert(defOp).second)
      return;
    for (Value operand : defOp->getOperands())
      self(self, operand);
  };
  visitValue(visitValue, pattern.output);

  for (Operation &op : pattern.body->without_terminator())
    if (reachable.contains(&op))
      ops.push_back(&op);
}

static ProbeResult probeGreedyPattern(const GreedyPatternBlock &pattern,
                                      const MatchBinding &binding,
                                      Value rootValue,
                                      ArrayRef<Value> cutInputs,
                                      const GreedyStructuralIndex &index) {
  ProbeResult result;
  struct LocalProbeEntry {
    OperationName opName;
    SmallVector<uint64_t, 3> operandSignals;
    uint64_t signalId = 0;
  };

  uint64_t nextSignalId = index.getNextSignalId();
  auto allocateSignal = [&]() { return nextSignalId++; };

  DenseMap<Value, uint64_t> signalMap;
  SmallVector<LocalProbeEntry, 8> localEntries;
  std::array<uint64_t, 2> localConstantSignals = {0, 0};

  auto findLocalSignal =
      [&](OperationName opName,
          ArrayRef<uint64_t> operandSignals) -> std::optional<uint64_t> {
    for (const auto &entry : localEntries)
      if (entry.opName == opName && entry.operandSignals == operandSignals)
        return entry.signalId;
    return std::nullopt;
  };

  auto getConstantSignal = [&](bool one) -> uint64_t {
    if (Value existing = index.lookupConstant(one))
      return index.getSignalId(existing);
    uint64_t &localSignal = localConstantSignals[one];
    if (!localSignal)
      localSignal = allocateSignal();
    return localSignal;
  };

  auto bindInputSignal = [&](unsigned inputIndex) -> uint64_t {
    Value inputValue = cutInputs[binding.inputPermutation[inputIndex]];
    uint64_t rawSignal = index.getSignalId(inputValue);
    if (((binding.inputNegationMask >> inputIndex) & 1u) == 0)
      return rawSignal;

    SmallVector<uint64_t, 3> operandSignals = {(rawSignal << 1) | 1u};
    OperationName inverterName =
        getGreedyInverterOperationName(inputValue.getContext(),
                                       pattern.inverterKind);
    if (Value existingValue = index.lookupValue(inverterName, operandSignals)) {
      result.reusedOps.push_back(existingValue.getDefiningOp());
      return index.getSignalId(existingValue);
    }

    if (auto localSignal = findLocalSignal(inverterName, operandSignals))
      return *localSignal;

    uint64_t signal = allocateSignal();
    localEntries.push_back(
        LocalProbeEntry{inverterName, std::move(operandSignals), signal});
    return signal;
  };

  for (BlockArgument arg : pattern.body->getArguments())
    signalMap[arg] = bindInputSignal(arg.getArgNumber());

  SmallVector<Operation *, 8> reachableOps;
  collectReachablePatternOps(pattern, reachableOps);
  for (Operation *patternOp : reachableOps) {
    if (auto constant = dyn_cast<hw::ConstantOp>(patternOp)) {
      if (!constant.getType().isInteger(1))
        return ProbeResult();
      signalMap[constant.getResult()] = getConstantSignal(constant.getValue().isOne());
      continue;
    }
    if (auto wireOp = dyn_cast<hw::WireOp>(patternOp)) {
      signalMap[wireOp.getResult()] = signalMap.lookup(wireOp.getInput());
      continue;
    }

    auto key = computeStructuralKey(patternOp, [&](Value value) {
      return signalMap.lookup(value);
    });
    if (!key)
      return ProbeResult();

    if (Value existingValue = index.lookupValue(key->opName, key->operandSignals)) {
      if (auto *op = existingValue.getDefiningOp())
        result.reusedOps.push_back(op);
      signalMap[patternOp->getResult(0)] = index.getSignalId(existingValue);
      continue;
    }

    if (auto localSignal = findLocalSignal(key->opName, key->operandSignals)) {
      signalMap[patternOp->getResult(0)] = *localSignal;
      continue;
    }

    uint64_t signal = allocateSignal();
    signalMap[patternOp->getResult(0)] = signal;
    if (isAreaCountedLogicOp(patternOp))
      ++result.newArea;
    localEntries.push_back(
        LocalProbeEntry{key->opName, std::move(key->operandSignals), signal});
  }

  result.rootSignal = signalMap.lookup(pattern.output);
  if (binding.outputNegated) {
    SmallVector<uint64_t, 3> operandSignals = {(result.rootSignal << 1) | 1u};
    OperationName inverterName =
        getGreedyInverterOperationName(rootValue.getContext(),
                                       pattern.inverterKind);
    if (Value existingValue = index.lookupValue(inverterName, operandSignals)) {
      if (auto *op = existingValue.getDefiningOp())
        result.reusedOps.push_back(op);
      result.rootSignal = index.getSignalId(existingValue);
    } else {
      result.rootSignal = allocateSignal();
      localEntries.push_back(LocalProbeEntry{
          inverterName, std::move(operandSignals), result.rootSignal});
    }
  }

  llvm::sort(result.reusedOps);
  result.reusedOps.erase(
      std::unique(result.reusedOps.begin(), result.reusedOps.end()),
      result.reusedOps.end());
  return result;
}

static ProbeResult probeGreedyPattern(const GreedyPatternBlock &pattern,
                                      const MatchBinding &binding,
                                      const LocalCut &cut,
                                      const GreedyStructuralIndex &index) {
  return probeGreedyPattern(pattern, binding, cut.root, cut.leaves, index);
}

static SmallVector<Operation *>
computeDoomedCone(Operation *rootOp, ArrayRef<Operation *> preservedOps) {
  SmallPtrSet<Operation *, 8> preserved(preservedOps.begin(),
                                        preservedOps.end());
  SmallPtrSet<Operation *, 16> doomedSet;
  DenseMap<Operation *, unsigned> removedUses;
  SmallVector<Operation *> doomed;

  auto recurse = [&](auto &&self, Operation *op) -> void {
    if (!isGreedyDoomedTrackableOp(op) || preserved.contains(op) ||
        doomedSet.contains(op))
      return;
    doomedSet.insert(op);
    doomed.push_back(op);
    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !isGreedyDoomedTrackableOp(defOp) ||
          preserved.contains(defOp))
        continue;
      unsigned removed = ++removedUses[defOp];
      if (removed == getUseCount(defOp->getResult(0)))
        self(self, defOp);
    }
  };

  recurse(recurse, rootOp);
  return doomed;
}

static SmallVector<Operation *>
collectPreservedOps(const LocalCut &cut, ArrayRef<Operation *> reusedOps) {
  SmallVector<Operation *, 8> preservedOps(reusedOps.begin(), reusedOps.end());
  SmallPtrSet<Operation *, 8> preservedSet(preservedOps.begin(),
                                           preservedOps.end());
  for (Value leaf : cut.leaves) {
    auto *defOp = leaf.getDefiningOp();
    if (!defOp || !isGreedyDoomedTrackableOp(defOp) ||
        !preservedSet.insert(defOp).second)
      continue;
    preservedOps.push_back(defOp);
  }
  return preservedOps;
}

static void eraseDoomedCone(mlir::PatternRewriter &rewriter, Operation *rootOp,
                            ArrayRef<Operation *> doomed) {
  SmallVector<Operation *, 16> pending;
  pending.reserve(doomed.size());
  for (Operation *doomedOp : doomed)
    if (doomedOp != rootOp && doomedOp && doomedOp->getBlock())
      pending.push_back(doomedOp);

  bool changed = true;
  while (changed && !pending.empty()) {
    changed = false;
    unsigned write = 0;
    for (Operation *doomedOp : pending) {
      if (!doomedOp || !doomedOp->getBlock())
        continue;
      if (!doomedOp->use_empty()) {
        pending[write++] = doomedOp;
        continue;
      }
      rewriter.eraseOp(doomedOp);
      changed = true;
    }
    pending.resize(write);
  }

  LLVM_DEBUG({
    if (!pending.empty())
      llvm::dbgs() << "failed to erase " << pending.size()
                   << " doomed ops due to remaining uses\n";
  });
}

static FailureOr<Operation *> materializeGreedyPattern(
    OpBuilder &builder, const LocalCut &cut, const MatchBinding &binding,
    const GreedyPatternBlock &pattern, GreedyStructuralIndex &index) {
  auto *rootOp = cut.root.getDefiningOp();
  assert(rootOp && "cut root must be a valid operation");

  Location loc = rootOp->getLoc();
  DenseMap<uint64_t, Value> freeInverters;
  IRMapping mapping;

  auto getConstant = [&](bool one) -> Value {
    if (Value existing = index.lookupConstant(one))
      return existing;
    return hw::ConstantOp::create(builder, loc, APInt(1, one ? 1 : 0));
  };

  auto bindInputValue = [&](unsigned inputIndex) -> std::pair<Value, uint64_t> {
    Value inputValue = cut.leaves[binding.inputPermutation[inputIndex]];
    uint64_t rawSignal = index.getSignalId(inputValue);
    if (((binding.inputNegationMask >> inputIndex) & 1u) == 0)
      return {inputValue, rawSignal};

    SmallVector<uint64_t, 3> operandSignals = {(rawSignal << 1) | 1u};
    OperationName inverterName =
        getGreedyInverterOperationName(builder.getContext(),
                                       pattern.inverterKind);
    if (Value existingValue = index.lookupValue(inverterName, operandSignals))
      return {existingValue, index.getSignalId(existingValue)};

    if (auto it = freeInverters.find(rawSignal); it != freeInverters.end())
      return {it->second, index.getSignalId(it->second)};

    Value inv =
        materializePatternInverter(builder, loc, pattern.inverterKind, inputValue);
    freeInverters[rawSignal] = inv;
    uint64_t signal = index.getOrCreateSignalId(inv);
    return {inv, signal};
  };

  for (BlockArgument arg : pattern.body->getArguments()) {
    auto [value, signal] = bindInputValue(arg.getArgNumber());
    (void)signal;
    mapping.map(arg, value);
  }

  SmallVector<Operation *, 8> reachableOps;
  collectReachablePatternOps(pattern, reachableOps);
  for (Operation *patternOp : reachableOps) {
    if (auto constant = dyn_cast<hw::ConstantOp>(patternOp)) {
      mapping.map(constant.getResult(), getConstant(constant.getValue().isOne()));
      continue;
    }
    if (auto wireOp = dyn_cast<hw::WireOp>(patternOp)) {
      mapping.map(wireOp.getResult(), mapping.lookup(wireOp.getInput()));
      continue;
    }

    auto key = computeStructuralKey(patternOp, [&](Value value) {
      return index.getSignalId(mapping.lookup(value));
    });
    if (!key)
      return failure();

    if (Value existingValue = index.lookupValue(key->opName, key->operandSignals)) {
      mapping.map(patternOp->getResult(0), existingValue);
      continue;
    }

    Operation *cloned = builder.clone(*patternOp, mapping);
    mapping.map(patternOp->getResult(0), cloned->getResult(0));
  }

  Value result = mapping.lookup(pattern.output);
  if (binding.outputNegated)
    result = materializePatternInverter(builder, loc, pattern.inverterKind,
                                        result);
  if (auto *op = result.getDefiningOp())
    return op;
  return hw::WireOp::create(builder, loc, result).getOperation();
}

struct GreedyRewriteCandidate {
  LocalCut cut;
  struct GreedyMatchedPattern {
    const GreedyCutRewritePattern *pattern = nullptr;
    SmallVector<DelayType, 1> arrivalTimes;
    double area = 0.0;
    MatchBinding binding;
    GreedyPatternBlock patternBlock;
    ProbeResult probe;
  } match;
  SmallVector<Operation *, 8> preservedOps;
  int gain = 0;
  unsigned newArea = 0;
  unsigned patternDepth = 0;
};

static std::optional<GreedyRewriteCandidate::GreedyMatchedPattern>
patternMatchCut(const GreedyCutRewritePatternSet &patterns,
                const GreedyCutRewriterOptions &options, const LocalCut &cut,
                const GreedyStructuralIndex &structuralIndex) {
  if (!cut.root || cut.leaves.empty())
    return {};

  const GreedyCutRewritePattern *bestPattern = nullptr;
  MatchBinding bestBinding;
  std::optional<GreedyPatternBlock> bestPatternBlock;
  ProbeResult bestProbe;
  DelayType bestOutputDelay = 0;
  double bestArea = 0.0;
  auto pickBest = [&](const GreedyCutRewritePattern *pattern,
                      const MatchResult &matchResult,
                      const MatchBinding &binding) {
    auto patternBlock = pattern->speculate(cut);
    if (failed(patternBlock) || !patternBlock->body || !patternBlock->output)
      return;

    auto patternTruthTable =
        evaluateGreedyPatternTruthTable(*patternBlock, binding,
                                        cut.getInputSize());
    if (failed(patternTruthTable))
      return;
    if (!(*patternTruthTable == getLocalCutTruthTable(cut, options)))
      return;

    ProbeResult probe = probeGreedyPattern(*patternBlock, binding, cut,
                                           structuralIndex);
    DelayType outputDelay = 0;
    for (DelayType delay : matchResult.getDelays())
      outputDelay = std::max(outputDelay, delay);

    if (!bestPattern || probe.newArea < bestProbe.newArea ||
        (probe.newArea == bestProbe.newArea && matchResult.area < bestArea) ||
        (probe.newArea == bestProbe.newArea && matchResult.area == bestArea &&
         outputDelay < bestOutputDelay)) {
      bestPattern = pattern;
      bestPatternBlock = std::move(*patternBlock);
      bestProbe = std::move(probe);
      bestArea = matchResult.area;
      bestOutputDelay = outputDelay;
      bestBinding = binding;
    }
  };

  const auto &npnToPatternMap = patterns.getNPNToPatternMap();
  if (!npnToPatternMap.empty()) {
    const auto &cutNPN = cut.getNPNClass(options);
    auto it = npnToPatternMap.find(
        {cutNPN.truthTable.table, cutNPN.truthTable.numInputs});
    if (it != npnToPatternMap.end()) {
      for (auto &[patternNPN, pattern] : it->second) {
        auto matchResult = pattern->match(cut);
        if (!matchResult)
          continue;
        pickBest(pattern, *matchResult,
                 getBindingForPattern(cut, &patternNPN, options));
      }
    }
  }

  for (const GreedyCutRewritePattern *pattern : patterns.getNonNPNPatterns()) {
    if (auto matchResult = pattern->match(cut))
      pickBest(pattern, *matchResult, getIdentityBinding(cut.getInputSize()));
  }

  if (!bestPattern)
    return {};
  return GreedyRewriteCandidate::GreedyMatchedPattern{
      bestPattern,
      SmallVector<DelayType, 1>{bestOutputDelay},
      bestArea,
      std::move(bestBinding),
      std::move(*bestPatternBlock),
      std::move(bestProbe)};
}

class GreedyRootPattern : public mlir::RewritePattern {
public:
  GreedyRootPattern(MLIRContext *context, GreedyCutRewriter &driver,
                    GreedyStructuralIndex &structuralIndex)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        driver(driver), structuralIndex(structuralIndex) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->getNumResults() || !op->getResult(0).getType().isInteger(1))
      return failure();
    if (!isGreedyExpandableValue(op->getResult(0)))
      return failure();
    if (!isAreaCountedLogicOp(op))
      return failure();

    Value rootValue = op->getResult(0);
    if (rootValue.use_empty())
      return failure();

    SmallVector<LocalCut, 16> rootCuts;
    enumerateGreedyCutsForRoot(rootValue, driver.getOptions().maxCutInputSize,
                               greedyMaxExpansionDepth, structuralIndex,
                               rootCuts);
    driver.noteCutsCreated(rootCuts.size());
    unsigned currentSupportSize =
        rootCuts.empty() ? op->getNumOperands() : rootCuts.front().getInputSize();

    std::optional<GreedyRewriteCandidate> bestCandidate;
    for (LocalCut &cut : rootCuts) {
      auto matched = patternMatchCut(driver.getPatterns(), driver.getOptions(),
                                     cut, structuralIndex);
      if (!matched)
        continue;
      ProbeResult &probe = matched->probe;
      if (probe.rootSignal == structuralIndex.getSignalId(rootValue))
        continue;

      SmallVector<Operation *, 8> preservedOps =
          collectPreservedOps(cut, probe.reusedOps);
      auto doomed = computeDoomedCone(op, preservedOps);
      int doomedArea = static_cast<int>(llvm::count_if(
          doomed, [](Operation *doomedOp) { return isAreaCountedLogicOp(doomedOp); }));
      int gain = doomedArea - static_cast<int>(probe.newArea);
      unsigned patternDepth = computePatternBlockDepth(matched->patternBlock);
      bool isUsefulZeroGain =
          gain == 0 &&
          (cut.getInputSize() < currentSupportSize || patternDepth < cut.depth);
      if (gain < 0 || (gain == 0 && !isUsefulZeroGain))
        continue;
      LLVM_DEBUG({
        llvm::dbgs() << "candidate cut with gain " << gain
                     << " (new area: " << probe.newArea
                     << ", preserved ops: " << probe.reusedOps.size()
                     << ", doomed ops: " << doomed.size()
                     << ", cut inputs: " << cut.getInputSize()
                     << ", current inputs: " << currentSupportSize
                     << ", pattern depth: " << patternDepth
                     << ", cut depth: " << cut.depth << ")\n";
      });

      if (!bestCandidate || gain > bestCandidate->gain ||
          (gain == bestCandidate->gain &&
           probe.newArea < bestCandidate->newArea) ||
          (gain == bestCandidate->gain &&
           probe.newArea == bestCandidate->newArea &&
           cut.getInputSize() < bestCandidate->cut.getInputSize()) ||
          (gain == bestCandidate->gain &&
           probe.newArea == bestCandidate->newArea &&
           cut.getInputSize() == bestCandidate->cut.getInputSize() &&
           patternDepth < bestCandidate->patternDepth)) {
        bestCandidate = GreedyRewriteCandidate{
            cut, *matched, std::move(preservedOps), gain, probe.newArea,
            patternDepth};
      }
    }

    if (!bestCandidate)
      return failure();

    auto doomed = computeDoomedCone(op, bestCandidate->preservedOps);
    rewriter.setInsertionPoint(op);
    auto result = materializeGreedyPattern(
        rewriter, bestCandidate->cut, bestCandidate->match.binding,
        bestCandidate->match.patternBlock, structuralIndex);
    if (failed(result))
      return failure();

    if (driver.getOptions().attachDebugTiming) {
      auto array = rewriter.getI64ArrayAttr(bestCandidate->match.arrivalTimes);
      (*result)->setAttr("test.arrival_times", array);
    }

    rewriter.replaceOp(op, *result);
    eraseDoomedCone(rewriter, op, doomed);

    driver.noteCutRewritten();
    return success();
  }

private:
  GreedyCutRewriter &driver;
  GreedyStructuralIndex &structuralIndex;
};

} // namespace

const NPNClass &LocalCut::getNPNClass(const CutRewriterOptions &options) const {
  if (!truthTable) {
    DenseMap<Value, llvm::APInt> seedValues;
    for (auto [index, leaf] : llvm::enumerate(leaves))
      seedValues[leaf] = circt::createVarMask(leaves.size(), index, true);

    Block *block = isa<BlockArgument>(root) ? cast<BlockArgument>(root).getOwner()
                                            : root.getDefiningOp()->getBlock();
    auto evaluated =
        getTruthTable(ValueRange(root), block, leaves.size(), seedValues);
    assert(succeeded(evaluated) && "failed to evaluate local cut truth table");
    truthTable.emplace(*evaluated);
  }

  if (!npnClass)
    npnClass.emplace(computeNPNClassFromTruthTable(*truthTable, options));
  return *npnClass;
}

bool GreedyCutRewritePattern::useTruthTableMatcher(
    SmallVectorImpl<NPNClass> &matchingNPNClasses) const {
  return false;
}

static void
dumpRegisteredGreedyCutRewritePattern(const GreedyCutRewritePattern &pattern,
                                      ArrayRef<NPNClass> npnClasses,
                                      bool usesTruthTableMatcher) {
  LLVM_DEBUG({
    llvm::dbgs() << "registered greedy cut-rewrite pattern '"
                 << pattern.getPatternName() << "'"
                 << " outputs=" << pattern.getNumOutputs() << " matcher="
                 << (usesTruthTableMatcher ? "truth-table" : "custom");
    if (usesTruthTableMatcher)
      llvm::dbgs() << " npn-classes=" << npnClasses.size();
    llvm::dbgs() << "\n";

    if (!usesTruthTableMatcher)
      return;

    for (const auto &npnClass : npnClasses) {
      SmallString<32> ttString;
      npnClass.truthTable.table.toStringUnsigned(ttString, 16);
      llvm::dbgs() << "  tt=" << ttString
                   << " inputs=" << npnClass.truthTable.numInputs
                   << " in-neg=0x";
      llvm::dbgs().write_hex(npnClass.inputNegation);
      llvm::dbgs() << " out-neg=0x";
      llvm::dbgs().write_hex(npnClass.outputNegation);
      llvm::dbgs() << " perm=[";
      llvm::interleaveComma(npnClass.inputPermutation, llvm::dbgs());
      llvm::dbgs() << "]\n";
    }
  });
}

GreedyCutRewritePatternSet::GreedyCutRewritePatternSet(
    llvm::SmallVector<std::unique_ptr<GreedyCutRewritePattern>, 4> patterns)
    : patterns(std::move(patterns)) {
  for (auto &pattern : this->patterns) {
    SmallVector<NPNClass, 2> npnClasses;
    bool result = pattern->useTruthTableMatcher(npnClasses);
    dumpRegisteredGreedyCutRewritePattern(*pattern, npnClasses, result);
    if (result) {
      for (auto npnClass : npnClasses) {
        npnToPatternMap[{npnClass.truthTable.table,
                         npnClass.truthTable.numInputs}]
            .push_back(std::make_pair(std::move(npnClass), pattern.get()));
      }
    } else {
      nonNPNPatterns.push_back(pattern.get());
    }
  }
}

LogicalResult GreedyCutRewriter::run(Operation *topOp) {
  for (auto &pattern : patterns.patterns) {
    if (pattern->getNumOutputs() > 1)
      return mlir::emitError(pattern->getLoc(),
                             "Greedy cut rewriter does not support patterns "
                             "with multiple outputs yet");
  }

  auto moduleOp = dyn_cast<hw::HWModuleOp>(topOp);
  if (!moduleOp)
    return topOp->emitError("Greedy cut rewriter expects an hw.module op");

  Block *block = moduleOp.getBodyBlock();
  GreedyStructuralIndex structuralIndex;
  structuralIndex.initialize(block);
  GreedyStructuralIndexListener listener(structuralIndex);

  mlir::RewritePatternSet rewritePatterns(topOp->getContext());
  rewritePatterns.add<GreedyRootPattern>(topOp->getContext(), *this,
                                         structuralIndex);
  mlir::FrozenRewritePatternSet frozenPatterns(std::move(rewritePatterns));

  mlir::GreedyRewriteConfig config;
  config.setListener(&listener);
  config.setScope(&moduleOp->getRegion(0));
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  config.enableFolding(false);
  config.enableConstantCSE(false);
  if (options.maxIterations != 0)
    config.setMaxNumRewrites(options.maxIterations);
  else
    config.setMaxNumRewrites(mlir::GreedyRewriteConfig::kNoLimit);
  config.setMaxIterations(mlir::GreedyRewriteConfig::kNoLimit);

  return mlir::applyPatternsGreedily(topOp, frozenPatterns, config);
}
