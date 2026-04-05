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
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <array>
#include <optional>
#include <utility>

using namespace circt;
using namespace circt::synth;

namespace {

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
    return static_cast<unsigned>(
        llvm::hash_combine(key.opName, llvm::hash_combine_range(
                                           key.operandSignals.begin(),
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

struct LocalFrontierLeaf {
  Value value;
  unsigned depth = 0;
};

static NPNClass computeNPNClassFromTruthTable(const BinaryTruthTable &truthTable,
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

static FailureOr<llvm::APInt>
evaluateLocalCutValue(Value value, unsigned numInputs,
                      DenseMap<Value, unsigned> &leafToIndex,
                      DenseMap<Value, llvm::APInt> &cache) {
  if (auto it = cache.find(value); it != cache.end())
    return it->second;
  if (auto it = leafToIndex.find(value); it != leafToIndex.end()) {
    auto tt = circt::createVarMask(numInputs, it->second, true);
    cache[value] = tt;
    return tt;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return blockArg.getOwner()->getParentOp()->emitError()
           << "local cut input missing from leaf set";

  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return emitError(value.getLoc(), "unsupported value in local cut");

  auto memoize = [&](llvm::APInt result) -> FailureOr<llvm::APInt> {
    cache[value] = result;
    return result;
  };

  if (auto constant = dyn_cast<hw::ConstantOp>(defOp)) {
    if (!constant.getType().isInteger(1))
      return constant.emitError("local cut only supports i1 constants");
    return memoize(constant.getValue().isOne()
                       ? llvm::APInt::getAllOnes(1u << numInputs)
                       : llvm::APInt(1u << numInputs, 0));
  }
  if (auto wireOp = dyn_cast<hw::WireOp>(defOp))
    return memoize(*evaluateLocalCutValue(wireOp.getInput(), numInputs,
                                          leafToIndex, cache));
  if (auto choiceOp = dyn_cast<synth::ChoiceOp>(defOp))
    return memoize(*evaluateLocalCutValue(choiceOp.getInputs().front(),
                                          numInputs, leafToIndex, cache));
  if (auto andOp = dyn_cast<aig::AndInverterOp>(defOp)) {
    SmallVector<llvm::APInt, 3> inputs;
    inputs.reserve(andOp.getInputs().size());
    for (Value operand : andOp.getInputs())
      inputs.push_back(
          *evaluateLocalCutValue(operand, numInputs, leafToIndex, cache));
    return memoize(andOp.evaluate(inputs));
  }
  if (auto xorOp = dyn_cast<comb::XorOp>(defOp)) {
    if (xorOp.getNumOperands() == 0)
      return emitError(xorOp.getLoc(), "xor must have at least one operand");
    llvm::APInt result(1u << numInputs, 0);
    for (auto [index, operand] : llvm::enumerate(xorOp.getOperands())) {
      auto input = *evaluateLocalCutValue(operand, numInputs, leafToIndex, cache);
      result = index == 0 ? input : result ^ input;
    }
    return memoize(result);
  }
  if (auto majOp = dyn_cast<mig::MajorityInverterOp>(defOp)) {
    SmallVector<llvm::APInt, 3> inputs;
    inputs.reserve(majOp.getInputs().size());
    for (Value operand : majOp.getInputs())
      inputs.push_back(
          *evaluateLocalCutValue(operand, numInputs, leafToIndex, cache));
    return memoize(majOp.evaluate(inputs));
  }
  if (auto dotOp = dyn_cast<dig::DotInverterOp>(defOp)) {
    SmallVector<llvm::APInt, 3> inputs;
    inputs.reserve(dotOp.getInputs().size());
    for (Value operand : dotOp.getInputs())
      inputs.push_back(
          *evaluateLocalCutValue(operand, numInputs, leafToIndex, cache));
    return memoize(dotOp.evaluate(inputs));
  }

  return defOp->emitError("unsupported operation in local cut truth-table evaluation");
}

static MatchBinding getBindingForPattern(const LocalCut &cut,
                                         const NPNClass *patternNPN,
                                         const CutRewriterOptions &options) {
  MatchBinding binding = getIdentityBinding(cut.getInputSize());
  if (!patternNPN)
    return binding;

  const auto &cutNPN = cut.getNPNClass(options);
  cutNPN.getInputPermutation(*patternNPN, binding.inputPermutation);
  binding.inputNegationMask = static_cast<uint8_t>(
      cutNPN.inputNegation ^ patternNPN->inputNegation);
  binding.outputNegated =
      static_cast<bool>((cutNPN.outputNegation ^ patternNPN->outputNegation) &
                        1u);
  return binding;
}

static void buildGreedyValueOrder(Block *block,
                                  DenseMap<Value, unsigned> &valueOrder) {
  valueOrder.clear();
  unsigned nextOrder = 0;
  for (BlockArgument argument : block->getArguments())
    valueOrder.try_emplace(argument, nextOrder++);
  for (Operation &op : *block)
    for (Value result : op.getResults())
      valueOrder.try_emplace(result, nextOrder++);
}

static unsigned getGreedyValueOrder(const DenseMap<Value, unsigned> &valueOrder,
                                    Value value) {
  auto it = valueOrder.find(value);
  assert(it != valueOrder.end() && "missing greedy value order");
  return it->second;
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

static void dedupNormalizedLocalFrontier(
    SmallVectorImpl<LocalFrontierLeaf> &frontier) {
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

static void normalizeAndDedupLocalFrontier(
    SmallVectorImpl<LocalFrontierLeaf> &frontier,
    const DenseMap<Value, unsigned> &valueOrder) {
  llvm::sort(frontier, [&](const LocalFrontierLeaf &lhs,
                           const LocalFrontierLeaf &rhs) {
    return getGreedyValueOrder(valueOrder, lhs.value) <
           getGreedyValueOrder(valueOrder, rhs.value);
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

static void enumerateGreedyCutsForRoot(
    Value root, unsigned maxInputs, unsigned maxExpansionDepth,
    const DenseMap<Value, unsigned> &valueOrder,
    SmallVectorImpl<LocalCut> &cuts) {
  cuts.clear();
  if (!isGreedyExpandableValue(root))
    return;

  SmallVector<LocalFrontierLeaf, 8> initialFrontier;
  appendExpandedGreedyLeaf(root, 1, initialFrontier);
  normalizeAndDedupLocalFrontier(initialFrontier, valueOrder);
  if (initialFrontier.size() > maxInputs)
    return;

  auto addCut = [&](ArrayRef<LocalFrontierLeaf> frontier) {
    LocalCut cut;
    cut.root = root;
    cut.leaves.reserve(frontier.size());
    for (const auto &leaf : frontier)
      cut.leaves.push_back(leaf.value);
    if (!cut.leaves.empty() && !hasEquivalentGreedyCut(cuts, cut))
      cuts.push_back(std::move(cut));
  };

  auto recurse = [&](auto &&self,
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
      normalizeAndDedupLocalFrontier(expanded, valueOrder);
      if (expanded.empty() || expanded.size() > maxInputs)
        continue;
      self(self, expanded);
    }
  };

  recurse(recurse, initialFrontier);
}

static bool isRecipeCommutative(CandidateRecipeNode::Kind kind) {
  return kind == CandidateRecipeNode::And || kind == CandidateRecipeNode::Xor ||
         kind == CandidateRecipeNode::Maj;
}

static bool isRecipeAreaCounted(CandidateRecipeNode::Kind kind) {
  return kind == CandidateRecipeNode::Identity ||
         kind == CandidateRecipeNode::And || kind == CandidateRecipeNode::Xor ||
         kind == CandidateRecipeNode::Maj || kind == CandidateRecipeNode::Dot;
}

static bool isAreaCountedLogicOp(Operation *op) {
  return isa<aig::AndInverterOp, mig::MajorityInverterOp, dig::DotInverterOp,
             comb::XorOp>(op);
}

static unsigned getUseCount(Value value) {
  return static_cast<unsigned>(std::distance(value.use_begin(), value.use_end()));
}

static OperationName getRecipeOperationName(MLIRContext *context,
                                            const CandidateRecipe &recipe,
                                            CandidateRecipeNode::Kind kind) {
  switch (kind) {
  case CandidateRecipeNode::Identity:
    switch (recipe.inverterKind) {
    case RecipeInverterKind::Aig:
      return OperationName(aig::AndInverterOp::getOperationName(), context);
    case RecipeInverterKind::Mig:
      return OperationName(mig::MajorityInverterOp::getOperationName(),
                           context);
    case RecipeInverterKind::Dig:
      return OperationName(dig::DotInverterOp::getOperationName(), context);
    }
    break;
  case CandidateRecipeNode::And:
    return OperationName(aig::AndInverterOp::getOperationName(), context);
  case CandidateRecipeNode::Xor:
    return OperationName(comb::XorOp::getOperationName(), context);
  case CandidateRecipeNode::Maj:
    return OperationName(mig::MajorityInverterOp::getOperationName(), context);
  case CandidateRecipeNode::Dot:
    return OperationName(dig::DotInverterOp::getOperationName(), context);
  case CandidateRecipeNode::Input:
  case CandidateRecipeNode::Const0:
  case CandidateRecipeNode::Const1:
    break;
  }
  llvm_unreachable("recipe node does not have an operation name");
}

static void canonicalizeStructuralOperands(CandidateRecipeNode::Kind kind,
                                           SmallVectorImpl<uint64_t> &signals) {
  if (!isRecipeCommutative(kind))
    return;
  llvm::sort(signals);
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

  Value lookupValue(OperationName opName, ArrayRef<uint64_t> operandSignals) const {
    auto it = valueTable.find(StructuralKey(
        opName, SmallVector<uint64_t, 3>(operandSignals.begin(),
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
    auto key = computeKey(op);
    if (!key)
      return;

    valueTable[*key].push_back(result);
    opToKey.try_emplace(op, std::move(*key));
  }

  void noteErased(Operation *op) {
    if (!op)
      return;
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

private:
  std::optional<StructuralKey> computeKey(Operation *op) {
    SmallVector<uint64_t, 3> operandSignals;
    if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
      operandSignals.reserve(andOp.getNumOperands());
      for (auto [i, operand] : llvm::enumerate(andOp.getInputs()))
        operandSignals.push_back((getSignalId(operand) << 1) |
                                 static_cast<uint64_t>(andOp.isInverted(i)));
      canonicalizeStructuralOperands(
          andOp.getNumOperands() == 1 ? CandidateRecipeNode::Identity
                                      : CandidateRecipeNode::And,
          operandSignals);
      return StructuralKey(op->getName(), std::move(operandSignals));
    }
    if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
      operandSignals.reserve(2);
      for (Value operand : xorOp->getOperands())
        operandSignals.push_back(getSignalId(operand) << 1);
      canonicalizeStructuralOperands(CandidateRecipeNode::Xor, operandSignals);
      return StructuralKey(op->getName(), std::move(operandSignals));
    }
    if (auto majOp = dyn_cast<mig::MajorityInverterOp>(op)) {
      operandSignals.reserve(majOp.getNumOperands());
      for (auto [i, operand] : llvm::enumerate(majOp.getOperands()))
        operandSignals.push_back((getSignalId(operand) << 1) |
                                 static_cast<uint64_t>(majOp.isInverted(i)));
      canonicalizeStructuralOperands(
          majOp.getNumOperands() == 1 ? CandidateRecipeNode::Identity
                                      : CandidateRecipeNode::Maj,
          operandSignals);
      return StructuralKey(op->getName(), std::move(operandSignals));
    }
    if (auto dotOp = dyn_cast<dig::DotInverterOp>(op)) {
      operandSignals.reserve(dotOp.getNumOperands());
      for (auto [i, operand] : llvm::enumerate(dotOp.getOperands()))
        operandSignals.push_back((getSignalId(operand) << 1) |
                                 static_cast<uint64_t>(dotOp.isInverted(i)));
      canonicalizeStructuralOperands(
          dotOp.getNumOperands() == 1 ? CandidateRecipeNode::Identity
                                      : CandidateRecipeNode::Dot,
          operandSignals);
      return StructuralKey(op->getName(), std::move(operandSignals));
    }
    return std::nullopt;
  }

  Block *block = nullptr;
  DenseMap<Value, uint64_t> signalIds;
  DenseMap<StructuralKey, SmallVector<Value, 1>> valueTable;
  DenseMap<Operation *, StructuralKey> opToKey;
  uint64_t nextSignalId = 1;
};

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

static Value materializeRecipeInverter(OpBuilder &builder, Location loc,
                                       RecipeInverterKind kind, Value input) {
  std::array<Value, 1> operands = {input};
  std::array<bool, 1> inverted = {true};
  switch (kind) {
  case RecipeInverterKind::Aig:
    return aig::AndInverterOp::create(builder, loc, operands, inverted);
  case RecipeInverterKind::Mig:
    return mig::MajorityInverterOp::create(builder, loc, operands, inverted);
  case RecipeInverterKind::Dig:
    return dig::DotInverterOp::create(builder, loc, operands, inverted);
  }
  llvm_unreachable("unsupported recipe inverter kind");
}

static ProbeResult probeCandidateRecipe(const CandidateRecipe &recipe,
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

  SmallVector<uint64_t, 8> nodeSignals(recipe.nodes.size(), 0);
  SmallVector<LocalProbeEntry, 8> localEntries;

  auto findLocalSignal = [&](OperationName opName,
                             ArrayRef<uint64_t> operandSignals)
      -> std::optional<uint64_t> {
    for (const auto &entry : localEntries)
      if (entry.opName == opName && entry.operandSignals == operandSignals)
        return entry.signalId;
    return std::nullopt;
  };

  auto bindInputSignal = [&](unsigned inputIndex) -> uint64_t {
    Value inputValue = cutInputs[binding.inputPermutation[inputIndex]];
    uint64_t rawSignal = index.getSignalId(inputValue);
    if (((binding.inputNegationMask >> inputIndex) & 1u) == 0)
      return rawSignal;

    SmallVector<uint64_t, 3> operandSignals = {(rawSignal << 1) | 1u};
    OperationName inverterName =
        getRecipeOperationName(inputValue.getContext(), recipe,
                               CandidateRecipeNode::Identity);
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

  for (unsigned i = 0; i != recipe.nodes.size(); ++i) {
    const auto &node = recipe.nodes[i];
    switch (node.kind) {
    case CandidateRecipeNode::Input:
      assert(i < recipe.numInputs && "input nodes must be first");
      nodeSignals[i] = bindInputSignal(i);
      break;
    case CandidateRecipeNode::Const0:
    case CandidateRecipeNode::Const1:
      nodeSignals[i] = allocateSignal();
      break;
    case CandidateRecipeNode::Identity:
    case CandidateRecipeNode::And:
    case CandidateRecipeNode::Xor:
    case CandidateRecipeNode::Maj:
    case CandidateRecipeNode::Dot: {
      SmallVector<uint64_t, 3> operandSignals;
      operandSignals.reserve(node.fanins.size());
      for (auto [bitIndex, faninNodeIndex] : llvm::enumerate(node.fanins)) {
        uint64_t signal = nodeSignals[faninNodeIndex];
        bool inverted = (node.inputInvertMask >> bitIndex) & 1u;
        operandSignals.push_back((signal << 1) | static_cast<uint64_t>(inverted));
      }
      canonicalizeStructuralOperands(node.kind, operandSignals);
      OperationName opName = getRecipeOperationName(rootValue.getContext(),
                                                    recipe, node.kind);
      if (Value existingValue = index.lookupValue(opName, operandSignals)) {
        if (auto *op = existingValue.getDefiningOp())
          result.reusedOps.push_back(op);
        nodeSignals[i] = index.getSignalId(existingValue);
        break;
      }

      if (auto localSignal = findLocalSignal(opName, operandSignals)) {
        nodeSignals[i] = *localSignal;
        break;
      }

      nodeSignals[i] = allocateSignal();
      if (isRecipeAreaCounted(node.kind))
        ++result.newArea;
      localEntries.push_back(
          LocalProbeEntry{opName, std::move(operandSignals), nodeSignals[i]});
      break;
    }
    }
  }

  result.rootSignal = nodeSignals[recipe.root];
  if (binding.outputNegated) {
    SmallVector<uint64_t, 3> operandSignals = {(result.rootSignal << 1) | 1u};
    OperationName inverterName =
        getRecipeOperationName(rootValue.getContext(), recipe,
                               CandidateRecipeNode::Identity);
    if (Value existingValue = index.lookupValue(inverterName, operandSignals)) {
      if (auto *op = existingValue.getDefiningOp())
        result.reusedOps.push_back(op);
      result.rootSignal = index.getSignalId(existingValue);
    } else {
      result.rootSignal = allocateSignal();
    }
  }

  llvm::sort(result.reusedOps);
  result.reusedOps.erase(
      std::unique(result.reusedOps.begin(), result.reusedOps.end()),
      result.reusedOps.end());
  return result;
}

static ProbeResult probeCandidateRecipe(const CandidateRecipe &recipe,
                                        const MatchBinding &binding,
                                        const LocalCut &cut,
                                        const GreedyStructuralIndex &index) {
  return probeCandidateRecipe(recipe, binding, cut.root, cut.leaves, index);
}

static LocalCut getLocalCut(CutEnumerator &enumerator, const Cut &cut) {
  LocalCut localCut;
  const auto &network = enumerator.getLogicNetwork();
  localCut.root = network.getValue(cut.getRootIndex());
  localCut.leaves.reserve(cut.inputs.size());
  for (uint32_t input : cut.inputs)
    localCut.leaves.push_back(network.getValue(input));
  if (cut.getTruthTable())
    localCut.truthTable = *cut.getTruthTable();
  localCut.npnClass = cut.getNPNClass(enumerator.getOptions());
  return localCut;
}

static SmallVector<Operation *>
computeDoomedCone(Operation *rootOp, ArrayRef<Operation *> preservedOps) {
  SmallPtrSet<Operation *, 8> preserved(preservedOps.begin(), preservedOps.end());
  SmallPtrSet<Operation *, 16> doomedSet;
  DenseMap<Operation *, unsigned> removedUses;
  SmallVector<Operation *> doomed;

  auto recurse = [&](auto &&self, Operation *op) -> void {
    if (!isAreaCountedLogicOp(op) || preserved.contains(op) ||
        doomedSet.contains(op))
      return;
    doomedSet.insert(op);
    doomed.push_back(op);
    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !isAreaCountedLogicOp(defOp) || preserved.contains(defOp))
        continue;
      unsigned removed = ++removedUses[defOp];
      if (removed == getUseCount(defOp->getResult(0)))
        self(self, defOp);
    }
  };

  recurse(recurse, rootOp);
  return doomed;
}

static FailureOr<Operation *> materializeRecipe(OpBuilder &builder,
                                                const LocalCut &cut,
                                                const MatchBinding &binding,
                                                const CandidateRecipe &recipe,
                                                GreedyStructuralIndex &index) {
  auto *rootOp = cut.root.getDefiningOp();
  assert(rootOp && "cut root must be a valid operation");

  Location loc = rootOp->getLoc();
  SmallVector<Value, 8> nodeValues(recipe.nodes.size());
  SmallVector<uint64_t, 8> nodeSignals(recipe.nodes.size(), 0);
  DenseMap<uint64_t, Value> freeInverters;
  Value constZero;
  Value constOne;

  auto getConstant = [&](bool one) -> Value {
    Value &cached = one ? constOne : constZero;
    if (!cached)
      cached = hw::ConstantOp::create(builder, loc, APInt(1, one ? 1 : 0));
    return cached;
  };

  auto bindInputValue = [&](unsigned inputIndex) -> std::pair<Value, uint64_t> {
    Value inputValue = cut.leaves[binding.inputPermutation[inputIndex]];
    uint64_t rawSignal = index.getSignalId(inputValue);
    if (((binding.inputNegationMask >> inputIndex) & 1u) == 0)
      return {inputValue, rawSignal};

    SmallVector<uint64_t, 3> operandSignals = {(rawSignal << 1) | 1u};
    OperationName inverterName =
        getRecipeOperationName(builder.getContext(), recipe,
                               CandidateRecipeNode::Identity);
    if (Value existingValue = index.lookupValue(inverterName, operandSignals))
      return {existingValue, index.getSignalId(existingValue)};

    if (auto it = freeInverters.find(rawSignal); it != freeInverters.end())
      return {it->second, index.getSignalId(it->second)};

    Value inv =
        materializeRecipeInverter(builder, loc, recipe.inverterKind, inputValue);
    freeInverters[rawSignal] = inv;
    uint64_t signal = index.getOrCreateSignalId(inv);
    return {inv, signal};
  };

  for (unsigned i = 0; i != recipe.nodes.size(); ++i) {
    const auto &node = recipe.nodes[i];
    switch (node.kind) {
    case CandidateRecipeNode::Input: {
      auto [value, signal] = bindInputValue(i);
      nodeValues[i] = value;
      nodeSignals[i] = signal;
      break;
    }
    case CandidateRecipeNode::Const0:
      nodeValues[i] = getConstant(false);
      nodeSignals[i] = index.getOrCreateSignalId(nodeValues[i]);
      break;
    case CandidateRecipeNode::Const1:
      nodeValues[i] = getConstant(true);
      nodeSignals[i] = index.getOrCreateSignalId(nodeValues[i]);
      break;
    case CandidateRecipeNode::Identity:
    case CandidateRecipeNode::And:
    case CandidateRecipeNode::Xor:
    case CandidateRecipeNode::Maj:
    case CandidateRecipeNode::Dot: {
      SmallVector<uint64_t, 3> operandSignals;
      SmallVector<Value, 3> operands;
      SmallVector<bool, 3> inverted;
      operandSignals.reserve(node.fanins.size());
      operands.reserve(node.fanins.size());
      inverted.reserve(node.fanins.size());
      for (auto [bitIndex, faninNodeIndex] : llvm::enumerate(node.fanins)) {
        bool isInverted = (node.inputInvertMask >> bitIndex) & 1u;
        operands.push_back(nodeValues[faninNodeIndex]);
        inverted.push_back(isInverted);
        operandSignals.push_back((nodeSignals[faninNodeIndex] << 1) |
                                 static_cast<uint64_t>(isInverted));
      }
      canonicalizeStructuralOperands(node.kind, operandSignals);
      OperationName opName =
          getRecipeOperationName(builder.getContext(), recipe, node.kind);

      if (Value existingValue = index.lookupValue(opName, operandSignals)) {
        nodeValues[i] = existingValue;
        nodeSignals[i] = index.getSignalId(existingValue);
        break;
      }

      Value value;
      switch (node.kind) {
      case CandidateRecipeNode::Identity:
        value = materializeRecipeInverter(builder, loc, recipe.inverterKind,
                                          operands.front());
        break;
      case CandidateRecipeNode::And:
        value = aig::AndInverterOp::create(builder, loc, operands, inverted);
        break;
      case CandidateRecipeNode::Xor:
        assert(operands.size() == 2 && "xor recipe nodes must be binary");
        value = comb::XorOp::create(builder, loc, operands[0], operands[1]);
        break;
      case CandidateRecipeNode::Maj:
        value = mig::MajorityInverterOp::create(builder, loc, operands,
                                                inverted);
        break;
      case CandidateRecipeNode::Dot:
        value = dig::DotInverterOp::create(builder, loc, operands, inverted);
        break;
      case CandidateRecipeNode::Input:
      case CandidateRecipeNode::Const0:
      case CandidateRecipeNode::Const1:
        llvm_unreachable("handled above");
      }

      nodeValues[i] = value;
      nodeSignals[i] = index.getOrCreateSignalId(value);
      break;
    }
    }
  }

  Value result = nodeValues[recipe.root];
  if (binding.outputNegated)
    result = materializeRecipeInverter(builder, loc, recipe.inverterKind,
                                       result);
  if (auto *op = result.getDefiningOp())
    return op;
  return hw::WireOp::create(builder, loc, result).getOperation();
}

static FailureOr<Operation *>
materializeRecipeWithFreshStructuralIndex(OpBuilder &builder, const LocalCut &cut,
                                          const MatchBinding &binding,
                                          const CandidateRecipe &recipe) {
  GreedyStructuralIndex index;
  index.initialize(cut.root.getDefiningOp()->getBlock());
  return materializeRecipe(builder, cut, binding, recipe, index);
}

struct GreedyRewriteCandidate {
  LocalCut cut;
  MatchedPattern match;
  CandidateRecipe recipe;
  SmallVector<Operation *, 8> preservedOps;
  int gain = 0;
  unsigned newArea = 0;
};

} // namespace

const NPNClass &LocalCut::getNPNClass(const CutRewriterOptions &options) const {
  if (!truthTable) {
    DenseMap<Value, unsigned> leafToIndex;
    for (auto [index, leaf] : llvm::enumerate(leaves))
      leafToIndex[leaf] = index;

    DenseMap<Value, llvm::APInt> cache;
    auto evaluated =
        evaluateLocalCutValue(root, leaves.size(), leafToIndex, cache);
    assert(succeeded(evaluated) && "failed to evaluate local cut truth table");
    truthTable.emplace(leaves.size(), 1, std::move(*evaluated));
  }

  if (!npnClass)
    npnClass.emplace(computeNPNClassFromTruthTable(*truthTable, options));
  return *npnClass;
}

std::optional<MatchResult>
SpeculativeCutRewritePattern::match(const LocalCut &cut) const {
  auto recipe = speculate(cut);
  if (failed(recipe))
    return std::nullopt;

  MatchResult result;
  result.area = recipe->area;
  result.setOwnedDelays(recipe->perInputDelays);
  return result;
}

std::optional<MatchResult>
SpeculativeCutRewritePattern::match(CutEnumerator &enumerator,
                                    const Cut &cut) const {
  return match(getLocalCut(enumerator, cut));
}

FailureOr<Operation *>
SpeculativeCutRewritePattern::rewrite(OpBuilder &builder,
                                      CutEnumerator &enumerator,
                                      const Cut &cut,
                                      const MatchedPattern &match) const {
  auto localCut = getLocalCut(enumerator, cut);
  auto recipe = speculate(localCut);
  if (failed(recipe))
    return failure();
  return materializeRecipeWithFreshStructuralIndex(builder, localCut,
                                                   match.getBinding(), *recipe);
}

std::optional<MatchedPattern>
GreedyCutRewriter::patternMatchCut(const LocalCut &cut) {
  if (!cut.root || cut.leaves.empty())
    return {};

  const CutRewritePattern *bestPattern = nullptr;
  MatchBinding bestBinding;
  DelayType bestOutputDelay = 0;
  double bestArea = 0.0;
  auto pickBest = [&](const CutRewritePattern *pattern,
                      const MatchResult &matchResult,
                      const MatchBinding &binding) {
    DelayType outputDelay = 0;
    for (DelayType delay : matchResult.getDelays())
      outputDelay = std::max(outputDelay, delay);

    if (!bestPattern || matchResult.area < bestArea ||
        (matchResult.area == bestArea && outputDelay < bestOutputDelay)) {
      bestPattern = pattern;
      bestArea = matchResult.area;
      bestOutputDelay = outputDelay;
      bestBinding = binding;
    }
  };

  if (!patterns.npnToPatternMap.empty()) {
    const auto &cutNPN = cut.getNPNClass(options);
    auto it = patterns.npnToPatternMap.find(
        {cutNPN.truthTable.table, cutNPN.truthTable.numInputs});
    if (it != patterns.npnToPatternMap.end()) {
      for (auto &[patternNPN, pattern] : it->second) {
        auto *specPattern =
            static_cast<const SpeculativeCutRewritePattern *>(pattern);
        auto matchResult = specPattern->match(cut);
        if (!matchResult)
          continue;
        pickBest(pattern, *matchResult,
                 getBindingForPattern(cut, &patternNPN, options));
      }
    }
  }

  for (const CutRewritePattern *pattern : patterns.nonNPNPatterns) {
    auto *specPattern =
        static_cast<const SpeculativeCutRewritePattern *>(pattern);
    if (auto matchResult = specPattern->match(cut))
      pickBest(pattern, *matchResult, getIdentityBinding(cut.getInputSize()));
  }

  if (!bestPattern)
    return {};
  SmallVector<DelayType, 1> arrivalTimes = {bestOutputDelay};
  return MatchedPattern(bestPattern, std::move(arrivalTimes), bestArea,
                        std::move(bestBinding));
}

LogicalResult GreedyCutRewriter::run(Operation *topOp) {
  for (auto &pattern : patterns.patterns) {
    if (pattern->getNumOutputs() > 1)
      return mlir::emitError(pattern->getLoc(),
                             "Greedy cut rewriter does not support patterns "
                             "with multiple outputs yet");
    if (!pattern->isSpeculative())
      return mlir::emitError(pattern->getLoc(),
                             "Greedy cut rewriter requires speculative "
                             "patterns");
  }

  unsigned iteration = 0;
  while (options.maxIterations == 0 || iteration < options.maxIterations) {
    if (failed(topologicallySortLogicNetwork(topOp)))
      return failure();

    Block *block = &topOp->getRegion(0).getBlocks().front();
    GreedyStructuralIndex structuralIndex;
    structuralIndex.initialize(block);
    DenseMap<Value, unsigned> valueOrder;
    buildGreedyValueOrder(block, valueOrder);
    bool changed = false;
    constexpr unsigned greedyMaxExpansionDepth = 3;
    SmallVector<LocalCut, 16> rootCuts;
    SmallVector<Operation *, 32> rootOps;
    for (Operation &op : llvm::reverse(*block)) {
      if (!op.getNumResults() || !op.getResult(0).getType().isInteger(1))
        continue;
      if (!isGreedyExpandableValue(op.getResult(0)))
        continue;
      if (!isAreaCountedLogicOp(&op))
        continue;
      rootOps.push_back(&op);
    }

    SmallVector<Operation *, 32> deferredDoomedOps;
    SmallPtrSet<Operation *, 32> deferredDoomedSet;
    for (Operation *rootOp : rootOps) {
      if ((options.maxIterations != 0 && iteration >= options.maxIterations))
        break;
      if (!rootOp || rootOp->getBlock() != block)
        continue;
      if (!rootOp->getNumResults() || !rootOp->getResult(0).getType().isInteger(1))
        continue;
      Value rootValue = rootOp->getResult(0);
      if (!isGreedyExpandableValue(rootValue) || !isAreaCountedLogicOp(rootOp) ||
          rootValue.use_empty())
        continue;

      std::optional<GreedyRewriteCandidate> bestCandidate;
      enumerateGreedyCutsForRoot(rootValue, options.maxCutInputSize,
                                 greedyMaxExpansionDepth, valueOrder, rootCuts);
      stats.numCutsCreated += rootCuts.size();
      for (LocalCut &cut : rootCuts) {
        auto matched = patternMatchCut(cut);
        if (!matched)
          continue;
        auto *specPattern =
            static_cast<const SpeculativeCutRewritePattern *>(
                matched->getPattern());

        auto recipe = specPattern->speculate(cut);
        if (failed(recipe))
          continue;

        ProbeResult probe =
            probeCandidateRecipe(*recipe, matched->getBinding(), cut,
                                 structuralIndex);
        if (probe.rootSignal == structuralIndex.getSignalId(rootValue))
          continue;

        auto doomed = computeDoomedCone(rootOp, probe.reusedOps);
        int gain =
            static_cast<int>(doomed.size()) - static_cast<int>(probe.newArea);
        if (gain <= 0)
          continue;

        if (!bestCandidate || gain > bestCandidate->gain ||
            (gain == bestCandidate->gain &&
             probe.newArea < bestCandidate->newArea) ||
            (gain == bestCandidate->gain &&
             probe.newArea == bestCandidate->newArea &&
             cut.getInputSize() < bestCandidate->cut.getInputSize())) {
          bestCandidate =
              GreedyRewriteCandidate{cut, *matched, *recipe, probe.reusedOps,
                                     gain, probe.newArea};
        }
      }

      if (!bestCandidate)
        continue;

      auto *rewriteRootOp = bestCandidate->cut.root.getDefiningOp();
      auto doomed = computeDoomedCone(rewriteRootOp, bestCandidate->preservedOps);
      GreedyStructuralIndexListener listener(structuralIndex);
      mlir::IRRewriter rewriter(topOp->getContext(), &listener);
      rewriter.setInsertionPoint(rewriteRootOp);
      auto result =
          materializeRecipe(rewriter, bestCandidate->cut,
                            bestCandidate->match.getBinding(),
                            bestCandidate->recipe, structuralIndex);
      if (failed(result))
        return failure();

      rewriter.replaceOp(rewriteRootOp, *result);
      for (Operation *doomedOp : doomed)
        if (doomedOp != rewriteRootOp &&
            deferredDoomedSet.insert(doomedOp).second) {
          structuralIndex.noteErased(doomedOp);
          deferredDoomedOps.push_back(doomedOp);
        }
      ++stats.numCutsRewritten;
      ++iteration;
      changed = true;
    }

    for (Operation *doomedOp : deferredDoomedOps)
      if (doomedOp->getBlock() && doomedOp->use_empty())
        doomedOp->erase();

    if ((options.maxIterations != 0 && iteration >= options.maxIterations))
      return success();
    if (!changed)
      return success();
  }

  return success();
}
