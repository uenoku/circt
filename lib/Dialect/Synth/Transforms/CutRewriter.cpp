//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a DAG-based boolean matching cut rewriting algorithm for
// applications like technology/LUT mapping and combinational logic
// optimization. The algorithm uses priority cuts and NPN
// (Negation-Permutation-Negation) canonical forms to efficiently match cuts
// against rewriting patterns.
//
// References:
//  "Combinational and Sequential Mapping with Priority Cuts", Alan Mishchenko,
//  Sungmin Cho, Satrajit Chatterjee and Robert Brayton, ICCAD 2007
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/CutRewriter.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/TruthTable.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#define DEBUG_TYPE "synth-cut-rewriter"

using namespace circt;
using namespace circt::synth;

//===----------------------------------------------------------------------===//
// LogicNetwork
//===----------------------------------------------------------------------===//

uint32_t LogicNetwork::getOrCreateIndex(Value value) {
  auto [it, inserted] = valueToIndex.try_emplace(value, gates.size());
  if (inserted) {
    indexToValue.push_back(value);
    gates.emplace_back(); // Will be filled in later
  }
  return it->second;
}

uint32_t LogicNetwork::getIndex(Value value) const {
  const auto it = valueToIndex.find(value);
  assert(it != valueToIndex.end() &&
         "Value not found in LogicNetwork - use getOrCreateIndex or check with "
         "hasIndex first");
  return it->second;
}

bool LogicNetwork::hasIndex(Value value) const {
  return valueToIndex.contains(value);
}

Value LogicNetwork::getValue(uint32_t index) const {
  // Index 0 and 1 are reserved for constants, they have no associated Value
  if (index == kConstant0 || index == kConstant1)
    return Value();

  assert(index < indexToValue.size() &&
         "Index out of bounds in LogicNetwork::getValue");
  return indexToValue[index];
}

void LogicNetwork::getValues(ArrayRef<uint32_t> indices,
                             SmallVectorImpl<Value> &values) const {
  values.clear();
  values.reserve(indices.size());
  for (uint32_t idx : indices)
    values.push_back(getValue(idx));
}

uint32_t LogicNetwork::addPrimaryInput(Value value) {
  const uint32_t index = getOrCreateIndex(value);
  gates[index] = LogicNetworkGate(nullptr, LogicNetworkGate::PrimaryInput);
  return index;
}

uint32_t LogicNetwork::addGate(Operation *op, LogicNetworkGate::Kind kind,
                               Value result, ArrayRef<Signal> operands) {
  const uint32_t index = getOrCreateIndex(result);
  gates[index] = LogicNetworkGate(op, kind, operands);
  return index;
}

LogicalResult LogicNetwork::buildFromBlock(Block *block) {
  // Pre-size vectors to reduce reallocations (rough estimate)
  const size_t estimatedSize =
      block->getArguments().size() + block->getOperations().size();
  indexToValue.reserve(estimatedSize);
  gates.reserve(estimatedSize);

  auto handleSingleInputGate = [&](Operation *op, Value result,
                                   const Signal &inputSignal) {
    if (!inputSignal.isInverted()) {
      // Non-inverted buffer: directly alias the result to the input
      valueToIndex[result] = inputSignal.getIndex();
      return;
    }
    // Inverted operation: create a NOT gate
    addGate(op, LogicNetworkGate::Identity, result, {inputSignal});
  };

  // Ensure all block arguments are indexed as primary inputs first
  for (Value arg : block->getArguments()) {
    if (!hasIndex(arg))
      addPrimaryInput(arg);
  }

  auto handleOtherResults = [&](Operation *op) {
    for (Value result : op->getResults()) {
      if (result.getType().isInteger(1) && !hasIndex(result))
        addPrimaryInput(result);
    }
  };

  // Process operations in topological order
  for (Operation &op : block->getOperations()) {
    LogicalResult result =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<aig::AndInverterOp>([&](aig::AndInverterOp andOp) {
              const auto inputs = andOp.getInputs();
              if (inputs.size() == 1) {
                // Single-input AND is a buffer or NOT gate
                const Signal inputSignal =
                    getOrCreateSignal(inputs[0], andOp.isInverted(0));
                handleSingleInputGate(andOp, andOp.getResult(), inputSignal);
              } else if (inputs.size() == 2) {
                const Signal lhsSignal =
                    getOrCreateSignal(inputs[0], andOp.isInverted(0));
                const Signal rhsSignal =
                    getOrCreateSignal(inputs[1], andOp.isInverted(1));
                addGate(andOp, LogicNetworkGate::And2, {lhsSignal, rhsSignal});
              } else {
                // Variadic AND gates with >2 inputs are treated as primary
                // inputs.
                handleOtherResults(andOp);
              }
              return success();
            })
            .Case<comb::XorOp>([&](comb::XorOp xorOp) {
              // Handle comb::XorOp
              if (xorOp->getNumOperands() != 2) {
                handleOtherResults(xorOp);
                return success();
              }
              const Signal lhsSignal =
                  getOrCreateSignal(xorOp.getOperand(0), false);
              const Signal rhsSignal =
                  getOrCreateSignal(xorOp.getOperand(1), false);
              addGate(xorOp, LogicNetworkGate::Xor2, {lhsSignal, rhsSignal});
              return success();
            })
            .Case<synth::mig::MajorityInverterOp>(
                [&](synth::mig::MajorityInverterOp majOp) {
                  // Handle synth::mig::MajorityInverterOp
                  if (majOp->getNumOperands() == 1) {
                    // Single input = inverter
                    const Signal inputSignal = getOrCreateSignal(
                        majOp.getOperand(0), majOp.isInverted(0));
                    handleSingleInputGate(majOp, majOp.getResult(),
                                          inputSignal);
                    return success();
                  }
                  if (majOp->getNumOperands() != 3) {
                    // Variadic MAJ is treated as primary inputs.
                    handleOtherResults(majOp);
                    return success();
                  }
                  const Signal aSignal = getOrCreateSignal(majOp.getOperand(0),
                                                           majOp.isInverted(0));
                  const Signal bSignal = getOrCreateSignal(majOp.getOperand(1),
                                                           majOp.isInverted(1));
                  const Signal cSignal = getOrCreateSignal(majOp.getOperand(2),
                                                           majOp.isInverted(2));
                  addGate(majOp, LogicNetworkGate::Maj3,
                          {aSignal, bSignal, cSignal});
                  return success();
                })
            .Case<hw::ConstantOp>([&](hw::ConstantOp constOp) {
              // Handle hw::ConstantOp - map 1-bit constants to
              // kConstant0/kConstant1
              Value result = constOp.getResult();
              if (!result.getType().isInteger(1)) {
                handleOtherResults(constOp);
                return success();
              }
              uint32_t constIdx =
                  constOp.getValue().isZero() ? kConstant0 : kConstant1;
              valueToIndex[result] = constIdx;
              return success();
            })
            .Default([&](Operation *defaultOp) {
              handleOtherResults(defaultOp);
              return success();
            });

    if (failed(result))
      return result;
  }

  return success();
}

void LogicNetwork::clear() {
  valueToIndex.clear();
  indexToValue.clear();
  gates.clear();
  // Re-add the constant nodes (index 0 = const0, index 1 = const1)
  gates.emplace_back(nullptr, LogicNetworkGate::Constant);
  gates.emplace_back(nullptr, LogicNetworkGate::Constant);
  // Placeholders for constants in indexToValue
  indexToValue.push_back(Value()); // const0
  indexToValue.push_back(Value()); // const1
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Return true if the gate at the given index is always a cut input.
// This includes primary inputs and non-simulatable operations.
static bool isAlwaysCutInput(const LogicNetwork &network, uint32_t index) {
  const auto &gate = network.getGate(index);
  return gate.isAlwaysCutInput();
}

// Return true if the new area/delay is better than the old area/delay in the
// context of the given strategy.
static bool compareDelayAndArea(OptimizationStrategy strategy, double newArea,
                                ArrayRef<DelayType> newDelay, double oldArea,
                                ArrayRef<DelayType> oldDelay) {
  if (strategy == OptimizationStrategyArea)
    return newArea < oldArea || (newArea == oldArea && newDelay < oldDelay);
  if (strategy == OptimizationStrategyTiming)
    return newDelay < oldDelay || (newDelay == oldDelay && newArea < oldArea);
  llvm_unreachable("Unknown mapping strategy");
}

LogicalResult circt::synth::topologicallySortLogicNetwork(Operation *topOp) {
  const auto isOperationReady = [](Value value, Operation *op) -> bool {
    // Topologically sort AIG ops and dataflow ops. Other operations can be
    // scheduled.
    return !(isa<aig::AndInverterOp, mig::MajorityInverterOp>(op) ||
             isa<comb::XorOp, comb::AndOp, comb::ExtractOp, comb::ReplicateOp,
                 comb::ConcatOp>(op));
  };

  if (failed(topologicallySortGraphRegionBlocks(topOp, isOperationReady)))
    return emitError(topOp->getLoc(),
                     "failed to sort operations topologically");
  return success();
}

/// Get the truth table for operations within a block.
/// This is used for computing truth tables from MLIR operations (e.g., for
/// technology library patterns), not for LogicNetwork-based cut enumeration.
FailureOr<BinaryTruthTable>
circt::synth::getTruthTable(ValueRange values, Block *block,
                            igraph::InstanceGraph *instanceGraph) {
  // Get the input arguments from the block
  llvm::SmallSetVector<Value, 4> inputArgs;
  for (Value arg : block->getArguments())
    inputArgs.insert(arg);

  // If there are no inputs, return an empty truth table
  if (inputArgs.empty())
    return BinaryTruthTable();

  const int64_t numInputs = inputArgs.size();
  const int64_t numOutputs = values.size();
  if (LLVM_UNLIKELY(numOutputs != 1 || numInputs >= maxTruthTableInputs)) {
    if (numOutputs == 0)
      return BinaryTruthTable(numInputs, 0);
    if (numInputs >= maxTruthTableInputs)
      return mlir::emitError(values.front().getLoc(),
                             "Truth table is too large");
    return mlir::emitError(values.front().getLoc(),
                           "Multiple outputs are not supported yet");
  }

  // Create a map to evaluate the operation
  DenseMap<Value, APInt> eval;
  for (uint32_t i = 0; i < numInputs; ++i)
    eval[inputArgs[i]] = circt::createVarMask(numInputs, i, true);

  // Simulate the operations in the block
  for (Operation &op : *block) {
    if (op.getNumResults() == 0)
      continue;

    // Support AIG, XOR, and MIG operations
    if (auto andOp = dyn_cast<aig::AndInverterOp>(&op)) {
      SmallVector<llvm::APInt, 2> inputs;
      inputs.reserve(andOp.getInputs().size());
      for (auto input : andOp.getInputs()) {
        auto it = eval.find(input);
        if (it == eval.end())
          return andOp.emitError("Input value not found in evaluation map");
        inputs.push_back(it->second);
      }
      eval[andOp.getResult()] = andOp.evaluate(inputs);
    } else if (auto xorOp = dyn_cast<comb::XorOp>(&op)) {
      auto it = eval.find(xorOp.getOperand(0));
      if (it == eval.end())
        return xorOp.emitError("Input value not found in evaluation map");
      llvm::APInt result = it->second;
      for (unsigned i = 1; i < xorOp.getNumOperands(); ++i) {
        it = eval.find(xorOp.getOperand(i));
        if (it == eval.end())
          return xorOp.emitError("Input value not found in evaluation map");
        result ^= it->second;
      }
      eval[xorOp.getResult()] = result;
    } else if (auto migOp = dyn_cast<synth::mig::MajorityInverterOp>(&op)) {
      SmallVector<llvm::APInt, 3> inputs;
      inputs.reserve(migOp.getInputs().size());
      for (auto input : migOp.getInputs()) {
        auto it = eval.find(input);
        if (it == eval.end())
          return migOp.emitError("Input value not found in evaluation map");
        inputs.push_back(it->second);
      }
      eval[migOp.getResult()] = migOp.evaluate(inputs);
    } else if (auto instanceOp = dyn_cast<hw::InstanceOp>(&op)) {
      // Simulate through hw.instance by looking up the referenced module's
      // truth table. This enables supergate modules that compose library cells
      // via instances rather than inlining primitive ops.
      if (!instanceGraph)
        return instanceOp.emitError(
            "hw.instance encountered but no InstanceGraph provided for truth "
            "table simulation");
      auto *moduleNode = instanceGraph->getReferencedModuleImpl(
          cast<igraph::InstanceOpInterface>(instanceOp.getOperation()));
      auto refModule = dyn_cast<hw::HWModuleOp>(moduleNode);
      if (!refModule)
        return instanceOp.emitError(
            "Instance references non-HWModuleOp for truth table simulation");

      // Get the truth table of the referenced module.
      auto *refBody = refModule.getBodyBlock();
      SmallVector<Value> refOutputs;
      for (auto result : refBody->getTerminator()->getOperands())
        refOutputs.push_back(result);
      auto refTT = getTruthTable(refOutputs, refBody);
      if (failed(refTT))
        return instanceOp.emitError(
            "Failed to compute truth table for referenced module");
      if (refTT->numOutputs != 1)
        return instanceOp.emitError(
            "Multi-output instances not supported in truth table simulation");

      unsigned refNumInputs = refTT->numInputs;
      unsigned numRows = 1u << numInputs; // simulation vector width

      // Compose: for each simulation pattern, extract the instance input bits,
      // look up the truth table, and set the output bit.
      APInt result(numRows, 0);
      SmallVector<APInt> instInputs;
      instInputs.reserve(instanceOp.getNumOperands());
      for (auto input : instanceOp.getOperands()) {
        auto it = eval.find(input);
        if (it == eval.end())
          return instanceOp.emitError(
              "Input value not found in evaluation map");
        instInputs.push_back(it->second);
      }

      for (unsigned row = 0; row < numRows; ++row) {
        // Build the index into the referenced module's truth table.
        unsigned ttIndex = 0;
        for (unsigned j = 0; j < refNumInputs; ++j)
          if (instInputs[j][row])
            ttIndex |= (1u << j);
        if (refTT->table[ttIndex])
          result.setBit(row);
      }

      // Map all instance results (only single-output supported).
      eval[instanceOp.getResult(0)] = result;
    } else if (!isa<hw::OutputOp>(&op)) {
      return op.emitError("Unsupported operation for truth table simulation");
    }
  }

  return BinaryTruthTable(numInputs, 1, eval[values[0]]);
}

//===----------------------------------------------------------------------===//
// Cut
//===----------------------------------------------------------------------===//

bool Cut::isTrivialCut() const {
  // A cut is a trivial cut if it has no root (rootIndex == 0 means no root)
  // and only one input
  return rootIndex == 0 && inputs.size() == 1;
}

const NPNClass &Cut::getNPNClass() const {
  // If the NPN is already computed, return it
  if (npnClass)
    return *npnClass;

  const auto &truthTable = *getTruthTable();

  // Compute the NPN canonical form
  auto canonicalForm = NPNClass::computeNPNCanonicalForm(truthTable);

  npnClass.emplace(std::move(canonicalForm));
  return *npnClass;
}

void Cut::getPermutatedInputIndices(
    const NPNClass &patternNPN,
    SmallVectorImpl<unsigned> &permutedIndices) const {
  auto npnClass = getNPNClass();
  npnClass.getInputPermutation(patternNPN, permutedIndices);
}

LogicalResult
Cut::getInputArrivalTimes(CutEnumerator &enumerator,
                          SmallVectorImpl<DelayType> &results) const {
  results.reserve(getInputSize());
  const auto &network = enumerator.getLogicNetwork();

  // Compute arrival times for each input.
  for (auto inputIndex : inputs) {
    if (isAlwaysCutInput(network, inputIndex)) {
      // If the input is a primary input, it has no delay.
      results.push_back(0);
      continue;
    }
    auto *cutSet = enumerator.getCutSet(inputIndex);
    assert(cutSet && "Input must have a valid cut set");

    // If there is no matching pattern, it means it's not possible to use the
    // input in the cut rewriting. Return empty vector to indicate failure.
    auto *bestCut = cutSet->getBestMatchedCut();
    if (!bestCut)
      return failure();

    const auto &matchedPattern = *bestCut->getMatchedPattern();

    // Get the value for result number lookup
    mlir::Value inputValue = network.getValue(inputIndex);
    // Otherwise, the cut input is an op result. Get the arrival time
    // from the matched pattern.
    results.push_back(matchedPattern.getArrivalTime(
        cast<mlir::OpResult>(inputValue).getResultNumber()));
  }

  return success();
}

void Cut::dump(llvm::raw_ostream &os, const LogicNetwork &network) const {
  os << "// === Cut Dump ===\n";
  os << "Cut with " << getInputSize() << " inputs";
  if (rootIndex != 0) {
    auto *rootOp = network.getGate(rootIndex).getOperation();
    if (rootOp)
      os << " and root: " << *rootOp;
  }
  os << "\n";

  if (isTrivialCut()) {
    mlir::Value inputVal = network.getValue(inputs[0]);
    os << "Primary input cut: " << inputVal << "\n";
    return;
  }

  os << "Inputs (indices): \n";
  for (auto [idx, inputIndex] : llvm::enumerate(inputs)) {
    mlir::Value inputVal = network.getValue(inputIndex);
    os << "  Input " << idx << " (index " << inputIndex << "): " << inputVal
       << "\n";
  }

  if (rootIndex != 0) {
    os << "\nRoot operation: \n";
    if (auto *rootOp = network.getGate(rootIndex).getOperation())
      rootOp->print(os);
    os << "\n";
  }

  auto &npnClass = getNPNClass();
  npnClass.dump(os);

  os << "// === Cut End ===\n";
}

unsigned Cut::getInputSize() const { return inputs.size(); }

unsigned Cut::getOutputSize(const LogicNetwork &network) const {
  if (rootIndex == 0)
    return 1; // Trivial cut has 1 output
  auto *rootOp = network.getGate(rootIndex).getOperation();
  return rootOp ? rootOp->getNumResults() : 1;
}

/// Simulate a gate and return its truth table.
static inline llvm::APInt applyGateSemantics(LogicNetworkGate::Kind kind,
                                             const llvm::APInt &a) {
  switch (kind) {
  case LogicNetworkGate::Identity:
    return a;
  default:
    llvm_unreachable("Unsupported unary operation for truth table computation");
  }
}

static inline llvm::APInt applyGateSemantics(LogicNetworkGate::Kind kind,
                                             const llvm::APInt &a,
                                             const llvm::APInt &b) {
  switch (kind) {
  case LogicNetworkGate::And2:
    return a & b;
  case LogicNetworkGate::Xor2:
    return a ^ b;
  default:
    llvm_unreachable(
        "Unsupported binary operation for truth table computation");
  }
}

static inline llvm::APInt applyGateSemantics(LogicNetworkGate::Kind kind,
                                             const llvm::APInt &a,
                                             const llvm::APInt &b,
                                             const llvm::APInt &c) {
  switch (kind) {
  case LogicNetworkGate::Maj3:
    return (a & b) | (a & c) | (b & c);
  default:
    llvm_unreachable(
        "Unsupported ternary operation for truth table computation");
  }
}

/// Simulate a gate and return its truth table.
static llvm::APInt simulateGate(const LogicNetwork &network, uint32_t index,
                                llvm::DenseMap<uint32_t, llvm::APInt> &cache,
                                unsigned numInputs) {
  // Check cache first
  auto cacheIt = cache.find(index);
  if (cacheIt != cache.end())
    return cacheIt->second;

  const auto &gate = network.getGate(index);
  llvm::APInt result;

  auto getEdgeTT = [&](const Signal &edge) {
    auto tt = simulateGate(network, edge.getIndex(), cache, numInputs);
    if (edge.isInverted())
      tt.flipAllBits();
    return tt;
  };

  switch (gate.getKind()) {
  case LogicNetworkGate::Constant:
    // Constant 0 or 1 - return all zeros or all ones
    if (index == LogicNetwork::kConstant0)
      result = llvm::APInt::getZero(1U << numInputs);
    else
      result = llvm::APInt::getAllOnes(1U << numInputs);
    break;

  case LogicNetworkGate::PrimaryInput:
    // Should be in cache already as cut input
    llvm_unreachable("Primary input not in cache - not a cut input?");

  case LogicNetworkGate::And2: {
    result = applyGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]),
                                getEdgeTT(gate.edges[1]));
    break;
  }

  case LogicNetworkGate::Xor2: {
    result = applyGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]),
                                getEdgeTT(gate.edges[1]));
    break;
  }

  case LogicNetworkGate::Maj3: {
    result =
        applyGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]),
                           getEdgeTT(gate.edges[1]), getEdgeTT(gate.edges[2]));
    break;
  }

  case LogicNetworkGate::Identity: {
    result = applyGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]));
    break;
  }
  }

  cache[index] = result;
  return result;
}

void Cut::computeTruthTable(const LogicNetwork &network) {
  if (isTrivialCut()) {
    // For a trivial cut, a truth table is simply the identity function.
    // 0 -> 0, 1 -> 1
    truthTable.emplace(1, 1, llvm::APInt(2, 2));
    return;
  }

  unsigned numInputs = inputs.size();
  if (numInputs >= maxTruthTableInputs) {
    llvm_unreachable("Too many inputs for truth table computation");
  }

  // Initialize cache with input variable masks
  llvm::DenseMap<uint32_t, llvm::APInt> cache;
  for (unsigned i = 0; i < numInputs; ++i) {
    cache[inputs[i]] = circt::createVarMask(numInputs, i, true);
  }

  // Simulate from root
  llvm::APInt result = simulateGate(network, rootIndex, cache, numInputs);

  truthTable.emplace(numInputs, 1, result);
}

// Forward declaration for lazy truth table computation
static BinaryTruthTable
computeTruthTableForGate(const LogicNetworkGate &rootGate,
                         const llvm::SmallVectorImpl<uint32_t> &mergedInputs,
                         const Cut *const *cuts, unsigned numCuts);

void Cut::computeTruthTableFromOperands(const LogicNetwork &network) {
  // Trivial cuts already have their TT computed
  if (isTrivialCut()) {
    computeTruthTable(network);
    return;
  }

  // If no operand cuts stored, fall back to simulation
  if (operandCuts.empty()) {
    computeTruthTable(network);
    return;
  }

  // Use the fast incremental method via computeTruthTableForGate
  const auto &gate = network.getGate(rootIndex);
  BinaryTruthTable tt = computeTruthTableForGate(
      gate, inputs, operandCuts.data(), operandCuts.size());
  truthTable.emplace(std::move(tt));
}

void Cut::computeSignature() {
  signature = 0;
  for (auto index : inputs) {
    // Use a folded 64-bit signature so high node indices also contribute.
    // This keeps signature checks useful in large networks while remaining a
    // conservative prefilter (exact checks still decide correctness).
    signature |= (1ULL << (index & 63));
  }
}

bool Cut::dominates(const Cut &other) const {
  return dominates(other.inputs, other.signature);
}

bool Cut::dominates(ArrayRef<uint32_t> otherInputs, uint64_t otherSig) const {

  if (getInputSize() > otherInputs.size())
    return false;

  if ((signature & otherSig) != signature)
    return false;

  return std::includes(otherInputs.begin(), otherInputs.end(), inputs.begin(),
                       inputs.end());
}

unsigned Cut::estimateMergedSize(const Cut &other) const {
  // Use popcount on the OR of signatures to estimate merged size
  uint64_t mergedSig = signature | other.signature;
  return llvm::popcount(mergedSig);
}

/// Helper to get and expand a truth table for an operand in the merged space.
/// The operand can be:
/// 1. A cut input (returns variable mask)
/// 2. A specific cut's output (returns that cut's truth table, expanded)
static llvm::APInt
getExpandedTruthTable(uint32_t operandIdx, bool isInverted,
                      const llvm::SmallVectorImpl<uint32_t> &mergedInputs,
                      unsigned numMergedInputs, const Cut *const *cuts,
                      unsigned numCuts) {

  auto lookupMergedPos = [&](uint32_t idx) -> std::optional<unsigned> {
    auto *it = llvm::find(mergedInputs, idx);
    if (it == mergedInputs.end())
      return std::nullopt;
    return static_cast<unsigned>(std::distance(mergedInputs.begin(), it));
  };

  // Handle constants directly: they are not cut inputs but may appear as
  // operands of gates inside a cut.
  if (operandIdx == LogicNetwork::kConstant0) {
    auto result = llvm::APInt::getZero(1U << numMergedInputs);
    if (isInverted)
      result.flipAllBits();
    return result;
  }
  if (operandIdx == LogicNetwork::kConstant1) {
    auto result = llvm::APInt::getAllOnes(1U << numMergedInputs);
    if (isInverted)
      result.flipAllBits();
    return result;
  }

  // Check if operand is one of the merged inputs
  if (auto pos = lookupMergedPos(operandIdx)) {
    // It's a cut input - return variable mask (inverted if needed)
    return circt::createVarMask(numMergedInputs, *pos, !isInverted);
  }

  // Otherwise, find which cut produces this operand
  for (unsigned i = 0; i < numCuts; ++i) {
    if (!cuts[i])
      continue;

    uint32_t cutOutput =
        cuts[i]->isTrivialCut() ? cuts[i]->inputs[0] : cuts[i]->getRootIndex();

    if (cutOutput == operandIdx) {
      // Found the cut - expand its truth table to merged space
      const auto &cutTT = *cuts[i]->getTruthTable();

      // Build mapping for this cut's inputs to merged positions
      SmallVector<unsigned, 8> mapping;
      mapping.reserve(cuts[i]->inputs.size());
      unsigned mergedPos = 0;
      for (auto idx : cuts[i]->inputs) {
        // Both vectors are sorted. Advance once and never rewind to keep this
        // mapping build linear in total input count.
        while (mergedPos < mergedInputs.size() && mergedInputs[mergedPos] < idx)
          ++mergedPos;
        assert(mergedPos < mergedInputs.size() &&
               mergedInputs[mergedPos] == idx &&
               "cut input must exist in merged inputs");
        mapping.push_back(mergedPos);
      }

      auto result = circt::detail::expandTruthTableForMergedInputs(
          cutTT.table, mapping, numMergedInputs);
      if (isInverted)
        result.flipAllBits();
      return result;
    }
  }

  llvm_unreachable("Operand not found in cuts or inputs");
}

/// Compute the truth table for a gate based on its operand cuts.
/// This uses incremental computation which is much faster than simulation.
static BinaryTruthTable
computeTruthTableForGate(const LogicNetworkGate &rootGate,
                         const llvm::SmallVectorImpl<uint32_t> &mergedInputs,
                         const Cut *const *cuts, unsigned numCuts) {

  unsigned numMergedInputs = mergedInputs.size();

  auto getEdgeTT = [&](unsigned edgeIdx) {
    const auto &edge = rootGate.edges[edgeIdx];
    return getExpandedTruthTable(edge.getIndex(), edge.isInverted(),
                                 mergedInputs, numMergedInputs, cuts, numCuts);
  };

  BinaryTruthTable result;
  switch (rootGate.getKind()) {
  case LogicNetworkGate::And2:
  case LogicNetworkGate::Xor2:
    result = BinaryTruthTable(
        numMergedInputs, 1,
        applyGateSemantics(rootGate.getKind(), getEdgeTT(0), getEdgeTT(1)));
    break;
  case LogicNetworkGate::Maj3:
    result =
        BinaryTruthTable(numMergedInputs, 1,
                         applyGateSemantics(rootGate.getKind(), getEdgeTT(0),
                                            getEdgeTT(1), getEdgeTT(2)));
    break;
  case LogicNetworkGate::Identity:
    result =
        BinaryTruthTable(numMergedInputs, 1,
                         applyGateSemantics(rootGate.getKind(), getEdgeTT(0)));
    break;
  default:
    llvm_unreachable("Unsupported operation for truth table computation");
  }

  return result;
}

static Cut getAsTrivialCut(uint32_t index, const LogicNetwork &network) {
  // Create a trivial cut for a value
  Cut cut;
  cut.inputs.push_back(index);
  // Compute truth table eagerly for trivial cut
  cut.computeTruthTable(network);
  cut.computeSignature();
  return cut;
}

//===----------------------------------------------------------------------===//
// MatchedPattern
//===----------------------------------------------------------------------===//

ArrayRef<DelayType> MatchedPattern::getArrivalTimes() const {
  assert(pattern && "Pattern must be set to get arrival time");
  return arrivalTimes;
}

DelayType MatchedPattern::getArrivalTime(unsigned index) const {
  assert(pattern && "Pattern must be set to get arrival time");
  return arrivalTimes[index];
}

DelayType MatchedPattern::getWorstOutputArrivalTime() const {
  assert(pattern && "Pattern must be set to get arrival time");
  return *std::max_element(arrivalTimes.begin(), arrivalTimes.end());
}

const CutRewritePattern *MatchedPattern::getPattern() const {
  assert(pattern && "Pattern must be set to get the pattern");
  return pattern;
}

double MatchedPattern::getArea() const {
  assert(pattern && "Pattern must be set to get area");
  return area;
}

//===----------------------------------------------------------------------===//
// CutSet
//===----------------------------------------------------------------------===//

Cut *CutSet::getBestMatchedCut() const { return bestCut; }

unsigned CutSet::size() const { return cuts.size(); }

void CutSet::addCut(Cut *cut) {
  assert(!isFrozen && "Cannot add cuts to a frozen cut set");
  cuts.push_back(cut);
}

ArrayRef<Cut *> CutSet::getCuts() const { return cuts; }

// Remove duplicate cuts and non-minimal cuts. A cut is non-minimal if there
// exists another cut that is a subset of it. We use the signature for efficient
// subset checking since inputs are now indices.
static void removeDuplicateAndNonMinimalCuts(SmallVectorImpl<Cut *> &cuts) {
  auto dumpInputs = [](llvm::raw_ostream &os,
                       const llvm::SmallVectorImpl<uint32_t> &inputs) {
    os << "{";
    llvm::interleaveComma(inputs, os);
    os << "}";
  };
  // Sort by size, then lexicographically by inputs. This enables cheap exact
  // duplicate elimination and tighter candidate filtering for subset checks.
  std::sort(cuts.begin(), cuts.end(), [](const Cut *a, const Cut *b) {
    if (a->getInputSize() != b->getInputSize())
      return a->getInputSize() < b->getInputSize();
    return std::lexicographical_compare(a->inputs.begin(), a->inputs.end(),
                                        b->inputs.begin(), b->inputs.end());
  });

  // Group kept cuts by input size so subset checks only visit smaller cuts.
  unsigned maxCutSize = cuts.empty() ? 0 : cuts.back()->getInputSize();
  llvm::SmallVector<llvm::SmallVector<Cut *, 4>, 16> keptBySize(maxCutSize + 1);

  // Compact kept cuts in-place.
  unsigned uniqueCount = 0;
  for (Cut *cut : cuts) {
    unsigned cutSize = cut->getInputSize();

    // Fast exact duplicate check: with lexicographic sort, duplicates are
    // adjacent among cuts with equal size.
    if (uniqueCount > 0) {
      Cut *lastKept = cuts[uniqueCount - 1];
      if (lastKept->getInputSize() == cutSize &&
          lastKept->inputs == cut->inputs)
        continue;
    }

    bool isDominated = false;
    for (unsigned existingSize = 1; existingSize < cutSize && !isDominated;
         ++existingSize) {
      for (const Cut *existingCut : keptBySize[existingSize]) {
        if (!existingCut->dominates(*cut))
          continue;

        LLVM_DEBUG({
          llvm::dbgs() << "Dropping non-minimal cut ";
          dumpInputs(llvm::dbgs(), cut->inputs);
          llvm::dbgs() << " due to subset ";
          dumpInputs(llvm::dbgs(), existingCut->inputs);
          llvm::dbgs() << "\n";
        });
        isDominated = true;
        break;
      }
    }

    if (isDominated)
      continue;

    cuts[uniqueCount++] = cut;
    keptBySize[cutSize].push_back(cut);
  }

  LLVM_DEBUG(llvm::dbgs() << "Original cuts: " << cuts.size()
                          << " Unique cuts: " << uniqueCount << "\n");

  // Resize the cuts vector to the number of surviving cuts.
  cuts.resize(uniqueCount);
}

void CutSet::finalize(
    const CutRewriterOptions &options,
    llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut,
    const LogicNetwork &logicNetwork) {

  // Remove duplicate/non-minimal cuts first so all follow-up work only runs on
  // survivors.
  removeDuplicateAndNonMinimalCuts(cuts);

  // Compute truth tables lazily, then match cuts to collect timing/area data.
  for (Cut *cut : cuts) {
    if (!cut->getTruthTable().has_value())
      cut->computeTruthTableFromOperands(logicNetwork);

    assert(cut->getInputSize() <= options.maxCutInputSize &&
           "Cut input size exceeds maximum allowed size");

    if (auto matched = matchCut(*cut))
      cut->setMatchedPattern(std::move(*matched));
  }

  // Sort cuts by priority to select the most promising ones.
  // Priority is determined by the optimization strategy:
  // - Trivial cuts (direct connections) have highest priority
  // - Among matched cuts, compare by area/delay based on the strategy
  // - Matched cuts are preferred over unmatched cuts
  // See "Combinational and Sequential Mapping with Priority Cuts" by Mishchenko
  // et al., ICCAD 2007 for more details.
  // TODO: Use a priority queue instead of sorting for better performance.

  // Partition the cuts into trivial and non-trivial cuts.
  auto *trivialCutsEnd =
      std::stable_partition(cuts.begin(), cuts.end(),
                            [](const Cut *cut) { return cut->isTrivialCut(); });

  auto isBetterCut = [&options](const Cut *a, const Cut *b) {
    assert(!a->isTrivialCut() && !b->isTrivialCut() &&
           "Trivial cuts should have been excluded");
    const auto &aMatched = a->getMatchedPattern();
    const auto &bMatched = b->getMatchedPattern();

    if (aMatched && bMatched)
      return compareDelayAndArea(
          options.strategy, aMatched->getArea(), aMatched->getArrivalTimes(),
          bMatched->getArea(), bMatched->getArrivalTimes());

    if (static_cast<bool>(aMatched) != static_cast<bool>(bMatched))
      return static_cast<bool>(aMatched);

    return a->getInputSize() < b->getInputSize();
  };
  std::stable_sort(trivialCutsEnd, cuts.end(), isBetterCut);

  // Keep only the top-K cuts to bound growth.
  if (cuts.size() > options.maxCutSizePerRoot)
    cuts.resize(options.maxCutSizePerRoot);

  // Select the best cut using default delay-oriented strategy
  selectBestCut(options.strategy);

  LLVM_DEBUG({
    llvm::dbgs() << "Finalized cut set with " << cuts.size() << " cuts and "
                 << (bestCut
                         ? "matched pattern to " + bestCut->getMatchedPattern()
                                                       ->getPattern()
                                                       ->getPatternName()
                         : "no matched pattern")
                 << "\n";
  });

  isFrozen = true; // Mark the cut set as frozen
}

void CutSet::selectBestCut(OptimizationStrategy strategy,
                           std::optional<DelayType> requiredTime) {
  bestCut = nullptr;
  for (Cut *cut : cuts) {
    const auto &currentMatch = cut->getMatchedPattern();
    if (!currentMatch)
      continue;

    // If required time is set, skip cuts that violate the timing constraint
    if (requiredTime) {
      if (currentMatch->getWorstOutputArrivalTime() > *requiredTime)
        continue;
    }

    // For delay-first strategy (or no requiredTime), the cuts are already
    // sorted by priority from finalize(), so the first matched cut is best.
    bestCut = cut;
    break;
  }
}

//===----------------------------------------------------------------------===//
// CutRewritePattern
//===----------------------------------------------------------------------===//

bool CutRewritePattern::useTruthTableMatcher(
    SmallVectorImpl<NPNClass> &matchingNPNClasses) const {
  return false;
}

//===----------------------------------------------------------------------===//
// CutRewritePatternSet
//===----------------------------------------------------------------------===//

CutRewritePatternSet::CutRewritePatternSet(
    llvm::SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns)
    : patterns(std::move(patterns)) {
  // Initialize the NPN to pattern map
  for (auto &pattern : this->patterns) {
    SmallVector<NPNClass, 2> npnClasses;
    auto result = pattern->useTruthTableMatcher(npnClasses);
    if (result) {
      for (auto npnClass : npnClasses) {
        // Create a NPN class from the truth table
        npnToPatternMap[{npnClass.truthTable.table,
                         npnClass.truthTable.numInputs}]
            .push_back(std::make_pair(std::move(npnClass), pattern.get()));
      }
    } else {
      // If the pattern does not provide NPN classes, we use a special key
      // to indicate that it should be considered for all cuts.
      nonNPNPatterns.push_back(pattern.get());
    }
  }
}

//===----------------------------------------------------------------------===//
// CutEnumerator
//===----------------------------------------------------------------------===//

CutEnumerator::CutEnumerator(const CutRewriterOptions &options)
    : options(options) {}

CutSet *CutEnumerator::createNewCutSet(uint32_t index) {
  CutSet *cutSet = new (cutSetAllocator.Allocate()) CutSet();
  auto [cutSetPtr, inserted] = cutSets.try_emplace(index, cutSet);
  assert(inserted && "Cut set already exists for this index");
  return cutSetPtr->second;
}

llvm::SmallVector<std::pair<uint32_t, CutSet *>> CutEnumerator::takeCutSets() {
  llvm::SmallVector<std::pair<uint32_t, CutSet *>> result;
  result.reserve(cutSets.size());
  for (auto &[idx, cutSet] : cutSets)
    result.emplace_back(idx, cutSet);
  cutSets.clear();
  return result;
}

void CutEnumerator::clear() {
  cutSets.clear();
  processingOrder.clear();
  logicNetwork.clear();
  cutAllocator.DestroyAll();
  cutSetAllocator.DestroyAll();
}

LogicalResult CutEnumerator::visitLogicOp(uint32_t nodeIndex) {
  const auto &gate = logicNetwork.getGate(nodeIndex);
  auto *logicOp = gate.getOperation();
  assert(logicOp && logicOp->getNumResults() == 1 &&
         "Logic operation must have a single result");

  unsigned numFanins = gate.getNumFanins();

  // Validate operation constraints
  // TODO: Variadic operations and non-single-bit results can be supported
  if (numFanins > 3)
    return logicOp->emitError("Cut enumeration supports at most 3 operands, "
                              "found: ")
           << numFanins;
  if (!logicOp->getOpResult(0).getType().isInteger(1))
    return logicOp->emitError()
           << "Supported logic operations must have a single bit "
              "result type but found: "
           << logicOp->getResult(0).getType();

  SmallVector<const CutSet *, 2> operandCutSets;
  operandCutSets.reserve(numFanins);

  // Collect cut sets for each fanin (using LogicNetwork edges)
  for (unsigned i = 0; i < numFanins; ++i) {
    uint32_t faninIndex = gate.edges[i].getIndex();
    auto *operandCutSet = getCutSet(faninIndex);
    if (!operandCutSet)
      return logicOp->emitError("Failed to get cut set for fanin index ")
             << faninIndex;
    operandCutSets.push_back(operandCutSet);
  }

  // Reorder operands to increase early signature-pruning effectiveness during
  // recursive enumeration. Processing operands with larger cuts first tends to
  // hit the cut-size bound earlier and prunes subtrees sooner.
  std::sort(operandCutSets.begin(), operandCutSets.end(),
            [](const CutSet *a, const CutSet *b) {
              // Fanin count is <= 3, so recomputing this score in the
              // comparator is still cheaper than materializing extra arrays.
              auto maxCutSize = [](const CutSet *set) {
                unsigned maxSize = 0;
                for (const Cut *cut : set->getCuts())
                  maxSize = std::max(maxSize, cut->getInputSize());
                return maxSize;
              };
              return maxCutSize(a) > maxCutSize(b);
            });

  // Create the singleton cut (just this operation as a trivial cut)
  Cut *primaryInputCut = new (cutAllocator.Allocate())
      Cut(getAsTrivialCut(nodeIndex, logicNetwork));

  auto *resultCutSet = createNewCutSet(nodeIndex);
  processingOrder.push_back(nodeIndex);

  // Keep a local candidate list and perform dominance pruning during
  // generation to reduce finalize-time work.
  SmallVector<Cut *, 32> candidateCuts;
  candidateCuts.push_back(primaryInputCut);
  // Soft cap for generation-time candidate pool. Keep it comfortably above the
  // final per-root limit so we preserve quality, but prevent explosive growth
  // at larger cut settings (especially 10/8 and above).
  const unsigned generationCap =
      std::max(options.maxCutSizePerRoot + 8, options.maxCutSizePerRoot * 3);

  // Schedule cut set finalization when exiting this scope
  llvm::scope_exit prune([&]() {
    // Finalize cut set: remove duplicates, limit size, and match patterns
    resultCutSet->finalize(options, matchCut, logicNetwork);
  });

  // Cache maxCutInputSize to avoid repeated access
  unsigned maxInputSize = options.maxCutInputSize;

  // This lambda generates nested loops at runtime to iterate over all
  // combinations of cuts from N operands
  auto enumerateCutCombinations = [&](auto &&self, unsigned operandIdx,
                                      SmallVector<const Cut *, 3> &cutPtrs,
                                      uint64_t currentSig) -> void {
    // Base case: all operands processed, create merged cut
    if (operandIdx == numFanins) {
      // Efficient k-way merge: inputs are sorted, so dedup and constant
      // filtering can be done while merging. Abort early once we exceed the
      // cut-size limit to avoid building doomed merged cuts.
      SmallVector<uint32_t, 6> mergedInputs;
      auto appendMergedInput = [&](uint32_t value) {
        if (value == LogicNetwork::kConstant0 ||
            value == LogicNetwork::kConstant1)
          return true;
        if (!mergedInputs.empty() && mergedInputs.back() == value)
          return true;
        mergedInputs.push_back(value);
        return mergedInputs.size() <= maxInputSize;
      };

      if (numFanins == 1) {
        // Single input: copy while filtering constants.
        mergedInputs.reserve(
            std::min<size_t>(cutPtrs[0]->inputs.size(), maxInputSize));
        for (uint32_t value : cutPtrs[0]->inputs)
          if (!appendMergedInput(value))
            return;
      } else if (numFanins == 2) {
        // Two-way merge (common case for AND gates)
        const auto &inputs0 = cutPtrs[0]->inputs;
        const auto &inputs1 = cutPtrs[1]->inputs;
        mergedInputs.reserve(
            std::min<size_t>(inputs0.size() + inputs1.size(), maxInputSize));

        unsigned i = 0, j = 0;
        while (i < inputs0.size() || j < inputs1.size()) {
          uint32_t next;
          if (j == inputs1.size() ||
              (i < inputs0.size() && inputs0[i] <= inputs1[j])) {
            next = inputs0[i++];
            if (j < inputs1.size() && inputs1[j] == next)
              ++j;
          } else {
            next = inputs1[j++];
          }
          if (!appendMergedInput(next))
            return;
        }
      } else {
        // Three-way merge (for MAJ/MUX gates)
        const SmallVectorImpl<uint32_t> &inputs0 = cutPtrs[0]->inputs;
        const SmallVectorImpl<uint32_t> &inputs1 = cutPtrs[1]->inputs;
        const SmallVectorImpl<uint32_t> &inputs2 = cutPtrs[2]->inputs;
        mergedInputs.reserve(std::min<size_t>(
            inputs0.size() + inputs1.size() + inputs2.size(), maxInputSize));

        unsigned i = 0, j = 0, k = 0;
        while (i < inputs0.size() || j < inputs1.size() || k < inputs2.size()) {
          // Find minimum among available elements
          uint32_t minVal = UINT32_MAX;
          if (i < inputs0.size())
            minVal = std::min(minVal, inputs0[i]);
          if (j < inputs1.size())
            minVal = std::min(minVal, inputs1[j]);
          if (k < inputs2.size())
            minVal = std::min(minVal, inputs2[k]);

          // Advance all iterators pointing to minVal (handles duplicates)
          if (i < inputs0.size() && inputs0[i] == minVal)
            i++;
          if (j < inputs1.size() && inputs1[j] == minVal)
            j++;
          if (k < inputs2.size() && inputs2[k] == minVal)
            k++;

          if (!appendMergedInput(minVal))
            return;
        }
      }

      // Dominance check against already kept candidates.
      uint64_t mergedSig = 0;
      for (uint32_t idx : mergedInputs)
        mergedSig |= (1ULL << (idx & 63));

      for (const Cut *existing : candidateCuts) {
        if (existing->dominates(mergedInputs, mergedSig))
          return;
      }

      // Create the merged cut.
      Cut *mergedCut = new (cutAllocator.Allocate()) Cut();
      mergedCut->setRootIndex(nodeIndex);
      mergedCut->inputs = std::move(mergedInputs);
      mergedCut->computeSignature();

      // Store operand cuts for lazy truth table computation using fast
      // incremental method (after duplicate removal in finalize)
      mergedCut->setOperandCuts(cutPtrs);

      // Remove candidates dominated by the new cut.
      llvm::erase_if(candidateCuts, [&](const Cut *existing) {
        return mergedCut->dominates(*existing);
      });
      candidateCuts.push_back(mergedCut);

      if (candidateCuts.size() > generationCap) {
        auto keepBegin = candidateCuts.begin() + 1;
        // Keep trivial cut at index 0 and retain lexicographically smaller
        // low-support candidates first when trimming the generation pool.
        std::sort(keepBegin, candidateCuts.end(),
                  [](const Cut *a, const Cut *b) {
                    if (a->getInputSize() != b->getInputSize())
                      return a->getInputSize() < b->getInputSize();
                    return std::lexicographical_compare(
                        a->inputs.begin(), a->inputs.end(), b->inputs.begin(),
                        b->inputs.end());
                  });
        candidateCuts.resize(generationCap);
      }

      LLVM_DEBUG({
        if (mergedCut->inputs.size() >= 4) {
          llvm::dbgs() << "Generated cut for node " << nodeIndex;
          if (logicOp)
            llvm::dbgs() << " (" << logicOp->getName() << ")";
          llvm::dbgs() << " inputs=";
          llvm::interleaveComma(mergedCut->inputs, llvm::dbgs());
          llvm::dbgs() << "\n";
        }
      });
      return;
    }

    // Recursive case: iterate over cuts for current operand
    const CutSet *currentCutSet = operandCutSets[operandIdx];
    for (const Cut *cut : currentCutSet->getCuts()) {
      uint64_t cutSig = cut->getSignature();
      uint64_t newSig = currentSig | cutSig;
      if (static_cast<unsigned>(llvm::popcount(newSig)) > maxInputSize)
        continue; // Early rejection based on signature

      cutPtrs.push_back(cut);

      // Recurse to next operand
      self(self, operandIdx + 1, cutPtrs, newSig);

      cutPtrs.pop_back();
    }
  };

  // Start the recursion with empty cut pointer list and zero signature
  SmallVector<const Cut *, 3> cutPtrs;
  cutPtrs.reserve(numFanins);
  enumerateCutCombinations(enumerateCutCombinations, 0, cutPtrs, 0ULL);

  for (Cut *cut : candidateCuts)
    resultCutSet->addCut(cut);

  return success();
}

LogicalResult CutEnumerator::enumerateCuts(
    Operation *topOp,
    llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts for module: " << topOp->getName()
                          << "\n");
  // Topologically sort the logic network
  if (failed(topologicallySortLogicNetwork(topOp)))
    return failure();

  // Store the pattern matching function for use during cut finalization
  this->matchCut = matchCut;

  // Build the flat logic network representation for efficient simulation
  auto &block = topOp->getRegion(0).getBlocks().front();
  if (failed(logicNetwork.buildFromBlock(&block)))
    return failure();

  for (const auto &[index, gate] : llvm::enumerate(logicNetwork.getGates())) {
    // Skip non-logic gates.
    if (!gate.isLogicGate())
      continue;

    // Ensure cut set exists for each logic gate
    if (failed(visitLogicOp(index)))
      return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Cut enumeration completed successfully\n");
  return success();
}

const CutSet *CutEnumerator::getCutSet(uint32_t index) {
  // Check if cut set already exists
  auto it = cutSets.find(index);
  if (it == cutSets.end()) {
    // Create new cut set for an unprocessed value (primary input or other)
    CutSet *cutSet = new (cutSetAllocator.Allocate()) CutSet();
    Cut *trivialCut =
        new (cutAllocator.Allocate()) Cut(getAsTrivialCut(index, logicNetwork));
    cutSet->addCut(trivialCut);
    auto [newIt, inserted] = cutSets.insert({index, cutSet});
    assert(inserted && "Cut set already exists for this index");
    it = newIt;
  }

  return it->second;
}

/// Generate a human-readable name for a value used in test output.
/// This function creates meaningful names for values to make debug output
/// and test results more readable and understandable.
static StringRef
getTestVariableName(Value value, DenseMap<OperationName, unsigned> &opCounter) {
  if (auto *op = value.getDefiningOp()) {
    // Handle values defined by operations
    // First, check if the operation already has a name hint attribute
    if (auto name = op->getAttrOfType<StringAttr>("sv.namehint"))
      return name.getValue();

    // For single-result operations, generate a unique name based on operation
    // type
    if (op->getNumResults() == 1) {
      auto opName = op->getName();
      auto count = opCounter[opName]++;

      // Create a unique name by appending a counter to the operation name
      SmallString<16> nameStr;
      nameStr += opName.getStringRef();
      nameStr += "_";
      nameStr += std::to_string(count);

      // Store the generated name as a hint attribute for future reference
      auto nameAttr = StringAttr::get(op->getContext(), nameStr);
      op->setAttr("sv.namehint", nameAttr);
      return nameAttr;
    }

    // Multi-result operations or other cases get a generic name
    return "<unknown>";
  }

  // Handle block arguments
  auto blockArg = cast<BlockArgument>(value);
  auto hwOp =
      dyn_cast<circt::hw::HWModuleOp>(blockArg.getOwner()->getParentOp());
  if (!hwOp)
    return "<unknown>";

  // Return the formal input name from the hardware module
  return hwOp.getInputName(blockArg.getArgNumber());
}

void CutEnumerator::dump() const {
  DenseMap<OperationName, unsigned> opCounter;
  for (auto index : processingOrder) {
    auto it = cutSets.find(index);
    if (it == cutSets.end())
      continue;
    auto &cutSet = *it->second;
    mlir::Value value = logicNetwork.getValue(index);
    llvm::outs() << getTestVariableName(value, opCounter) << " "
                 << cutSet.getCuts().size() << " cuts:";
    for (const Cut *cut : cutSet.getCuts()) {
      llvm::outs() << " {";
      llvm::interleaveComma(cut->inputs, llvm::outs(), [&](uint32_t inputIdx) {
        mlir::Value inputVal = logicNetwork.getValue(inputIdx);
        llvm::outs() << getTestVariableName(inputVal, opCounter);
      });
      auto &pattern = cut->getMatchedPattern();
      llvm::outs() << "}"
                   << "@t" << cut->getTruthTable()->table.getZExtValue() << "d";
      if (pattern) {
        llvm::outs() << *std::max_element(pattern->getArrivalTimes().begin(),
                                          pattern->getArrivalTimes().end());
      } else {
        llvm::outs() << "0";
      }
    }
    llvm::outs() << "\n";
  }
  llvm::outs() << "Cut enumeration completed successfully\n";
}

void CutEnumerator::computeRequiredTimes() {
  // Collect the set of node indices that are in the processing order
  // (i.e., internal logic nodes). Nodes used by operations outside this set
  // are primary outputs (POs).
  DenseSet<uint32_t> logicNodes;
  for (auto index : processingOrder)
    logicNodes.insert(index);

  // Step 1: Initialize all required times to infinity.
  // Then set PO required times to their own worst arrival time.
  // This preserves each output's delay rather than only the critical path.
  for (auto &[index, cutSet] : cutSets)
    cutSet->requiredTime = std::numeric_limits<DelayType>::max();

  for (auto index : processingOrder) {
    auto it = cutSets.find(index);
    if (it == cutSets.end())
      continue;
    auto *cutSet = it->second;
    auto *bestCut = cutSet->getBestMatchedCut();
    if (!bestCut)
      continue;

    // Check if this node is a primary output: it has uses outside the logic
    // network (e.g., by hw.output or other non-logic ops).
    Value value = logicNetwork.getValue(index);
    bool isPO = false;
    if (value) {
      for (auto &use : value.getUses()) {
        Operation *user = use.getOwner();
        // If the user has no results, it's a sink (e.g., hw.output) → PO
        if (user->getNumResults() == 0) {
          isPO = true;
          break;
        }
        // If the user's result isn't a logic node we process, it's a PO
        Value userResult = user->getResult(0);
        if (!logicNetwork.hasIndex(userResult) ||
            !logicNodes.count(logicNetwork.getIndex(userResult))) {
          isPO = true;
          break;
        }
      }
    }

    if (isPO) {
      const auto &matched = *bestCut->getMatchedPattern();
      cutSet->requiredTime = matched.getWorstOutputArrivalTime();
      LLVM_DEBUG(llvm::dbgs() << "PO node " << index << " required time: "
                              << cutSet->requiredTime << "\n");
    }
  }

  // Step 3: Walk in reverse topo order, propagate required times
  for (auto it = processingOrder.rbegin(); it != processingOrder.rend(); ++it) {
    uint32_t index = *it;
    auto csIt = cutSets.find(index);
    if (csIt == cutSets.end())
      continue;
    auto *cutSet = csIt->second;
    auto *bestCut = cutSet->getBestMatchedCut();
    if (!bestCut)
      continue;

    const auto &matched = *bestCut->getMatchedPattern();
    auto delays = matched.getPattern()->match(
        const_cast<CutEnumerator &>(*this), *bestCut);
    if (!delays)
      continue;

    auto patternDelays = delays->getDelays();
    unsigned numInputs = bestCut->getInputSize();

    // For each input of the best cut, propagate required time
    for (unsigned i = 0; i < numInputs; ++i) {
      uint32_t inputIndex = bestCut->inputs[i];
      if (isAlwaysCutInput(logicNetwork, inputIndex))
        continue;

      auto inputIt = cutSets.find(inputIndex);
      if (inputIt == cutSets.end())
        continue;

      // requiredTime for input = this node's requiredTime - delay through gate
      // For single-output patterns, output index is 0
      DelayType delayThrough = patternDelays[i]; // delay from input i to output
      DelayType inputRequired = cutSet->requiredTime - delayThrough;
      inputIt->second->requiredTime =
          std::min(inputIt->second->requiredTime, inputRequired);
    }
  }
}

void CutEnumerator::reselectCutsForAreaFlow() {
  LLVM_DEBUG(llvm::dbgs() << "Re-selecting cuts for area flow...\n");

  // Initialize bestArrivalTime from the current delay-optimal mapping
  for (auto index : processingOrder) {
    auto it = cutSets.find(index);
    if (it == cutSets.end())
      continue;
    auto *bestCut = it->second->getBestMatchedCut();
    if (bestCut)
      it->second->bestArrivalTime =
          bestCut->getMatchedPattern()->getWorstOutputArrivalTime();
  }

  // Helper: compute arrival time of a cut based on current input bestCuts.
  // Uses CutSet::bestArrivalTime which is updated in topo order during
  // re-selection, so upstream values are always fresh.
  auto computeArrivalTime = [&](const Cut &cut) -> DelayType {
    const auto &matched = *cut.getMatchedPattern();
    auto matchResult =
        matched.getPattern()->match(const_cast<CutEnumerator &>(*this), cut);
    if (!matchResult)
      return std::numeric_limits<DelayType>::max();

    auto patternDelays = matchResult->getDelays();
    unsigned numInputs = cut.getInputSize();
    DelayType worstArrival = 0;

    for (unsigned i = 0; i < numInputs; ++i) {
      uint32_t inputIndex = cut.inputs[i];
      DelayType inputArrival = 0;
      if (!isAlwaysCutInput(logicNetwork, inputIndex)) {
        auto inputIt = cutSets.find(inputIndex);
        if (inputIt != cutSets.end())
          inputArrival = inputIt->second->bestArrivalTime;
      }
      worstArrival = std::max(worstArrival, patternDelays[i] + inputArrival);
    }
    return worstArrival;
  };

  // Walk in topo order (bottom-up from inputs to outputs)
  for (auto index : processingOrder) {
    auto it = cutSets.find(index);
    if (it == cutSets.end())
      continue;
    auto *cutSet = it->second;

    Cut *bestAreaFlowCut = nullptr;
    double bestFlow = std::numeric_limits<double>::max();

    for (Cut *cut : cutSet->getCuts()) {
      const auto &matched = cut->getMatchedPattern();
      if (!matched)
        continue;

      // Recompute arrival time based on current input bestCuts
      DelayType arrivalTime = computeArrivalTime(*cut);

      // Skip cuts that violate required time
      if (arrivalTime > cutSet->requiredTime)
        continue;

      // Compute area flow for this cut
      double flow = matched->getArea();
      for (uint32_t inputIndex : cut->inputs) {
        if (isAlwaysCutInput(logicNetwork, inputIndex))
          continue;

        auto inputIt = cutSets.find(inputIndex);
        if (inputIt == cutSets.end())
          continue;

        // Estimate fanout using the number of uses of the value
        Value inputValue = logicNetwork.getValue(inputIndex);
        unsigned numRefs = 0;
        if (inputValue)
          for (auto &use : inputValue.getUses())
            (void)use, ++numRefs;
        numRefs = std::max(numRefs, 1u);

        flow += inputIt->second->areaFlow / numRefs;
      }

      if (flow < bestFlow) {
        bestFlow = flow;
        bestAreaFlowCut = cut;
      }
    }

    if (bestAreaFlowCut) {
      cutSet->setBestCut(bestAreaFlowCut);
      cutSet->areaFlow = bestFlow;
      // Update arrival time for downstream nodes to use
      cutSet->bestArrivalTime = computeArrivalTime(*bestAreaFlowCut);
    }
  }
}

//===----------------------------------------------------------------------===//
// CutRewriter
//===----------------------------------------------------------------------===//

LogicalResult CutRewriter::run(Operation *topOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "Starting Cut Rewriter\n";
    llvm::dbgs() << "Mode: "
                 << (OptimizationStrategyArea == options.strategy ? "area"
                                                                  : "timing")
                 << "\n";
    llvm::dbgs() << "Max input size: " << options.maxCutInputSize << "\n";
    llvm::dbgs() << "Max cut size: " << options.maxCutSizePerRoot << "\n";
  });

  // Currently we don't support patterns with multiple outputs.
  // So check that.
  // TODO: This must be removed when we support multiple outputs.
  for (auto &pattern : patterns.patterns) {
    if (pattern->getNumOutputs() > 1) {
      return mlir::emitError(pattern->getLoc(),
                             "Cut rewriter does not support patterns with "
                             "multiple outputs yet");
    }
  }

  // First sort the operations topologically to ensure we can process them
  // in a valid order.
  if (failed(topologicallySortLogicNetwork(topOp)))
    return failure();

  // Enumerate cuts for all nodes (initial delay-oriented selection)
  if (failed(enumerateCuts(topOp)))
    return failure();

  // Dump cuts if testing priority cuts.
  if (options.testPriorityCuts) {
    cutEnumerator.dump();
    return success();
  }

  // Area recovery pass: re-select cuts to minimize area while preserving
  // the optimal delay (required time constraint).
  if (options.enableAreaRecovery) {
    cutEnumerator.computeRequiredTimes();
    cutEnumerator.reselectCutsForAreaFlow();
  }

  // Select best cuts and perform mapping
  if (failed(runBottomUpRewrite(topOp)))
    return failure();

  return success();
}

LogicalResult CutRewriter::enumerateCuts(Operation *topOp) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts...\n");

  return cutEnumerator.enumerateCuts(
      topOp, [&](const Cut &cut) -> std::optional<MatchedPattern> {
        // Match the cut against the patterns
        return patternMatchCut(cut);
      });
}

ArrayRef<std::pair<NPNClass, const CutRewritePattern *>>
CutRewriter::getMatchingPatternsFromTruthTable(const Cut &cut) const {
  if (patterns.npnToPatternMap.empty())
    return {};

  auto &npnClass = cut.getNPNClass();
  auto it = patterns.npnToPatternMap.find(
      {npnClass.truthTable.table, npnClass.truthTable.numInputs});
  if (it == patterns.npnToPatternMap.end())
    return {};
  return it->getSecond();
}

std::optional<MatchedPattern> CutRewriter::patternMatchCut(const Cut &cut) {
  if (cut.isTrivialCut())
    return {};

  const auto &network = cutEnumerator.getLogicNetwork();
  const CutRewritePattern *bestPattern = nullptr;
  SmallVector<DelayType, 4> inputArrivalTimes;
  SmallVector<DelayType, 1> bestArrivalTimes;
  double bestArea = 0.0;
  inputArrivalTimes.reserve(cut.getInputSize());
  bestArrivalTimes.reserve(cut.getOutputSize(network));

  // Compute arrival times for each input.
  if (failed(cut.getInputArrivalTimes(cutEnumerator, inputArrivalTimes)))
    return {};

  auto computeArrivalTimeAndPickBest =
      [&](const CutRewritePattern *pattern, const MatchResult &matchResult,
          llvm::function_ref<unsigned(unsigned)> mapIndex) {
        SmallVector<DelayType, 1> outputArrivalTimes;
        // Compute the maximum delay for each output from inputs.
        for (unsigned outputIndex = 0, outputSize = cut.getOutputSize(network);
             outputIndex < outputSize; ++outputIndex) {
          // Compute the arrival time for this output.
          DelayType outputArrivalTime = 0;
          auto delays = matchResult.getDelays();
          for (unsigned inputIndex = 0, inputSize = cut.getInputSize();
               inputIndex < inputSize; ++inputIndex) {
            // Map pattern input i to cut input through NPN transformations
            unsigned cutOriginalInput = mapIndex(inputIndex);
            outputArrivalTime =
                std::max(outputArrivalTime,
                         delays[outputIndex * inputSize + inputIndex] +
                             inputArrivalTimes[cutOriginalInput]);
          }

          outputArrivalTimes.push_back(outputArrivalTime);
        }

        // Update the arrival time
        if (!bestPattern ||
            compareDelayAndArea(options.strategy, matchResult.area,
                                outputArrivalTimes, bestArea,
                                bestArrivalTimes)) {
          LLVM_DEBUG({
            llvm::dbgs() << "== Matched Pattern ==============\n";
            llvm::dbgs() << "Matching cut: \n";
            cut.dump(llvm::dbgs(), network);
            llvm::dbgs() << "Found better pattern: "
                         << pattern->getPatternName();
            llvm::dbgs() << " with area: " << matchResult.area;
            llvm::dbgs() << " and input arrival times: ";
            for (unsigned i = 0; i < inputArrivalTimes.size(); ++i) {
              llvm::dbgs() << " " << inputArrivalTimes[i];
            }
            llvm::dbgs() << " and arrival times: ";

            for (auto arrivalTime : outputArrivalTimes) {
              llvm::dbgs() << " " << arrivalTime;
            }
            llvm::dbgs() << "\n";
            llvm::dbgs() << "== Matched Pattern End ==============\n";
          });

          bestArrivalTimes = std::move(outputArrivalTimes);
          bestArea = matchResult.area;
          bestPattern = pattern;
        }
      };

  auto matchingPatterns = getMatchingPatternsFromTruthTable(cut);
  if (matchingPatterns.empty()) {
    LLVM_DEBUG({
      if (cut.getInputSize() >= 4) {
        llvm::dbgs() << "No NPN match for cut with " << cut.getInputSize()
                     << " inputs\n";
        cut.dump(llvm::dbgs(), network);
        llvm::SmallString<64> cutTableStr;
        cut.getNPNClass().truthTable.table.toString(cutTableStr, 10, false);
        llvm::dbgs() << "NPN key: table=" << cutTableStr
                     << " inputs=" << cut.getNPNClass().truthTable.numInputs
                     << "\n";
        llvm::dbgs() << "Available NPN patterns for " << cut.getInputSize()
                     << " inputs:\n";
        for (const auto &entry : patterns.npnToPatternMap) {
          if (entry.first.second != cut.getInputSize())
            continue;
          llvm::SmallString<64> patternTableStr;
          entry.first.first.toString(patternTableStr, 10, false);
          llvm::dbgs() << "  table=" << patternTableStr << " patterns=";
          llvm::interleaveComma(entry.second, llvm::dbgs(),
                                [](const auto &pair) {
                                  llvm::dbgs() << pair.second->getPatternName();
                                });
          llvm::dbgs() << "\n";
        }
      }
    });
  }

  for (auto &[patternNPN, pattern] : matchingPatterns) {
    assert(patternNPN.truthTable.numInputs == cut.getInputSize() &&
           "Pattern input size must match cut input size");
    auto matchResult = pattern->match(cutEnumerator, cut);
    if (!matchResult)
      continue;
    auto &cutNPN = cut.getNPNClass();

    // Get the input mapping from pattern's NPN class to cut's NPN class
    SmallVector<unsigned> inputMapping;
    cutNPN.getInputPermutation(patternNPN, inputMapping);
    computeArrivalTimeAndPickBest(pattern, *matchResult,
                                  [&](unsigned i) { return inputMapping[i]; });
  }

  for (const CutRewritePattern *pattern : patterns.nonNPNPatterns) {
    if (auto matchResult = pattern->match(cutEnumerator, cut))
      computeArrivalTimeAndPickBest(pattern, *matchResult,
                                    [&](unsigned i) { return i; });
  }

  if (!bestPattern)
    return {}; // No matching pattern found

  return MatchedPattern(bestPattern, std::move(bestArrivalTimes), bestArea);
}

LogicalResult CutRewriter::runBottomUpRewrite(Operation *top) {
  LLVM_DEBUG(llvm::dbgs() << "Performing cut-based rewriting...\n");
  const auto &network = cutEnumerator.getLogicNetwork();
  const auto &cutSets = cutEnumerator.getCutSets();
  auto processingOrder = cutEnumerator.getProcessingOrder();

  // Note: Don't clear cutEnumerator yet - we need it during rewrite
  UnusedOpPruner pruner;
  PatternRewriter rewriter(top->getContext());

  // Process in reverse topological order
  for (auto index : llvm::reverse(processingOrder)) {
    auto it = cutSets.find(index);
    if (it == cutSets.end())
      continue;

    mlir::Value value = network.getValue(index);
    auto &cutSet = *it->second;

    if (value.use_empty()) {
      if (auto *op = value.getDefiningOp())
        pruner.eraseNow(op);
      continue;
    }

    if (isAlwaysCutInput(network, index)) {
      // If the value is a primary input, skip it
      LLVM_DEBUG(llvm::dbgs() << "Skipping inputs: " << value << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Cut set for value: " << value << "\n");
    auto *bestCut = cutSet.getBestMatchedCut();
    if (!bestCut) {
      if (options.allowNoMatch)
        continue; // No matching pattern found, skip this value
      return emitError(value.getLoc(), "No matching cut found for value: ")
             << value;
    }

    // Get the root operation from LogicNetwork
    auto *rootOp = network.getGate(bestCut->getRootIndex()).getOperation();
    rewriter.setInsertionPoint(rootOp);
    const auto &matchedPattern = bestCut->getMatchedPattern();
    auto result = matchedPattern->getPattern()->rewrite(rewriter, cutEnumerator,
                                                        *bestCut);
    if (failed(result))
      return failure();

    rewriter.replaceOp(rootOp, *result);

    if (options.attachDebugTiming) {
      auto array = rewriter.getI64ArrayAttr(matchedPattern->getArrivalTimes());
      (*result)->setAttr("test.arrival_times", array);
    }
  }

  // Clear the enumerator after rewriting is complete
  cutEnumerator.clear();
  return success();
}
