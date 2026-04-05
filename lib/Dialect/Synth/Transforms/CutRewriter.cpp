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
//  "Improvements to technology mapping for LUT-based FPGAs", Alan Mishchenko,
//  Satrajit Chatterjee and Robert Brayton, FPGA 2006
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/CutRewriter.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/TruthTable.h"
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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#define DEBUG_TYPE "synth-cut-rewriter"

using namespace circt;
using namespace circt::synth;

namespace {

static SmallVector<unsigned>
expandInputPermutation(const std::array<uint8_t, 4> &permutation) {
  SmallVector<unsigned> result;
  result.reserve(permutation.size());
  for (uint8_t index : permutation)
    result.push_back(index);
  return result;
}

struct NPNTransform4 {
  std::array<uint8_t, 16> outputToSource = {};
  std::array<uint8_t, 16> inverseOutputToSource = {};
  std::array<uint8_t, 4> inputPermutation = {};
  uint8_t inputNegation = 0;
  bool outputNegation = false;
};

static uint8_t
permuteNegationMask4(uint8_t negationMask,
                     const std::array<unsigned, 4> &permutation) {
  uint8_t result = 0;
  for (unsigned i = 0; i != permutation.size(); ++i)
    if (negationMask & (1u << permutation[i]))
      result |= 1u << i;
  return result;
}

static uint16_t
applyNPNTransform4(uint16_t truthTable,
                   const std::array<uint8_t, 16> &outputToSource,
                   bool outputNegation) {
  uint16_t result = 0;
  for (unsigned output = 0; output != 16; ++output) {
    unsigned bit = (truthTable >> outputToSource[output]) & 1u;
    if (outputNegation)
      bit ^= 1u;
    result |= static_cast<uint16_t>(bit << output);
  }
  return result;
}

static void
buildCanonicalOrderNPNTransforms4(SmallVectorImpl<NPNTransform4> &transforms) {
  transforms.clear();
  transforms.reserve(24 * 16 * 2);

  for (unsigned negMask = 0; negMask != 16; ++negMask) {
    std::array<unsigned, 4> permutation = {0, 1, 2, 3};
    do {
      std::array<unsigned, 4> inversePermutation = {};
      for (unsigned i = 0; i != 4; ++i)
        inversePermutation[permutation[i]] = i;

      uint8_t currentNegMask = permuteNegationMask4(negMask, permutation);
      for (unsigned outputNegation = 0; outputNegation != 2; ++outputNegation) {
        NPNTransform4 transform;
        transform.inputNegation = currentNegMask;
        transform.outputNegation = outputNegation;
        for (unsigned i = 0; i != 4; ++i)
          transform.inputPermutation[i] = permutation[i];

        for (unsigned output = 0; output != 16; ++output) {
          unsigned source = 0;
          for (unsigned input = 0; input != 4; ++input) {
            unsigned bit = (output >> inversePermutation[input]) & 1u;
            bit ^= (negMask >> input) & 1u;
            source |= bit << input;
          }
          transform.outputToSource[output] = source;
          transform.inverseOutputToSource[source] = output;
        }
        transforms.push_back(transform);
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }
}

static void
collectNPN4Representatives(ArrayRef<NPNTransform4> transforms,
                           SmallVectorImpl<uint16_t> &representatives) {
  llvm::BitVector seen(1u << 16, false);
  representatives.clear();

  for (unsigned seed = 0; seed != (1u << 16); ++seed) {
    if (seen.test(seed))
      continue;

    uint16_t representative = seed;
    for (const auto &transform : transforms) {
      uint16_t member = applyNPNTransform4(seed, transform.outputToSource,
                                           transform.outputNegation);
      seen.set(member);
      representative = std::min(representative, member);
    }
    representatives.push_back(representative);
  }
}

} // namespace

void circt::synth::collectCanonicalNPN4Representatives(
    SmallVectorImpl<uint16_t> &representatives) {
  SmallVector<NPNTransform4, 24 * 16 * 2> transforms;
  buildCanonicalOrderNPNTransforms4(transforms);
  collectNPN4Representatives(transforms, representatives);
}

NPNTable::NPNTable() {
  SmallVector<NPNTransform4, 24 * 16 * 2> transforms;
  buildCanonicalOrderNPNTransforms4(transforms);

  SmallVector<uint16_t, 222> representatives;
  collectCanonicalNPN4Representatives(representatives);

  llvm::BitVector initialized(entries4.size(), false);
  auto isBetterEntry = [&](const Entry4 &candidate, const Entry4 &current) {
    if (candidate.representative != current.representative)
      return candidate.representative < current.representative;
    if (candidate.inputNegation != current.inputNegation)
      return candidate.inputNegation < current.inputNegation;
    return candidate.outputNegation < current.outputNegation;
  };

  for (uint16_t representative : representatives) {
    for (const auto &transform : transforms) {
      uint16_t member =
          applyNPNTransform4(representative, transform.inverseOutputToSource,
                             transform.outputNegation);

      Entry4 candidate;
      candidate.representative = representative;
      candidate.inputPermutation = transform.inputPermutation;
      candidate.inputNegation = transform.inputNegation;
      candidate.outputNegation = transform.outputNegation;

      if (!initialized.test(member) ||
          isBetterEntry(candidate, entries4[member])) {
        entries4[member] = candidate;
        initialized.set(member);
      }
    }
  }

  assert(initialized.all() && "expected to populate all 4-input NPN entries");
}

bool NPNTable::lookup(const BinaryTruthTable &tt, NPNClass &result) const {
  if (tt.numInputs != 4 || tt.numOutputs != 1)
    return false;

  const auto &entry = entries4[tt.table.getZExtValue()];
  result =
      NPNClass(BinaryTruthTable(4, 1, llvm::APInt(16, entry.representative)),
               expandInputPermutation(entry.inputPermutation),
               entry.inputNegation, entry.outputNegation);
  return true;
}

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
            .Case<synth::dig::DotInverterOp>([&](synth::dig::DotInverterOp op) {
              if (op->getNumOperands() == 1) {
                const Signal inputSignal =
                    getOrCreateSignal(op.getOperand(0), op.isInverted(0));
                handleSingleInputGate(op, op.getResult(), inputSignal);
                return success();
              }
              if (op->getNumOperands() != 3) {
                handleOtherResults(op);
                return success();
              }
              const Signal xSignal =
                  getOrCreateSignal(op.getOperand(0), op.isInverted(0));
              const Signal ySignal =
                  getOrCreateSignal(op.getOperand(1), op.isInverted(1));
              const Signal zSignal =
                  getOrCreateSignal(op.getOperand(2), op.isInverted(2));
              addGate(op, LogicNetworkGate::Dot3, {xSignal, ySignal, zSignal});
              return success();
            })
            .Case<hw::ConstantOp>([&](hw::ConstantOp constOp) {
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
            .Case<synth::ChoiceOp>([&](synth::ChoiceOp choiceOp) {
              if (!choiceOp.getResult().getType().isInteger(1)) {
                handleOtherResults(choiceOp);
                return success();
              }
              addGate(choiceOp, LogicNetworkGate::Choice, choiceOp.getResult(),
                      {});
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

namespace {
static NPNClass computeNPNClassFromTruthTable(const BinaryTruthTable &truthTable,
                                              const CutRewriterOptions &options);
}

// Return true if the gate at the given index is always a cut input.
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
    // Topologically sort AIG ops, MIG ops, and dataflow ops. Other operations
    // can be scheduled.
    return !(isa<aig::AndInverterOp, mig::MajorityInverterOp>(op) ||
             isa<dig::DotInverterOp, synth::ChoiceOp>(op) ||
             isa<comb::XorOp, comb::AndOp, comb::OrOp, comb::ExtractOp,
                 comb::ReplicateOp, comb::ConcatOp>(op));
  };

  if (failed(topologicallySortGraphRegionBlocks(topOp, isOperationReady)))
    return emitError(topOp->getLoc(),
                     "failed to sort operations topologically");
  return success();
}

/// Get the truth table for operations within a block.
FailureOr<BinaryTruthTable> circt::synth::getTruthTable(ValueRange values,
                                                        Block *block) {
  llvm::SmallSetVector<Value, 4> inputArgs;
  for (Value arg : block->getArguments())
    inputArgs.insert(arg);

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

    // Support constants, trivial forwarding wires, and the boolean primitives
    // used by the current cut-rewrite clients.
    if (auto constant = dyn_cast<hw::ConstantOp>(&op)) {
      if (!constant.getType().isInteger(1))
        return constant.emitError("Constant results must be single bit");
      bool value = constant.getValueAttr().getValue()[0];
      eval[constant.getResult()] = value ? APInt::getAllOnes(1u << numInputs)
                                         : APInt(1u << numInputs, 0);
    } else if (auto wire = dyn_cast<hw::WireOp>(&op)) {
      auto it = eval.find(wire.getInput());
      if (it == eval.end())
        return wire.emitError("Input value not found in evaluation map");
      eval[wire.getResult()] = it->second;
    } else if (auto choice = dyn_cast<synth::ChoiceOp>(&op)) {
      auto it = eval.find(choice.getInputs().front());
      if (it == eval.end())
        return choice.emitError("Input value not found in evaluation map");
      eval[choice.getResult()] = it->second;
    } else if (auto andOp = dyn_cast<aig::AndInverterOp>(&op)) {
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
    } else if (auto digOp = dyn_cast<synth::dig::DotInverterOp>(&op)) {
      SmallVector<llvm::APInt, 3> inputs;
      inputs.reserve(digOp.getInputs().size());
      for (auto input : digOp.getInputs()) {
        auto it = eval.find(input);
        if (it == eval.end())
          return digOp.emitError("Input value not found in evaluation map");
        inputs.push_back(it->second);
      }
      eval[digOp.getResult()] = digOp.evaluate(inputs);
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
  // A cut is a trivial cut if it has no root (rootIndex == 0 sentinel)
  // and only one input
  return rootIndex == 0 && inputs.size() == 1;
}

const NPNClass &Cut::getNPNClass() const {
  CutRewriterOptions defaultOptions{};
  return getNPNClass(defaultOptions);
}

const NPNClass &Cut::getNPNClass(const CutRewriterOptions &options) const {
  // If the NPN is already computed, return it
  if (npnClass)
    return *npnClass;

  const auto &truthTable = *getTruthTable();
  npnClass.emplace(computeNPNClassFromTruthTable(truthTable, options));
  return *npnClass;
}

namespace {

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

void Cut::getPermutatedInputIndices(
    const CutRewriterOptions &options, const NPNClass &patternNPN,
    SmallVectorImpl<unsigned> &permutedIndices) const {
  const auto &npnClass = getNPNClass(options);
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
  case LogicNetworkGate::Dot3:
    return a ^ (c | (a & b));
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
  case LogicNetworkGate::Constant: {
    // Constant 0 or 1 - return all zeros or all ones
    if (index == LogicNetwork::kConstant0)
      result = llvm::APInt::getZero(1U << numInputs);
    else
      result = llvm::APInt::getAllOnes(1U << numInputs);
    break;
  }

  case LogicNetworkGate::PrimaryInput:
    // Should be in cache already as cut input
    llvm_unreachable("Primary input not in cache - not a cut input?");

  case LogicNetworkGate::And2:
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

  case LogicNetworkGate::Dot3: {
    result =
        applyGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]),
                           getEdgeTT(gate.edges[1]), getEdgeTT(gate.edges[2]));
    break;
  }

  case LogicNetworkGate::Identity: {
    result = applyGateSemantics(gate.getKind(), getEdgeTT(gate.edges[0]));
    break;
  }

  case LogicNetworkGate::Choice: {
    auto choiceOp = cast<synth::ChoiceOp>(gate.getOperation());
    result =
        simulateGate(network, network.getIndex(choiceOp.getInputs().front()),
                     cache, numInputs);
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

static llvm::APInt
expandTruthTableForMergedInputs(const llvm::APInt &tt,
                                ArrayRef<unsigned> inputMapping,
                                unsigned numMergedInputs) {
  unsigned numOrigInputs = inputMapping.size();
  unsigned mergedSize = 1U << numMergedInputs;

  if (numOrigInputs == numMergedInputs) {
    bool isIdentity = true;
    for (unsigned i = 0; i < numOrigInputs && isIdentity; ++i)
      isIdentity = inputMapping[i] == i;
    if (isIdentity)
      return tt.zext(mergedSize);
  }

  if (numMergedInputs <= 6) {
    uint64_t origTT = tt.getZExtValue();
    uint64_t result = 0;

    static constexpr uint64_t kVarMasks[6] = {
        0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL,
        0xF0F0F0F0F0F0F0F0ULL, 0xFF00FF00FF00FF00ULL,
        0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL};

    uint64_t sizeMask = (mergedSize == 64) ? ~0ULL : ((1ULL << mergedSize) - 1);
    unsigned origSize = 1U << numOrigInputs;
    for (unsigned origIdx = 0; origIdx < origSize; ++origIdx) {
      if (((origTT >> origIdx) & 1ULL) == 0)
        continue;

      uint64_t pattern = sizeMask;
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        unsigned mergedPos = inputMapping[i];
        bool origBit = ((origIdx >> i) & 1U) != 0;
        uint64_t varMask = kVarMasks[mergedPos] & sizeMask;
        pattern &= origBit ? varMask : ~varMask;
      }
      result |= pattern;
    }
    return llvm::APInt(mergedSize, result);
  }

  if (numMergedInputs <= 8) {
    constexpr unsigned kMaxFastInputs = 8;
    constexpr unsigned kMaxFastWords = (1U << kMaxFastInputs) / 64;

    struct FastMaskCache {
      bool initialized = false;
      std::array<
          std::array<std::array<uint64_t, kMaxFastWords>, kMaxFastInputs>,
          kMaxFastInputs + 1>
          masks{};
    };

    static FastMaskCache fastMaskCache;
    if (!fastMaskCache.initialized) {
      for (unsigned n = 1; n <= kMaxFastInputs; ++n) {
        unsigned size = 1U << n;
        for (unsigned var = 0; var < n; ++var)
          for (unsigned idx = 0; idx < size; ++idx) {
            if (((idx >> var) & 1U) == 0)
              continue;
            unsigned word = idx >> 6;
            unsigned bit = idx & 63;
            fastMaskCache.masks[n][var][word] |= 1ULL << bit;
          }
      }
      fastMaskCache.initialized = true;
    }

    unsigned numWords = (mergedSize + 63) >> 6;
    std::array<uint64_t, kMaxFastWords> fullMask{};
    for (unsigned w = 0; w < numWords; ++w)
      fullMask[w] = ~0ULL;
    if (unsigned tail = mergedSize & 63)
      fullMask[numWords - 1] = (1ULL << tail) - 1;

    std::array<uint64_t, kMaxFastWords> result{};
    unsigned origSize = 1U << numOrigInputs;
    for (unsigned origIdx = 0; origIdx < origSize; ++origIdx) {
      if (!tt[origIdx])
        continue;

      auto pattern = fullMask;
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        unsigned mergedPos = inputMapping[i];
        bool origBit = ((origIdx >> i) & 1U) != 0;
        const auto &varMask = fastMaskCache.masks[numMergedInputs][mergedPos];
        for (unsigned w = 0; w < numWords; ++w)
          pattern[w] &= origBit ? varMask[w] : ~varMask[w];
      }

      for (unsigned w = 0; w < numWords; ++w)
        result[w] |= pattern[w];
    }

    return llvm::APInt(mergedSize,
                       llvm::ArrayRef<uint64_t>(result.data(), numWords));
  }

  if (numMergedInputs <= 10) {
    constexpr unsigned kMaxMergedInputs = 10;
    constexpr unsigned kMaxWords = (1U << kMaxMergedInputs) / 64;

    unsigned numWords = (mergedSize + 63) >> 6;
    std::array<uint64_t, kMaxWords> result{};
    std::array<uint32_t, kMaxMergedInputs> mergedBitMasks{};
    std::array<unsigned, kMaxMergedInputs> origBitMasks{};
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      mergedBitMasks[i] = 1U << inputMapping[i];
      origBitMasks[i] = 1U << i;
    }

    const uint64_t *srcWords = tt.getRawData();
    for (unsigned mergedIdx = 0; mergedIdx < mergedSize; ++mergedIdx) {
      unsigned origIdx = 0;
      for (unsigned i = 0; i < numOrigInputs; ++i)
        if (mergedIdx & mergedBitMasks[i])
          origIdx |= origBitMasks[i];

      if (((srcWords[origIdx >> 6] >> (origIdx & 63)) & 1ULL) == 0)
        continue;
      result[mergedIdx >> 6] |= 1ULL << (mergedIdx & 63);
    }

    return llvm::APInt(mergedSize,
                       llvm::ArrayRef<uint64_t>(result.data(), numWords));
  }

  llvm::APInt result = llvm::APInt::getZero(mergedSize);
  for (unsigned mergedIdx = 0; mergedIdx < mergedSize; ++mergedIdx) {
    unsigned origIdx = 0;
    for (unsigned i = 0; i < numOrigInputs; ++i)
      if ((mergedIdx >> inputMapping[i]) & 1U)
        origIdx |= 1U << i;
    if (tt[origIdx])
      result.setBit(mergedIdx);
  }

  return result;
}

static llvm::APInt
getExpandedTruthTable(uint32_t operandIdx, bool isInverted,
                      const llvm::SmallVectorImpl<uint32_t> &mergedInputs,
                      unsigned numMergedInputs,
                      ArrayRef<const Cut *> operandCuts) {
  auto lookupMergedPos = [&](uint32_t idx) -> std::optional<unsigned> {
    auto it = llvm::find(mergedInputs, idx);
    if (it == mergedInputs.end())
      return std::nullopt;
    return static_cast<unsigned>(std::distance(mergedInputs.begin(), it));
  };

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

  if (auto pos = lookupMergedPos(operandIdx)) {
    return circt::createVarMask(numMergedInputs, *pos, !isInverted);
  }

  for (const Cut *cut : operandCuts) {
    if (!cut)
      continue;

    uint32_t cutOutput =
        cut->isTrivialCut() ? cut->inputs[0] : cut->getRootIndex();
    if (cutOutput != operandIdx)
      continue;

    const auto &cutTT = *cut->getTruthTable();
    SmallVector<unsigned, 8> mapping;
    mapping.reserve(cut->inputs.size());
    unsigned mergedPos = 0;
    for (uint32_t idx : cut->inputs) {
      while (mergedPos < mergedInputs.size() && mergedInputs[mergedPos] < idx)
        ++mergedPos;
      assert(mergedPos < mergedInputs.size() && mergedInputs[mergedPos] == idx &&
             "cut input must exist in merged inputs");
      mapping.push_back(mergedPos);
    }

    auto result =
        expandTruthTableForMergedInputs(cutTT.table, mapping, numMergedInputs);
    if (isInverted)
      result.flipAllBits();
    return result;
  }

  llvm_unreachable("Operand not found in cuts or merged inputs");
}

static BinaryTruthTable
computeTruthTableForGate(const LogicNetworkGate &rootGate,
                         const llvm::SmallVectorImpl<uint32_t> &mergedInputs,
                         ArrayRef<const Cut *> operandCuts) {
  unsigned numMergedInputs = mergedInputs.size();

  auto getEdgeTT = [&](unsigned edgeIdx) {
    const auto &edge = rootGate.edges[edgeIdx];
    return getExpandedTruthTable(edge.getIndex(), edge.isInverted(),
                                 mergedInputs, numMergedInputs, operandCuts);
  };

  switch (rootGate.getKind()) {
  case LogicNetworkGate::And2:
  case LogicNetworkGate::Xor2:
    return BinaryTruthTable(
        numMergedInputs, 1,
        applyGateSemantics(rootGate.getKind(), getEdgeTT(0), getEdgeTT(1)));
  case LogicNetworkGate::Maj3:
  case LogicNetworkGate::Dot3:
    return BinaryTruthTable(
        numMergedInputs, 1,
        applyGateSemantics(rootGate.getKind(), getEdgeTT(0), getEdgeTT(1),
                           getEdgeTT(2)));
  case LogicNetworkGate::Identity:
    return BinaryTruthTable(
        numMergedInputs, 1,
        applyGateSemantics(rootGate.getKind(), getEdgeTT(0)));
  case LogicNetworkGate::Choice: {
    assert(operandCuts.size() == 1 &&
           "choice cuts must keep exactly one selected operand cut");
    const Cut *selectedCut = operandCuts.front();
    uint32_t selectedOutput = selectedCut->isTrivialCut()
                                  ? selectedCut->inputs[0]
                                  : selectedCut->getRootIndex();
    return BinaryTruthTable(
        numMergedInputs, 1,
        getExpandedTruthTable(selectedOutput, false, mergedInputs,
                              numMergedInputs, operandCuts));
  }
  default:
    llvm_unreachable("Unsupported operation for truth table computation");
  }
}

void Cut::computeTruthTableFromOperands(const LogicNetwork &network) {
  if (isTrivialCut()) {
    computeTruthTable(network);
    return;
  }

  if (operandCuts.empty()) {
    computeTruthTable(network);
    return;
  }

  const auto &rootGate = network.getGate(rootIndex);
  truthTable.emplace(computeTruthTableForGate(rootGate, inputs, operandCuts));
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

static Cut getAsTrivialCut(uint32_t index, const LogicNetwork &network) {
  // Create a trivial cut for a value
  Cut cut;
  cut.inputs.push_back(index);
  // Compute truth table eagerly for trivial cut
  cut.computeTruthTable(network);
  cut.setSignature(1ULL << (index % 64)); // Set signature bit for this input
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

const CutRewritePattern *MatchedPattern::getPattern() const {
  assert(pattern && "Pattern must be set to get the pattern");
  return pattern;
}

double MatchedPattern::getArea() const {
  assert(pattern && "Pattern must be set to get area");
  return area;
}

namespace {

struct StructuralEntry {
  OperationName opName;
  SmallVector<uint64_t, 3> operandSignals;
  Value value;
};

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

static MatchBinding getBindingForPattern(const Cut &cut,
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

static void enumerateGreedyCutsForRoot(Value root, unsigned maxInputs,
                                       unsigned maxExpansionDepth,
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

static bool sameStructuralKey(OperationName opName,
                              ArrayRef<uint64_t> operandSignals,
                              const StructuralEntry &entry) {
  return entry.opName == opName && entry.operandSignals == operandSignals;
}

static Value findStructuralValue(ArrayRef<StructuralEntry> entries,
                                 OperationName opName,
                                 ArrayRef<uint64_t> operandSignals) {
  for (const auto &entry : entries)
    if (sameStructuralKey(opName, operandSignals, entry))
      return entry.value;
  return {};
}

static void appendStructuralEntry(SmallVectorImpl<StructuralEntry> &entries,
                                  OperationName opName,
                                  SmallVector<uint64_t, 3> operandSignals,
                                  Value value) {
  entries.push_back(
      StructuralEntry{opName, std::move(operandSignals), value});
}

static void collectStructuralEntries(
    Block *block, DenseMap<Value, uint64_t> &signalIds,
    SmallVectorImpl<StructuralEntry> &entries) {
  signalIds.clear();
  entries.clear();

  uint64_t nextId = 1;
  for (Value arg : block->getArguments())
    signalIds.try_emplace(arg, nextId++);

  auto getSignalId = [&](Value value) -> uint64_t {
    auto [it, inserted] = signalIds.try_emplace(value, nextId);
    if (inserted)
      ++nextId;
    return it->second;
  };

  for (Operation &op : block->getOperations()) {
    if (!op.getNumResults() || !op.getResult(0).getType().isInteger(1))
      continue;

    if (auto andOp = dyn_cast<aig::AndInverterOp>(&op)) {
      SmallVector<uint64_t, 3> operandSignals;
      operandSignals.reserve(andOp.getNumOperands());
      for (auto [i, operand] : llvm::enumerate(andOp.getInputs())) {
        operandSignals.push_back((getSignalId(operand) << 1) |
                                 static_cast<uint64_t>(andOp.isInverted(i)));
      }
      canonicalizeStructuralOperands(
          andOp.getNumOperands() == 1 ? CandidateRecipeNode::Identity
                                      : CandidateRecipeNode::And,
          operandSignals);
      appendStructuralEntry(entries, op.getName(), std::move(operandSignals),
                            op.getResult(0));
    } else if (auto xorOp = dyn_cast<comb::XorOp>(&op)) {
      SmallVector<uint64_t, 3> operandSignals;
      operandSignals.reserve(2);
      for (Value operand : xorOp->getOperands())
        operandSignals.push_back(getSignalId(operand) << 1);
      canonicalizeStructuralOperands(CandidateRecipeNode::Xor, operandSignals);
      appendStructuralEntry(entries, op.getName(), std::move(operandSignals),
                            op.getResult(0));
    } else if (auto majOp = dyn_cast<mig::MajorityInverterOp>(&op)) {
      SmallVector<uint64_t, 3> operandSignals;
      operandSignals.reserve(majOp.getNumOperands());
      for (auto [i, operand] : llvm::enumerate(majOp.getOperands())) {
        operandSignals.push_back((getSignalId(operand) << 1) |
                                 static_cast<uint64_t>(majOp.isInverted(i)));
      }
      canonicalizeStructuralOperands(
          majOp.getNumOperands() == 1 ? CandidateRecipeNode::Identity
                                      : CandidateRecipeNode::Maj,
          operandSignals);
      appendStructuralEntry(entries, op.getName(), std::move(operandSignals),
                            op.getResult(0));
    } else if (auto dotOp = dyn_cast<dig::DotInverterOp>(&op)) {
      SmallVector<uint64_t, 3> operandSignals;
      operandSignals.reserve(dotOp.getNumOperands());
      for (auto [i, operand] : llvm::enumerate(dotOp.getOperands())) {
        operandSignals.push_back((getSignalId(operand) << 1) |
                                 static_cast<uint64_t>(dotOp.isInverted(i)));
      }
      canonicalizeStructuralOperands(
          dotOp.getNumOperands() == 1 ? CandidateRecipeNode::Identity
                                      : CandidateRecipeNode::Dot,
          operandSignals);
      appendStructuralEntry(entries, op.getName(), std::move(operandSignals),
                            op.getResult(0));
    } else {
      getSignalId(op.getResult(0));
    }
  }
}

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
                                        Value rootValue, ArrayRef<Value> cutInputs,
                                        DenseMap<Value, uint64_t> &signalIds,
                                        ArrayRef<StructuralEntry> existing) {
  ProbeResult result;
  struct LocalProbeEntry {
    OperationName opName;
    SmallVector<uint64_t, 3> operandSignals;
    uint64_t signalId = 0;
  };

  uint64_t nextSignalId = signalIds.size() + 1;
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
    auto it = signalIds.find(inputValue);
    assert(it != signalIds.end() && "cut input must have a signal id");
    uint64_t rawSignal = it->second;
    if (((binding.inputNegationMask >> inputIndex) & 1u) == 0)
      return rawSignal;

    SmallVector<uint64_t, 3> operandSignals = {(rawSignal << 1) | 1u};
    OperationName inverterName =
        getRecipeOperationName(inputValue.getContext(), recipe,
                               CandidateRecipeNode::Identity);
    if (Value existingValue =
            findStructuralValue(existing, inverterName, operandSignals)) {
      result.reusedOps.push_back(existingValue.getDefiningOp());
      return signalIds.lookup(existingValue);
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
      if (Value existingValue =
              findStructuralValue(existing, opName, operandSignals)) {
        if (auto *op = existingValue.getDefiningOp())
          result.reusedOps.push_back(op);
        nodeSignals[i] = signalIds.lookup(existingValue);
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
    if (Value existingValue =
            findStructuralValue(existing, inverterName, operandSignals)) {
      if (auto *op = existingValue.getDefiningOp())
        result.reusedOps.push_back(op);
      result.rootSignal = signalIds.lookup(existingValue);
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
                                        DenseMap<Value, uint64_t> &signalIds,
                                        ArrayRef<StructuralEntry> existing) {
  return probeCandidateRecipe(recipe, binding, cut.root, cut.leaves, signalIds,
                              existing);
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
    if (!isAreaCountedLogicOp(op) || preserved.contains(op) || doomedSet.contains(op))
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
                                                const CandidateRecipe &recipe) {
  auto *rootOp = cut.root.getDefiningOp();
  assert(rootOp && "cut root must be a valid operation");

  DenseMap<Value, uint64_t> signalIds;
  SmallVector<StructuralEntry, 64> existing;
  collectStructuralEntries(rootOp->getBlock(), signalIds, existing);
  uint64_t nextSignalId = signalIds.size() + 1;

  auto assignSignalId = [&](Value value) -> uint64_t {
    auto [it, inserted] = signalIds.try_emplace(value, nextSignalId);
    if (inserted)
      ++nextSignalId;
    return it->second;
  };

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
    uint64_t rawSignal = signalIds.lookup(inputValue);
    if (((binding.inputNegationMask >> inputIndex) & 1u) == 0)
      return {inputValue, rawSignal};

    SmallVector<uint64_t, 3> operandSignals = {(rawSignal << 1) | 1u};
    OperationName inverterName =
        getRecipeOperationName(builder.getContext(), recipe,
                               CandidateRecipeNode::Identity);
    if (Value existingValue =
            findStructuralValue(existing, inverterName, operandSignals))
      return {existingValue, signalIds.lookup(existingValue)};

    if (auto it = freeInverters.find(rawSignal); it != freeInverters.end())
      return {it->second, signalIds.lookup(it->second)};

    Value inv =
        materializeRecipeInverter(builder, loc, recipe.inverterKind, inputValue);
    freeInverters[rawSignal] = inv;
    uint64_t signal = assignSignalId(inv);
    return {inv, signal};
  };

  SmallVector<StructuralEntry, 8> localEntries;

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
      nodeSignals[i] = assignSignalId(nodeValues[i]);
      break;
    case CandidateRecipeNode::Const1:
      nodeValues[i] = getConstant(true);
      nodeSignals[i] = assignSignalId(nodeValues[i]);
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

      if (Value existingValue =
              findStructuralValue(existing, opName, operandSignals)) {
        nodeValues[i] = existingValue;
        nodeSignals[i] = signalIds.lookup(existingValue);
        break;
      }
      if (Value localValue =
              findStructuralValue(localEntries, opName, operandSignals)) {
        nodeValues[i] = localValue;
        nodeSignals[i] = signalIds.lookup(localValue);
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
      nodeSignals[i] = assignSignalId(value);
      appendStructuralEntry(localEntries, opName, std::move(operandSignals),
                            value);
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

} // namespace

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
// exists another cut that is a subset of it.
static void removeDuplicateAndNonMinimalCuts(SmallVectorImpl<Cut *> &cuts) {
  auto dumpInputs = [](llvm::raw_ostream &os,
                       const llvm::SmallVectorImpl<uint32_t> &inputs) {
    os << "{";
    llvm::interleaveComma(inputs, os);
    os << "}";
  };
  // Sort by size, then lexicographically by inputs. This enables cheap exact
  // duplicate elimination and tighter candidate filtering for subset checks.
  std::stable_sort(cuts.begin(), cuts.end(), [](const Cut *a, const Cut *b) {
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

  // Select the best cut from the remaining candidates.
  bestCut = nullptr;
  for (Cut *cut : cuts) {
    const auto &currentMatch = cut->getMatchedPattern();
    if (!currentMatch)
      continue;
    bestCut = cut;
    break;
  }

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

//===----------------------------------------------------------------------===//
// CutRewritePattern
//===----------------------------------------------------------------------===//

bool CutRewritePattern::useTruthTableMatcher(
    SmallVectorImpl<NPNClass> &matchingNPNClasses) const {
  return false;
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

FailureOr<Operation *>
SpeculativeCutRewritePattern::rewrite(OpBuilder &builder, const LocalCut &cut,
                                      const MatchedPattern &match) const {
  auto recipe = speculate(cut);
  if (failed(recipe))
    return failure();
  return materializeRecipe(builder, cut, match.getBinding(), *recipe);
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
  return rewrite(builder, getLocalCut(enumerator, cut), match);
}

//===----------------------------------------------------------------------===//
// CutRewritePatternSet
//===----------------------------------------------------------------------===//

static void dumpRegisteredCutRewritePattern(const CutRewritePattern &pattern,
                                            ArrayRef<NPNClass> npnClasses,
                                            bool usesTruthTableMatcher) {
  LLVM_DEBUG({
    llvm::dbgs() << "registered cut-rewrite pattern '"
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

CutRewritePatternSet::CutRewritePatternSet(
    llvm::SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns)
    : patterns(std::move(patterns)) {
  // Initialize the NPN to pattern map
  for (auto &pattern : this->patterns) {
    SmallVector<NPNClass, 2> npnClasses;
    auto result = pattern->useTruthTableMatcher(npnClasses);
    dumpRegisteredCutRewritePattern(*pattern, npnClasses, result);
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
    : cutAllocator(stats.numCutsCreated),
      cutSetAllocator(stats.numCutSetsCreated), options(options) {}

CutSet *CutEnumerator::createNewCutSet(uint32_t index) {
  CutSet *cutSet = cutSetAllocator.create();
  auto [cutSetPtr, inserted] = cutSets.try_emplace(index, cutSet);
  assert(inserted && "Cut set already exists for this index");
  return cutSetPtr->second;
}

static Cut *getTrivialCut(CutSet *cutSet) {
  auto cuts = cutSet->getCuts();
  if (cuts.size() != 1 || !cuts.front()->isTrivialCut())
    return nullptr;
  return cuts.front();
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

  if (gate.getKind() == LogicNetworkGate::Choice) {
    auto choiceOp = cast<synth::ChoiceOp>(logicOp);

    auto cutSetIt = cutSets.find(nodeIndex);
    CutSet *resultCutSet = nullptr;
    if (cutSetIt == cutSets.end()) {
      resultCutSet = createNewCutSet(nodeIndex);
      Cut *primaryInputCut =
          cutAllocator.create(getAsTrivialCut(nodeIndex, logicNetwork));
      resultCutSet->addCut(primaryInputCut);
    } else {
      resultCutSet = cutSetIt->second;
      assert(getTrivialCut(resultCutSet) &&
             "existing cut set for unvisited logic op must be trivial");
    }
    processingOrder.push_back(nodeIndex);

    llvm::scope_exit prune(
        [&]() { resultCutSet->finalize(options, matchCut, logicNetwork); });

    for (Value operand : choiceOp.getInputs()) {
      auto *operandCutSet = getCutSet(logicNetwork.getIndex(operand));
      if (!operandCutSet)
        return logicOp->emitError("Failed to get cut set for choice operand");

      for (const Cut *operandCut : operandCutSet->getCuts()) {
        if (operandCut->isTrivialCut())
          continue;
        auto *choiceCut = cutAllocator.create();
        choiceCut->setRootIndex(nodeIndex);
        choiceCut->inputs = operandCut->inputs;
        choiceCut->setOperandCuts({operandCut});
        resultCutSet->addCut(choiceCut);
      }
    }
    return success();
  }

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

  // A vector to hold cut sets for each operand along with their max cut input
  // size.
  SmallVector<std::pair<const CutSet *, unsigned>, 2> operandCutSets;
  operandCutSets.reserve(numFanins);

  // Collect cut sets for each fanin (using LogicNetwork edges)
  for (unsigned i = 0; i < numFanins; ++i) {
    uint32_t faninIndex = gate.edges[i].getIndex();
    auto *operandCutSet = getCutSet(faninIndex);
    if (!operandCutSet)
      return logicOp->emitError("Failed to get cut set for fanin index ")
             << faninIndex;

    // Find the largest cut size among the operand's cuts for sorting heuristic
    // later.
    unsigned maxInputCutSize = 0;
    for (auto *cut : operandCutSet->getCuts())
      maxInputCutSize = std::max(maxInputCutSize, cut->getInputSize());
    operandCutSets.push_back(std::make_pair(operandCutSet, maxInputCutSize));
  }

  // Create the trivial cut for this node's output
  Cut *primaryInputCut =
      nullptr;

  auto cutSetIt = cutSets.find(nodeIndex);
  CutSet *resultCutSet = nullptr;
  if (cutSetIt == cutSets.end()) {
    primaryInputCut = cutAllocator.create(getAsTrivialCut(nodeIndex, logicNetwork));
    resultCutSet = createNewCutSet(nodeIndex);
    resultCutSet->addCut(primaryInputCut);
  } else {
    resultCutSet = cutSetIt->second;
    assert(getTrivialCut(resultCutSet) &&
           "existing cut set for unvisited logic op must be trivial");
  }
  processingOrder.push_back(nodeIndex);

  // Schedule cut set finalization when exiting this scope
  llvm::scope_exit prune([&]() {
    // Finalize cut set: remove duplicates, limit size, and match patterns
    resultCutSet->finalize(options, matchCut, logicNetwork);
  });

  // Sort operand cut sets by their largest cut size in descending order. This
  // heuristic improves efficiency of the k-way merge when generating cuts for
  // the current node by maximizing the chance of early pruning when the merged
  // cut exceeds the input size limit.
  llvm::stable_sort(operandCutSets,
                    [](const std::pair<const CutSet *, unsigned> &a,
                       const std::pair<const CutSet *, unsigned> &b) {
                      return a.second > b.second;
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

      // Create the merged cut.
      Cut *mergedCut = cutAllocator.create();
      mergedCut->setRootIndex(nodeIndex);
      mergedCut->inputs = std::move(mergedInputs);

      // Store operand cuts for lazy truth table computation using fast
      // incremental method (after duplicate removal in finalize)
      mergedCut->setOperandCuts(cutPtrs);
      mergedCut->setSignature(currentSig);
      resultCutSet->addCut(mergedCut);

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
    const CutSet *currentCutSet = operandCutSets[operandIdx].first;
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
    CutSet *cutSet = cutSetAllocator.create();
    Cut *trivialCut = cutAllocator.create(getAsTrivialCut(index, logicNetwork));
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

  auto &npnClass = cut.getNPNClass(options);
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
  MatchBinding bestBinding;
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
          const MatchBinding &binding,
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
          bestBinding = binding;
        }
      };

  for (auto &[patternNPN, pattern] : getMatchingPatternsFromTruthTable(cut)) {
    assert(patternNPN.truthTable.numInputs == cut.getInputSize() &&
           "Pattern input size must match cut input size");
    auto matchResult = pattern->match(cutEnumerator, cut);
    if (!matchResult)
      continue;
    auto &cutNPN = cut.getNPNClass(options);

    // Get the input mapping from pattern's NPN class to cut's NPN class
    SmallVector<unsigned> inputMapping;
    cutNPN.getInputPermutation(patternNPN, inputMapping);
    computeArrivalTimeAndPickBest(pattern, *matchResult,
                                  getBindingForPattern(cut, &patternNPN, options),
                                  [&](unsigned i) { return inputMapping[i]; });
  }

  for (const CutRewritePattern *pattern : patterns.nonNPNPatterns) {
    if (auto matchResult = pattern->match(cutEnumerator, cut))
      computeArrivalTimeAndPickBest(
          pattern, *matchResult, getIdentityBinding(cut.getInputSize()),
                                    [&](unsigned i) { return i; });
  }

  if (!bestPattern)
    return {}; // No matching pattern found

  return MatchedPattern(bestPattern, std::move(bestArrivalTimes), bestArea,
                        std::move(bestBinding));
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
                                                        *bestCut,
                                                        *matchedPattern);
    if (failed(result))
      return failure();

    rewriter.replaceOp(rootOp, *result);
    cutEnumerator.noteCutRewritten();

    if (options.attachDebugTiming) {
      auto array = rewriter.getI64ArrayAttr(matchedPattern->getArrivalTimes());
      (*result)->setAttr("test.arrival_times", array);
    }
  }

  // Clear the enumerator after rewriting is complete
  cutEnumerator.clear();
  return success();
}

struct GreedyRewriteCandidate {
  LocalCut cut;
  MatchedPattern match;
  CandidateRecipe recipe;
  int gain = 0;
  unsigned newArea = 0;
};

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

      DenseMap<Value, uint64_t> signalIds;
      SmallVector<StructuralEntry, 64> existing;
      collectStructuralEntries(block, signalIds, existing);

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

        ProbeResult probe = probeCandidateRecipe(*recipe, matched->getBinding(), cut,
                                                 signalIds, existing);
        if (probe.rootSignal == signalIds.lookup(rootValue))
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
              GreedyRewriteCandidate{cut, *matched, *recipe, gain, probe.newArea};
        }
      }

      if (!bestCandidate)
        continue;

      auto *rewriteRootOp = bestCandidate->cut.root.getDefiningOp();
      PatternRewriter rewriter(topOp->getContext());
      rewriter.setInsertionPoint(rewriteRootOp);
      auto *specPattern = static_cast<const SpeculativeCutRewritePattern *>(
          bestCandidate->match.getPattern());
      auto result = specPattern->rewrite(rewriter, bestCandidate->cut,
                                         bestCandidate->match);
      if (failed(result))
        return failure();

      DenseMap<Value, uint64_t> freshSignalIds;
      SmallVector<StructuralEntry, 64> freshExisting;
      collectStructuralEntries(block, freshSignalIds, freshExisting);
      ProbeResult freshProbe = probeCandidateRecipe(
          bestCandidate->recipe, bestCandidate->match.getBinding(),
          bestCandidate->cut, freshSignalIds, freshExisting);
      auto doomed = computeDoomedCone(rewriteRootOp, freshProbe.reusedOps);

      rewriter.replaceOp(rewriteRootOp, *result);
      for (Operation *doomedOp : doomed)
        if (doomedOp != rewriteRootOp &&
            deferredDoomedSet.insert(doomedOp).second)
          deferredDoomedOps.push_back(doomedOp);
      ++stats.numCutsRewritten;
      ++iteration;
      changed = true;
    }

    UnusedOpPruner pruner;
    for (Operation *doomedOp : deferredDoomedOps)
      if (doomedOp->getBlock() && doomedOp->use_empty())
        pruner.eraseNow(doomedOp);

    if ((options.maxIterations != 0 && iteration >= options.maxIterations))
      return success();
    if (!changed)
      return success();
  }

  return success();
}
