//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements network conversion utilities between MLIR and
// mockturtle networks. All classes here are implementation details.
//
//===----------------------------------------------------------------------===//

#include "NetworkConversion.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mockturtle/networks/aig.hpp"
#include "mockturtle/networks/block.hpp"
#include "mockturtle/networks/mig.hpp"
#include "mockturtle/views/cell_view.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mockturtle-network-conversion"

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::mockturtle_integration;

namespace {

//===----------------------------------------------------------------------===//
// Internal Helper Classes
//===----------------------------------------------------------------------===//

/// State tracking exported network nodes and their corresponding MLIR values.
struct ExportedState {
  /// Maps primary output indices to their original MLIR values
  llvm::MapVector<size_t, Value> outputToValue;
  /// Maps primary input indices to their original MLIR values
  llvm::MapVector<size_t, Value> indexToInput;
};

//===----------------------------------------------------------------------===//
// Network Exporter - MLIR to Mockturtle
//===----------------------------------------------------------------------===//

/// Converts MLIR operations to mockturtle network nodes.
/// Template parameters:
/// - Ntk: Mockturtle network type (aig_network, mig_network)
/// - OpType: MLIR operation type to convert (AndInverterOp, MajorityInverterOp)
template <typename Ntk, typename OpType>
class NetworkExporter {
public:
  using signal = mockturtle::signal<Ntk>;
  using node = typename Ntk::node;

  NetworkExporter(Block *block, Ntk &ntk, ExportedState &state)
      : block(block), ntk(ntk), state(state) {}

  /// Export all operations in the block to the network
  LogicalResult run();

private:
  /// Visit an operation and convert it to a network node
  /// Returns true if the operation was successfully converted
  bool visit(Operation *op);

  /// Get or create a network signal for an MLIR value
  /// Creates a primary input if the value is not yet mapped
  signal getOrCreateNode(Value v);

  /// Register a value as a primary output of the network
  void registerOutput(Value value);

  /// Map an MLIR value to a network signal
  void mapValueToSignal(Value v, signal s) {
    auto result = valueToNodeMap.insert({v, s});
    (void)result;
    assert(result.second && "Value already mapped to a signal");
  }

  Block *block;
  Ntk &ntk;
  ExportedState &state;
  DenseMap<Value, signal> valueToNodeMap;
};

using AIGExporter =
    NetworkExporter<mockturtle::aig_network, synth::aig::AndInverterOp>;
using MIGExporter =
    NetworkExporter<mockturtle::mig_network, synth::mig::MajorityInverterOp>;

//===----------------------------------------------------------------------===//
// Network Exporter Implementation
//===----------------------------------------------------------------------===//

template <typename Ntk, typename OpType>
LogicalResult NetworkExporter<Ntk, OpType>::run() {
  for (auto &op : block->getOperations()) {
    if (visit(&op)) {
      for (auto res : op.getResults()) {
        if (llvm::any_of(res.getUsers(),
                         [&](Operation *user) { return !isa<OpType>(user); })) {
          registerOutput(res);
          break;
        }
      }
    }
  }
  return success();
}

template <typename Ntk, typename OpType>
bool NetworkExporter<Ntk, OpType>::visit(Operation *op) {
  auto combOp = dyn_cast<OpType>(op);
  if (!combOp)
    return false;

  SmallVector<signal, 4> operands;
  auto inverted = combOp.getInverted();

  for (auto [operand, inv] : llvm::zip(combOp.getOperands(), inverted))
    operands.push_back(getOrCreateNode(operand) ^ inv);

  if constexpr (std::is_same_v<OpType, synth::aig::AndInverterOp>) {
    if (operands.size() == 1) {
      mapValueToSignal(combOp.getResult(), operands[0]);
      return true;
    }
    assert(operands.size() == 2 && "AIG AND must have 2 operands");
    mapValueToSignal(combOp.getResult(),
                     ntk.create_and(operands[0], operands[1]));
  } else if constexpr (std::is_same_v<OpType, synth::mig::MajorityInverterOp>) {
    if (operands.size() == 1) {
      mapValueToSignal(combOp.getResult(), operands[0]);
      return true;
    }
    assert(operands.size() == 3 && "MIG Majority must have 3 operands");
    mapValueToSignal(combOp.getResult(),
                     ntk.create_maj(operands[0], operands[1], operands[2]));
  }
  return true;
}

template <typename Ntk, typename OpType>
typename NetworkExporter<Ntk, OpType>::signal
NetworkExporter<Ntk, OpType>::getOrCreateNode(Value v) {
  auto it = valueToNodeMap.find(v);
  if (it != valueToNodeMap.end())
    return it->second;

  if (auto *defOp = v.getDefiningOp()) {
    // Recursively visit the defining operation.
    // Only proceed if the operation is successfully converted.
    if (visit(defOp))
      return valueToNodeMap.at(v);
  }

  // Create primary input
  auto pi = ntk.create_pi();
  auto piIndex = ntk.pi_index(pi.index);
  state.indexToInput[piIndex] = v;
  mapValueToSignal(v, pi);
  return pi;
}

template <typename Ntk, typename OpType>
void NetworkExporter<Ntk, OpType>::registerOutput(Value value) {
  auto node = getOrCreateNode(value);
  auto po = ntk.create_po(node);
  state.outputToValue[po] = value;
}

//===----------------------------------------------------------------------===//
// Network Importer - Mockturtle to MLIR
//===----------------------------------------------------------------------===//

/// Converts mockturtle network nodes back to MLIR operations.
/// Base class for network-specific importers.
/// Template parameter N is the expected number of fanins per gate.
template <typename Ntk, size_t N = 4>
class NetworkImporter {
public:
  using signal = typename Ntk::signal;
  using node = typename Ntk::node;

  NetworkImporter(const ExportedState &state, Ntk &ntk, OpBuilder &builder)
      : state(state), ntk(ntk), builder(builder) {}

  virtual ~NetworkImporter() = default;

  /// Import the network back to MLIR and replace original values
  void run() {
    SmallVector<Value> outputs;
    ntk.foreach_po(
        [&](auto const &f, auto i) { outputs.push_back(lowerSignal(f)); });

    for (auto [idx, value] : state.outputToValue) {
      value.replaceAllUsesWith(outputs[idx]);
    }

    // Run DCE
    mlir::PatternRewriter rewriter(builder.getContext());
    (void)mlir::runRegionDCE(rewriter,
                             builder.getBlock()->getParentOp()->getRegions());
  }

protected:
  /// Convert a network signal to an MLIR value.
  /// A signal represents a node output with potential complementation
  /// (inversion). This method must handle the signal's complement flag and
  /// return the corresponding MLIR value, potentially creating inverted
  /// operations.
  virtual Value lowerSignal(signal s) = 0;

  /// Convert a network gate node to an MLIR operation.
  /// Creates the appropriate MLIR operation (e.g., AndInverterOp,
  /// MajorityInverterOp, or hw::InstanceOp) for the given network node.
  virtual Operation *lowerGate(node n) = 0;

  /// Handle signal complementation (inversion).
  /// If isComplement is true, creates an inverted version of the value.
  /// For standard networks, this typically creates an inverted gate operation.
  /// For cell-view networks, complementation should not occur because that
  /// would mean inversion is not mapped to a cell.
  virtual Value lowerComplement(Value v, bool isComplement) = 0;

  /// Get the primary input index for a node.
  /// Maps a network PI node to its corresponding index in the exported state.
  /// This is used to retrieve the original MLIR value for primary inputs.
  virtual size_t getPIIndex(node n) = 0;

  /// Convert a network node to an MLIR operation or value.
  /// If the node represents a gate, it could have multiple outputs like MLIR's
  /// operations.
  llvm::PointerUnion<Operation *, Value> lowerNode(node n) {
    if (ntk.is_pi(n))
      return state.indexToInput.lookup(getPIIndex(n));

    if (ntk.is_constant(n)) {
      auto value = ntk.constant_value(n);
      // Create constant if not already created.
      if (!constants[value])
        constants[value] = builder.create<hw::ConstantOp>(
            builder.getUnknownLoc(), builder.getI1Type(), value);
      return constants[value];
    }

    // Map node to operation
    auto it = nodeToOpMap.find(n);
    if (it != nodeToOpMap.end())
      return it->second;

    it = nodeToOpMap.insert({n, lowerGate(n)}).first;
    return it->second;
  }

  /// Get the operands (fanins) of a node as MLIR values
  SmallVector<Value, N> getOperands(node n) {
    SmallVector<Value, N> results;
    ntk.foreach_fanin(
        n, [&](auto const &f) { results.push_back(lowerSignal(f)); });
    return results;
  }

  const ExportedState &state;
  Ntk &ntk;
  OpBuilder &builder;

  // Mapping from network nodes to their corresponding MLIR operations.
  DenseMap<node, Operation *> nodeToOpMap;

  Value constants[2] = {};
};

//===----------------------------------------------------------------------===//
// Standard Network Converters (AIG, MIG)
//===----------------------------------------------------------------------===//

/// Importer for standard logic networks (AIG, MIG).
/// Converts network gates to their corresponding MLIR operations.
template <typename Ntk, typename Op, size_t N>
class StandardNetworkConverter : public NetworkImporter<Ntk, N> {
public:
  using NetworkImporter<Ntk, N>::NetworkImporter;
  using node = typename Ntk::node;
  using signal = typename Ntk::signal;

protected:
  /// Create an MLIR operation for a network gate
  Operation *lowerGate(node n) override {
    auto children = this->getOperands(n);
    SmallVector<bool> isComplement(children.size(), false);
    return this->builder.template create<Op>(this->builder.getUnknownLoc(),
                                             children, isComplement);
  }

  /// Handle signal inversion by creating a complemented operation
  Value lowerComplement(Value v, bool isComplement) override {
    if (!isComplement)
      return v;
    return this->builder.template create<Op>(v.getLoc(), v, true);
  }

  /// Get the primary input index for a node
  size_t getPIIndex(node n) override { return this->ntk.pi_index(n); }

  /// Convert a network signal to MLIR value with complement handling
  Value lowerSignal(signal s) override {
    auto v = this->lowerNode(s.index);
    if (auto val = dyn_cast<Value>(v))
      return this->lowerComplement(val, s.complement);

    auto *op = cast<Operation *>(v);
    return this->lowerComplement(op->getResult(0), s.complement);
  }
};

using AIGNetworkConverter =
    StandardNetworkConverter<mockturtle::aig_network, synth::aig::AndInverterOp,
                             2>;
using MIGNetworkConverter =
    StandardNetworkConverter<mockturtle::mig_network,
                             synth::mig::MajorityInverterOp, 3>;

//===----------------------------------------------------------------------===//
// Cell View Converter (for technology mapping)
//===----------------------------------------------------------------------===//

/// Importer for cell-view networks (technology mapped circuits).
/// Converts technology-mapped gates to HW module instances.
class CellViewConverter
    : public NetworkImporter<mockturtle::cell_view<mockturtle::block_network>,
                             4> {
public:
  using Ntk = mockturtle::cell_view<mockturtle::block_network>;
  using node = typename Ntk::node;
  using signal = typename Ntk::signal;

  CellViewConverter(const ExportedState &state, Ntk &ntk, OpBuilder &builder,
                    const mlir::SymbolTable &symbolTable)
      : NetworkImporter(state, ntk, builder), symbolTable(symbolTable) {
    // Build PI index mapping since cell_view doesn't provide pi_index().
    ntk.foreach_pi([&](auto const &n, auto i) { nodeToPIIndex[n] = i; });
  }

protected:
  /// Create a hardware instance for a technology-mapped cell
  Operation *lowerGate(node n) override {
    assert(ntk.has_cell(n) && "Expected cell-mapped node");
    auto operands = getOperands(n);
    auto cell = ntk.get_cell(n);
    auto *moduleOp = symbolTable.lookup(cell.name);
    assert(moduleOp && "Cell module must have been defined");

    return builder.create<hw::InstanceOp>(
        builder.getUnknownLoc(), moduleOp,
        cell.name + "_" + std::to_string(cell.id), operands);
  }

  /// Convert a cell output signal to MLIR value (no complement support)
  Value lowerSignal(signal s) override {
    auto v = lowerNode(s.index);
    assert(!s.complement && "All cells should be non-inverted");

    if (auto val = dyn_cast<Value>(v))
      return val;

    auto *op = cast<Operation *>(v);
    assert(s.output < op->getNumResults() && "Output index out of range");
    return op->getResult(s.output);
  }

  /// Cells should never be inverted in technology mapping
  Value lowerComplement(Value v, bool isComplement) override {
    llvm::report_fatal_error("All cells should be non-inverted");
    return v;
  }

  /// Get the primary input index using pre-built mapping
  size_t getPIIndex(node n) override {
    assert(ntk.is_pi(n) && "Expected PI node");
    return nodeToPIIndex.at(n);
  }

private:
  const mlir::SymbolTable &symbolTable;
  DenseMap<node, size_t> nodeToPIIndex;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public Interface Implementations
//===----------------------------------------------------------------------===//

/// Export MLIR operations to AIG network, apply technology mapping, and
/// import the result back as hardware instances.
llvm::LogicalResult
circt::synth::mockturtle_integration::runAIGNetworkToCellView(
    const mlir::SymbolTable &symbolTable, mlir::Block *block,
    llvm::function_ref<
        mlir::FailureOr<mockturtle::cell_view<mockturtle::block_network>>(
            mockturtle::aig_network &)>
        mapFunction) {
  ExportedState state;
  mockturtle::aig_network ntk;
  AIGExporter exporter(block, ntk, state);

  if (failed(exporter.run()))
    return failure();

  auto result = mapFunction(ntk);
  if (failed(result))
    return failure();

  auto &cv = *result;
  LLVM_DEBUG(llvm::dbgs() << "Converted network has " << cv.num_gates()
                          << " gates, " << cv.num_pos() << " POs, and "
                          << cv.num_pis() << " PIs.\n");

  OpBuilder builder = OpBuilder::atBlockBegin(block);
  CellViewConverter converter(state, cv, builder, symbolTable);
  converter.run();

  return success();
}

/// Export MLIR AIG operations to mockturtle AIG network, apply transformations,
/// and import the result back to MLIR.
llvm::LogicalResult
circt::synth::mockturtle_integration::runAIGNetworkTransforms(
    mlir::Block *block,
    llvm::function_ref<llvm::LogicalResult(mockturtle::aig_network &)>
        transform) {
  ExportedState state;
  mockturtle::aig_network ntk;
  AIGExporter exporter(block, ntk, state);

  if (failed(exporter.run()))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Exported AIG network has " << ntk.num_gates()
                          << " AND gates, " << ntk.num_pos() << " POs, and "
                          << ntk.num_pis() << " PIs.\n");

  if (failed(transform(ntk)))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Transformed AIG network has " << ntk.num_gates()
                          << " AND gates, " << ntk.num_pos() << " POs, and "
                          << ntk.num_pis() << " PIs.\n");

  OpBuilder builder = OpBuilder::atBlockBegin(block);
  AIGNetworkConverter converter(state, ntk, builder);
  converter.run();

  return success();
}

/// Export MLIR MIG operations to mockturtle MIG network, apply transformations,
/// and import the result back to MLIR.
llvm::LogicalResult
circt::synth::mockturtle_integration::runMIGNetworkTransforms(
    mlir::Block *block,
    llvm::function_ref<llvm::LogicalResult(mockturtle::mig_network &)>
        transform) {
  ExportedState state;
  mockturtle::mig_network ntk;
  MIGExporter exporter(block, ntk, state);

  if (failed(exporter.run()))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Exported MIG network has " << ntk.num_gates()
                          << " gates, " << ntk.num_pos() << " POs, and "
                          << ntk.num_pis() << " PIs.\n");

  if (failed(transform(ntk)))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Transformed MIG network has " << ntk.num_gates()
                          << " gates, " << ntk.num_pos() << " POs, and "
                          << ntk.num_pis() << " PIs.\n");

  OpBuilder builder = OpBuilder::atBlockBegin(block);
  MIGNetworkConverter converter(state, ntk, builder);
  converter.run();

  return success();
}

/// Auto-detect network type (AIG/MIG) and apply transformations.
/// Handles blocks containing either or both network types.
llvm::LogicalResult circt::synth::mockturtle_integration::runNetworkTransforms(
    mlir::Block *block,
    llvm::function_ref<llvm::LogicalResult(Ntk &)> transform) {
  bool existMig = false, existAig = false;
  for (auto &op : block->getOperations()) {
    if (isa<synth::mig::MajorityInverterOp>(op))
      existMig = true;
    else if (isa<synth::aig::AndInverterOp>(op))
      existAig = true;
    if (existMig && existAig)
      break;
  }

  auto fn = [&](auto &ntk) {
    Ntk ntkVariant(&ntk);
    return transform(ntkVariant);
  };

  if (existMig && failed(runMIGNetworkTransforms(block, fn)))
    return failure();

  if (existAig && failed(runAIGNetworkTransforms(block, fn)))
    return failure();

  return success();
}