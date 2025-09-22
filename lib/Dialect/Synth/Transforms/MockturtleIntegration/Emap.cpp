//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Emap pass using mockturtle's technology mapping.
// This pass performs technology mapping to standard cells using the emap
// algorithm from the mockturtle library.
//
//===----------------------------------------------------------------------===//

#include "mockturtle/algorithms/emap.hpp"
#include "NetworkConversion.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mockturtle/io/genlib_reader.hpp"
#include "mockturtle/networks/aig.hpp"
#include "mockturtle/networks/block.hpp"
#include "mockturtle/networks/mig.hpp"
#include "mockturtle/views/cell_view.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <fstream>
#include <lorina/genlib.hpp>
#include <memory>

#define DEBUG_TYPE "synth-mockturtle-emap"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_MOCKTURTLEEMAP
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::mockturtle_integration;

namespace {

/// Emap pass using mockturtle's technology mapping algorithm.
struct MockturtleEmapPass
    : public impl::MockturtleEmapBase<MockturtleEmapPass> {
  using impl::MockturtleEmapBase<MockturtleEmapPass>::MockturtleEmapBase;
  static std::unique_ptr<mockturtle::tech_library<9>>
  parseTechnologyLibrary(StringRef path);
  LogicalResult
  importTechnologyLibrary(const mockturtle::tech_library<9> &techLib);
  LogicalResult
  runTechnologyMapping(const mockturtle::tech_library<9> &techLib);
  void runOnOperation() override;
};

} // namespace

/// Parse technology library from genlib file.
std::unique_ptr<mockturtle::tech_library<9>>
MockturtleEmapPass::parseTechnologyLibrary(StringRef genlibPath) {
  // Configure technology library parameters
  mockturtle::tech_library_params tps;
  tps.ignore_symmetries = true; // Speed up mapping with minor delay increase
  tps.verbose = false;

  // Read gate library from genlib file
  std::vector<mockturtle::gate> gates;
  std::ifstream in(genlibPath.str());
  if (!in.is_open() ||
      lorina::read_genlib(in, mockturtle::genlib_reader(gates)) !=
          lorina::return_code::success)
    return nullptr;

  // Create and return technology library
  return std::make_unique<mockturtle::tech_library<9>>(gates, tps);
}

LogicalResult MockturtleEmapPass::importTechnologyLibrary(
    const mockturtle::tech_library<9> &techLib) {
  auto module = getOperation();
  OpBuilder builder(module);

  for (auto &cell : techLib.get_cells()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Gate: " << cell.name << ", area: " << cell.area << "\n");
    auto name = builder.getStringAttr(cell.name);

    // Build port lists for the cell
    SmallVector<hw::PortInfo> outputPorts;
    llvm::MapVector<StringAttr, hw::PortInfo> inputPorts;

    for (auto &gate : cell.gates) {
      LLVM_DEBUG(llvm::dbgs() << "  Output: " << gate.output_name
                              << ", expr: " << gate.expression << "\n");

      // Add output port
      hw::PortInfo port;
      port.name = builder.getStringAttr(gate.output_name);
      port.dir = hw::PortInfo::Output;
      port.type = builder.getIntegerType(1);
      outputPorts.push_back(port);

      // Collect unique input pins
      for (const auto &p : gate.pins) {
        LLVM_DEBUG(llvm::dbgs() << "    Pin: " << p.name
                                << ", phase: " << (int)p.phase << "\n");
        hw::PortInfo inputPort;
        inputPort.name = builder.getStringAttr(p.name);
        if (inputPorts.find(inputPort.name) != inputPorts.end())
          continue;
        inputPort.dir = hw::PortInfo::Input;
        inputPort.type = builder.getIntegerType(1);
        inputPort.argNum = inputPorts.size();
        inputPorts[inputPort.name] = inputPort;
      }
    }

    // Combine input and output ports
    SmallVector<hw::PortInfo> ports;
    for (auto &it : inputPorts)
      ports.push_back(it.second);
    ports.append(outputPorts.begin(), outputPorts.end());

    // Create HW module for the cell
    builder.setInsertionPointToStart(module.getBody(0));
    auto op = builder.create<hw::HWModuleOp>(module.getLoc(), name, ports);
    OpBuilder::InsertionGuard guard(builder);
    auto *body = op.getBodyBlock();
    builder.setInsertionPointToStart(body);

    // Generate truth table operations for each output
    SmallVector<Value> outputs;
    for (auto &gate : cell.gates) {
      // Extract truth table from gate function
      SmallVector<bool> func;
      APInt bits(1 << gate.num_vars, gate.function._bits);
      for (auto i = 0u; i < (1u << gate.num_vars); i++)
        func.push_back(bits[i]);

      // Map gate inputs to module arguments
      SmallVector<Value> inputs;
      for (auto &p : llvm::reverse(gate.pins)) {
        auto pinName = builder.getStringAttr(p.name);
        auto arg = body->getArgument(inputPorts[pinName].argNum);
        inputs.push_back(arg);
      }

      // Create truth table operation
      auto tt = builder.create<comb::TruthTableOp>(
          module.getLoc(), inputs, builder.getBoolArrayAttr(func));
      outputs.push_back(tt.getResult());
    }

    // Connect outputs to module terminator
    op.getBodyBlock()->getTerminator()->setOperands(outputs);
  }

  return success();
}

LogicalResult MockturtleEmapPass::runTechnologyMapping(
    const mockturtle::tech_library<9> &techLib) {
  auto module = getOperation();

  // Configure emap parameters
  mockturtle::emap_params ps;
  ps.matching_mode = mockturtle::emap_params::hybrid;
  ps.area_oriented_mapping = false;
  ps.map_multioutput = true;
  ps.relax_required = 0;

  // Technology mapping function
  auto mapToStandardCells = [&](mockturtle::aig_network &ntk)
      -> FailureOr<mockturtle::cell_view<mockturtle::block_network>> {
    return mockturtle::emap<9>(ntk, techLib, ps);
  };

  // Apply technology mapping to all modules in parallel
  auto &symbolTable = getAnalysis<SymbolTable>();
  auto result = failableParallelForEach(
      &getContext(), module.getOps<hw::HWModuleOp>(),
      [&](hw::HWModuleOp hwModule) -> LogicalResult {
        return mockturtle_integration::runAIGNetworkToCellView(
            symbolTable, hwModule.getBodyBlock(), mapToStandardCells);
      });

  return result;
}

void MockturtleEmapPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Running Mockturtle Emap pass\n");

  // Parse technology library from genlib file
  auto techLib = parseTechnologyLibrary(genlibPath.getValue());
  if (!techLib)
    return signalPassFailure();

  // Import technology library as HW module declarations
  if (failed(importTechnologyLibrary(*techLib)))
    return signalPassFailure();

  // Run technology mapping
  if (failed(runTechnologyMapping(*techLib)))
    return signalPassFailure();
}
