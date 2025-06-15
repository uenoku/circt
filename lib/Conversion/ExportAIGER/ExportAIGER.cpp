//===- ExportAIGER.cpp - AIGER file export --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AIGER file export functionality.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportAIGER.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/Seq/SeqOpInterfaces.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Version.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/IR/Operation.h>
#include <string>

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::aig;
using namespace circt::seq;

#define DEBUG_TYPE "export-aiger"

namespace {

/// Main AIGER exporter class
class AIGERExporter {
public:
  AIGERExporter(hw::HWModuleOp module, llvm::raw_ostream &os,
                const ExportAIGEROptions &options)
      : module(module), os(os), options(options) {}

  /// Export the module to AIGER format
  LogicalResult exportModule();

  using Object = std::pair<Value, int>;

private:
  hw::HWModuleOp module;
  llvm::raw_ostream &os;
  const ExportAIGEROptions &options;

  // AIGER file data
  unsigned maxVarIndex = 0;
  unsigned getNumInputs() { return inputs.size(); }
  unsigned getNumLatches() { return latches.size(); }
  unsigned getNumOutputs() { return outputs.size(); }
  unsigned getNumAnds() { return andGates.size(); }

  // Data structures for tracking variables and gates
  // <Value, bitPos>
  DenseMap<Object, unsigned> valueLiteralMap;
  llvm::EquivalenceClasses<Object> valueEquivalence;
  SmallVector<std::pair<Object, StringAttr>> inputs;
  SmallVector<std::pair<std::pair<Object, StringAttr>, Object>>
      latches; // current, next
  SmallVector<std::pair<Object, StringAttr>> outputs;
  SmallVector<std::tuple<Object, Object, Object>> andGates; // lhs, rhs0, rhs1

  std::optional<Value> clock;

  /// Analyze the module and collect information
  LogicalResult analyzeModule();

  StringAttr getIndexName(StringAttr name, int bitPos) {
    if (!options.includeSymbolTable)
      return {};
    if (!name || bitPos == -1)
      return name;
    return StringAttr::get(name.getContext(),
                           name.getValue() + "_" + std::to_string(bitPos));
  }

  void addInput(Object obj, StringAttr name = {}, int bitPos = -1) {
    inputs.push_back({obj, getIndexName(name, bitPos)});
  }
  void addLatch(Object current, Object next, StringAttr name = {},
                int bitPos = -1) {
    latches.push_back({{current, getIndexName(name, bitPos)}, next});
  }
  void addOutput(Object obj, StringAttr name = {}, int bitPos = -1) {
    outputs.push_back({obj, getIndexName(name, bitPos)});
  }

  /// Analyze module ports (inputs/outputs)
  LogicalResult analyzePorts(hw::HWModuleOp module);

  /// Analyze operations in the module
  LogicalResult analyzeOperations(hw::HWModuleOp module);

  /// Assign literals to all values
  LogicalResult assignLiterals();

  /// Write the AIGER header
  LogicalResult writeHeader();

  /// Write inputs section
  LogicalResult writeInputs();

  /// Write latches section
  LogicalResult writeLatches();

  /// Write outputs section
  LogicalResult writeOutputs();

  /// Write AND gates section
  LogicalResult writeAndGates();

  /// Write symbol table
  LogicalResult writeSymbolTable();

  /// Write comments
  LogicalResult writeComments();

  /// Get or assign a literal for a value
  unsigned getLiteral(Object obj, bool inverted = false);

  /// Emit error message
  InFlightDiagnostic emitError(const Twine &message) {
    return mlir::emitError(module.getLoc(), message);
  }

  /// Helper method to write unsigned LEB128 encoded integers
  void writeUnsignedLEB128(unsigned value);
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// AIGERExporter Implementation
//===----------------------------------------------------------------------===//

LogicalResult AIGERExporter::exportModule() {
  LLVM_DEBUG(llvm::dbgs() << "Starting AIGER export\n");

  if (failed(analyzeModule()) || failed(writeHeader()) ||
      failed(writeInputs()) || failed(writeLatches()) ||
      failed(writeOutputs()) || failed(writeAndGates()) ||
      failed(writeSymbolTable()) || failed(writeComments()))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "AIGER export completed successfully\n");
  return success();
}

LogicalResult AIGERExporter::analyzeModule() {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing module\n");

  auto topModule = module;
  LLVM_DEBUG(llvm::dbgs() << "Found top module: " << topModule.getModuleName()
                          << "\n");

  // Analyze module ports
  if (failed(analyzePorts(topModule)))
    return failure();

  // Walk through all operations in the module body
  if (failed(analyzeOperations(topModule)))
    return failure();

  // Assign literals to all values
  if (failed(assignLiterals()))
    return failure();

  // Calculate final statistics
  maxVarIndex = (valueLiteralMap.size() > 0) ? valueLiteralMap.size() : 0;

  LLVM_DEBUG(llvm::dbgs() << "Analysis complete: M=" << maxVarIndex
                          << " I=" << getNumInputs() << " L=" << getNumLatches()
                          << " O=" << getNumOutputs() << " A=" << getNumAnds()
                          << "\n");

  return success();
}

LogicalResult AIGERExporter::writeHeader() {
  LLVM_DEBUG(llvm::dbgs() << "Writing AIGER header\n");

  // Write format identifier
  if (options.binaryFormat)
    os << "aig";
  else
    os << "aag";

  // Write M I L O A
  os << " " << maxVarIndex << " " << getNumInputs() << " " << getNumLatches()
     << " " << getNumOutputs() << " " << getNumAnds() << "\n";

  return success();
}

LogicalResult AIGERExporter::writeInputs() {
  LLVM_DEBUG(llvm::dbgs() << "Writing inputs\n");

  if (options.binaryFormat) {
    // In binary format, inputs are implicit
    return success();
  }

  // Write input literals
  for (auto [input, name] : inputs) {
    unsigned literal = getLiteral(input);
    os << literal << "\n";
  }

  return success();
}

LogicalResult AIGERExporter::writeLatches() {
  LLVM_DEBUG(llvm::dbgs() << "Writing latches\n");

  // Write latch definitions
  for (auto [current, next] : latches) {
    auto [currentObj, currentName] = current;
    unsigned currentLiteral = getLiteral(currentObj);

    // For next state, we need to handle potential inversions
    // Check if next state comes from an inverter or has inversions
    unsigned nextLiteral = getLiteral(next);

    if (options.binaryFormat) {
      os << nextLiteral << "\n";
    } else {
      os << currentLiteral << " " << nextLiteral << "\n";
    }
  }

  return success();
}

LogicalResult AIGERExporter::writeOutputs() {
  LLVM_DEBUG(llvm::dbgs() << "Writing outputs\n");

  // Write output literals
  for (auto [output, name] : outputs) {
    unsigned literal = getLiteral(output);
    os << literal << "\n";
  }

  return success();
}

LogicalResult AIGERExporter::writeAndGates() {
  LLVM_DEBUG(llvm::dbgs() << "Writing AND gates\n");

  if (options.binaryFormat) {
    // Implement binary format encoding for AND gates
    for (auto [lhs, rhs0, rhs1] : andGates) {
      unsigned lhsLiteral = getLiteral(lhs);

      // Get the AND-inverter operation to check inversion flags
      auto andInvOp = lhs.first.getDefiningOp<aig::AndInverterOp>();
      if (!andInvOp || andInvOp.getInputs().size() != 2) {
        return emitError("expected 2-input AND-inverter operation");
      }

      // Get operand literals with inversion
      bool rhs0Inverted = andInvOp.getInverted()[0];
      bool rhs1Inverted = andInvOp.getInverted()[1];

      unsigned rhs0Literal = getLiteral(rhs0, rhs0Inverted);
      unsigned rhs1Literal = getLiteral(rhs1, rhs1Inverted);

      // Ensure rhs0 >= rhs1 as required by AIGER format
      if (rhs0Literal < rhs1Literal) {
        std::swap(rhs0Literal, rhs1Literal);
        std::swap(rhs0, rhs1);
        std::swap(rhs0Inverted, rhs1Inverted);
      }

      // In binary format, we need to write the delta values
      // Delta0 = lhs - rhs0
      // Delta1 = rhs0 - rhs1
      unsigned delta0 = lhsLiteral - rhs0Literal;
      unsigned delta1 = rhs0Literal - rhs1Literal;

      if (lhsLiteral < rhs0Literal) {
        llvm::dbgs() << "lhs: " << lhs.first << "[" << lhs.second << "]\n";
        llvm::dbgs() << "rhs0: " << rhs0.first << "[" << rhs0.second << "]\n";
        llvm::dbgs() << "rhs1: " << rhs1.first << "[" << rhs1.second << "]\n";
        llvm::dbgs() << "Writing AND gate: " << lhsLiteral << " = "
                     << rhs0Literal << " & " << rhs1Literal
                     << " (deltas: " << delta0 << ", " << delta1 << ")\n";
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "Writing AND gate: " << lhsLiteral << " = " << rhs0Literal
                 << " & " << rhs1Literal << " (deltas: " << delta0 << ", "
                 << delta1 << ")\n");
      assert(lhsLiteral >= rhs0Literal && "lhsLiteral >= rhs0Literal");
      assert(rhs0Literal >= rhs1Literal && "rhs0Literal >= rhs1Literal");

      // Write deltas using variable-length encoding
      writeUnsignedLEB128(delta0);
      writeUnsignedLEB128(delta1);
    }
  } else {
    // ASCII format - no need for structural hashing
    for (auto [lhs, rhs0, rhs1] : andGates) {
      unsigned lhsLiteral = getLiteral(lhs);

      // Get the AND-inverter operation to check inversion flags
      auto andInvOp = lhs.first.getDefiningOp<aig::AndInverterOp>();
      if (!andInvOp || andInvOp.getInputs().size() != 2) {
        return emitError("expected 2-input AND-inverter operation");
      }

      // Get operand literals with inversion
      bool rhs0Inverted = andInvOp.getInverted()[0];
      bool rhs1Inverted = andInvOp.getInverted()[1];

      unsigned rhs0Literal = getLiteral(rhs0, rhs0Inverted);
      unsigned rhs1Literal = getLiteral(rhs1, rhs1Inverted);

      os << lhsLiteral << " " << rhs0Literal << " " << rhs1Literal << "\n";
    }
  }

  return success();
}

LogicalResult AIGERExporter::writeSymbolTable() {
  LLVM_DEBUG(llvm::dbgs() << "Writing symbol table\n");

  if (!options.includeSymbolTable)
    return success();

  for (auto [index, elem] : llvm::enumerate(inputs)) {
    auto [obj, name] = elem;
    if (!name)
      continue;
    os << "i" << index << " " << name.getValue() << "\n";
  }

  for (auto [index, elem] : llvm::enumerate(latches)) {
    auto [current, next] = elem;
    if (!current.second)
      continue;
    os << "l" << index << " " << current.second.getValue() << "\n";
  }

  for (auto [index, elem] : llvm::enumerate(outputs)) {
    auto [obj, name] = elem;
    if (!name)
      continue;
    os << "o" << index << " " << name.getValue() << "\n";
  }

  return success();
}

LogicalResult AIGERExporter::writeComments() {
  LLVM_DEBUG(llvm::dbgs() << "Writing comments\n");

  if (!options.includeComments)
    return success();

  // Write comment section
  os << "c\n";
  os << "Generated by " << circt::getCirctVersion() << "\n";

  return success();
}

unsigned AIGERExporter::getLiteral(Object obj, bool inverted) {
  // Handle constants
  auto value = obj.first;
  auto pos = obj.second;
  {
    auto it = valueLiteralMap.find({value, pos});
    if (it != valueLiteralMap.end()) {
      unsigned literal = it->second;
      return inverted ? literal ^ 1 : literal;
    }
  }

  if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
    APInt constValue = constOp.getValue();
    if (constValue.isZero()) {
      // FALSE constant = literal 0, inverted = literal 1 (TRUE)
      return inverted ? 1 : 0;
    }
    // TRUE constant = literal 1, inverted = literal 0 (FALSE)
    return inverted ? 0 : 1;
  }

  // Handle single-input AND-inverter (pure inverter)
  if (auto andInvOp = value.getDefiningOp<aig::AndInverterOp>()) {
    if (andInvOp.getInputs().size() == 1) {
      // This is a pure inverter - we need to find the input's literal
      Value input = andInvOp.getInputs()[0];
      bool inputInverted = andInvOp.getInverted()[0];
      unsigned inputLiteral = getLiteral({input, pos}, inputInverted);
      // Apply additional inversion if requested
      valueLiteralMap[{value, pos}] = inputLiteral;
    }
  }

  if (auto concat = value.getDefiningOp<comb::ConcatOp>()) {
    // Concatenation is handled by looking up the correct operand
    // based on the position
    int64_t bitPos = pos;
    for (auto operand : llvm::reverse(concat.getInputs())) {
      int64_t operandWidth = hw::getBitWidth(operand.getType());
      if (bitPos >= operandWidth) {
        bitPos -= operandWidth;
        continue;
      }
      valueLiteralMap[{value, pos}] = getLiteral({operand, bitPos});
      break;
    }
  }

  if (auto extract = value.getDefiningOp<comb::ExtractOp>()) {
    // Extract operation is handled by looking up the correct operand
    // based on the position
    int64_t bitPos = pos;
    valueLiteralMap[{value, pos}] =
        getLiteral({extract.getInput(), bitPos + extract.getLowBit()});
  }

  if (auto replicate = value.getDefiningOp<comb::ReplicateOp>()) {
    // Replication is handled by looking up the correct operand
    // based on the position
    int64_t bitPos = pos;
    int64_t operandWidth = hw::getBitWidth(replicate.getInput().getType());
    valueLiteralMap[{value, pos}] =
        getLiteral({replicate.getInput(), bitPos % operandWidth});
  }

  // Look up in the literal map
  auto it = valueLiteralMap.find({value, pos});
  if (it != valueLiteralMap.end()) {
    unsigned literal = it->second;
    return inverted ? literal ^ 1 : literal;
  }
  llvm::errs() << "Unhandled: Value not found in literal map: " << value << "["
               << pos << "]\n";
  assert(0 && "Value not found in literal map");

  // This should not happen if analysis was done correctly
  llvm_unreachable("Value not found in literal map");
}

LogicalResult AIGERExporter::analyzePorts(hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing module ports\n");

  auto inputNames = hwModule.getInputNames();
  auto outputNames = hwModule.getOutputNames();

  // Analyze input ports
  for (auto [index, arg] :
       llvm::enumerate(hwModule.getBodyBlock()->getArguments())) {
    // Skip clock inputs for now (they're handled separately in latches)
    if (isa<seq::ClockType>(arg.getType()))
      continue;

    // All other inputs should be i1 (1-bit assumption)
    for (int64_t i = 0; i < hw::getBitWidth(arg.getType()); ++i) {
      if (options.valueCallabck(arg, i, inputs.size())) {
        addInput({arg, i},
                 llvm::dyn_cast_or_null<StringAttr>(inputNames[index]), i);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "  Input " << index << ": " << arg << "\n");
  }

  // Analyze output ports by looking at hw.output operation
  auto *outputOp = hwModule.getBodyBlock()->getTerminator();
  for (auto [operand, name] :
       llvm::zip(outputOp->getOpOperands(), outputNames)) {
    for (int64_t i = 0; i < hw::getBitWidth(operand.get().getType()); ++i)
      if (options.operandCallback(operand, i, outputs.size())) {
        addOutput({operand.get(), i}, llvm::dyn_cast_or_null<StringAttr>(name),
                  i);
      }
    LLVM_DEBUG(llvm::dbgs() << "  Output: " << operand.get() << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << getNumInputs() << " inputs, "
                          << getNumOutputs() << " outputs from ports\n");
  return success();
}

LogicalResult AIGERExporter::analyzeOperations(hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing operations\n");
  // Sort the operations topologically.

  auto result = hwModule.walk([&](Region *region) {
    auto regionKindOp = dyn_cast<RegionKindInterface>(region->getParentOp());
    if (!regionKindOp ||
        regionKindOp.hasSSADominance(region->getRegionNumber()))
      return WalkResult::advance();

    // Graph region.
    for (auto &block : *region) {
      if (!sortTopologically(&block, [&](Value value, Operation *op) {
            return isa<seq::CompRegOp, seq::FirRegOp, hw::InstanceOp>(op);
          }))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return mlir::emitError(hwModule.getLoc(),
                           "failed to sort operations topologically");

  int randomNameIndex = 0;
  // Walk the operations.
  auto walkResult = hwModule.walk([&](Operation *op) {
    if (auto andInvOp = dyn_cast<aig::AndInverterOp>(op)) {
      // Handle AIG AND-inverter operations
      if (andInvOp.getInputs().size() == 1) {
        // Single input = inverter (not counted as AND gate in AIGER)
        LLVM_DEBUG(llvm::dbgs() << "  Found inverter: " << andInvOp << "\n");
      } else if (andInvOp.getInputs().size() == 2) {
        // Two inputs = AND gate
        for (int64_t i = 0; i < hw::getBitWidth(andInvOp.getType()); ++i) {
          andGates.push_back({{andInvOp.getResult(), i},
                              {andInvOp.getInputs()[0], i},
                              {andInvOp.getInputs()[1], i}});
        }

        LLVM_DEBUG(llvm::dbgs() << "  Found AND gate: " << andInvOp << "\n");
      } else {
        // Variadic AND gates need to be lowered first
        emitError("variadic AND gates not supported, run aig-lower-variadic "
                  "pass first");
        return WalkResult::interrupt();
      }
      options.notifyEmitted(andInvOp);
      return WalkResult::advance();
    }

    if (auto regOp = dyn_cast<seq::CompRegOp>(op)) {
      if (!regOp.getName()) {
        regOp.setName(StringAttr::get(op->getContext(),
                                      "unamed_reg_generated_for_aiger" +
                                          std::to_string(randomNameIndex++)));
      }
      // Handle registers (latches in AIGER)
      for (int64_t i = 0; i < hw::getBitWidth(regOp.getType()); ++i) {
        addLatch({regOp.getResult(), i}, {regOp.getInput(), i},
                 regOp.getNameAttr(), i);
      }
      LLVM_DEBUG(llvm::dbgs() << "  Found latch: " << regOp << "\n");
      options.notifyEmitted(regOp);
      return WalkResult::advance();
    }
    if (auto regOp = dyn_cast<seq::FirRegOp>(op)) {
      if (regOp.getNameAttr().getValue().empty()) {
        regOp.setName(StringAttr::get(op->getContext(),
                                      "unamed_reg_generated_for_aiger" +
                                          std::to_string(randomNameIndex++)));
      }
      // Handle registers (latches in AIGER)
      for (int64_t i = 0; i < hw::getBitWidth(regOp.getType()); ++i) {
        addLatch({regOp.getResult(), i}, {regOp.getNext(), i},
                 regOp.getNameAttr(), i);
      }
      LLVM_DEBUG(llvm::dbgs() << "  Found latch: " << regOp << "\n");
      options.notifyEmitted(regOp);
      return WalkResult::advance();
    }

    if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
      size_t pos = 0;
      for (auto operand : llvm::reverse(concat.getInputs())) {
        for (size_t i = 0; i < hw::getBitWidth(operand.getType()); ++i) {
          valueEquivalence.unionSets({operand, i}, {concat.getResult(), pos++});
        }
      }
    }
    if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
      for (size_t i = 0; i < hw::getBitWidth(extract.getType()); ++i) {
        valueEquivalence.unionSets(
            {extract.getInput(), extract.getLowBit() + i},
            {extract.getResult(), i});
      }
    }

    if (auto replicate = dyn_cast<comb::ReplicateOp>(op)) {
      for (size_t i = 0; i < hw::getBitWidth(replicate.getType()); ++i) {
        valueEquivalence.unionSets(
            {replicate.getInput(), i % hw::getBitWidth(replicate.getType())},
            {replicate.getResult(), i});
      }
    }

    if (isa<hw::HWModuleOp, hw::ConstantOp, hw::OutputOp, comb::ConcatOp,
            comb::ExtractOp, comb::ReplicateOp>(op))
      return WalkResult::advance();

    // Handle unknown operations
    if (options.handleUnknownOperation) {
      assert(options.operandCallback &&
             "unknownOperationOperandHandler must be set if "
             "handleUnknownOperation is true");
      assert(options.valueCallabck &&
             "unknownOperationResultHandler must be set if "
             "handleUnknownOperation is true");

      for (mlir::OpOperand &operand : op->getOpOperands()) {
        for (int64_t i = 0; i < hw::getBitWidth(operand.get().getType()); ++i) {
          if (options.operandCallback(operand, i, outputs.size()))
            addOutput({operand.get(), i});
        }
      }

      for (mlir::OpResult result : op->getOpResults()) {
        for (int64_t i = 0; i < hw::getBitWidth(result.getType()); ++i) {
          if (options.valueCallabck(result, i, inputs.size()))
            addInput({result, i});
          else {
            // Treat it as a constant
            valueLiteralMap[{result, i}] = 0;
          }
        }
      }

      return WalkResult::advance();
    }

    // Ignore other operations (hw.output, etc.)
    mlir::emitError(op->getLoc(), "unhandled operation: ") << *op;
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Found " << getNumAnds() << " AND gates, "
                          << getNumLatches() << " latches\n");
  return success();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const AIGERExporter::Object &obj) {
  return os << obj.first << "[" << obj.second << "]";
}
llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const std::pair<AIGERExporter::Object, StringAttr> &obj) {
  return os << obj.first
            << (obj.second ? " (" + obj.second.getValue() + ")" : "");
}

LogicalResult AIGERExporter::assignLiterals() {
  LLVM_DEBUG(llvm::dbgs() << "Assigning literals\n");

  unsigned nextLiteral =
      2; // Start from 2 (literal 0 = FALSE, literal 1 = TRUE)

  // Assign literals to inputs first
  for (auto input : inputs) {
    valueLiteralMap[input.first] = nextLiteral;
    LLVM_DEBUG(llvm::dbgs()
               << "  Input literal " << nextLiteral << ": " << input << "\n");
    nextLiteral += 2; // Even literals only (odd = inverted)
  }

  // Assign literals to latches (current state)
  for (auto [current, next] : latches) {
    valueLiteralMap[current.first] = nextLiteral;
    LLVM_DEBUG(llvm::dbgs()
               << "  Latch literal " << nextLiteral << ": " << current << "\n");
    nextLiteral += 2;
    auto clocked = current.first.first.getDefiningOp<seq::Clocked>();
    assert(clocked);
    if (!clock)
      clock = clocked.getClk();
    else if (clock != clocked.getClk())
      return emitError("multiple clocks found in the module");
  }

  if (clock)
    options.notifyClock(*clock);

  if (false && options.binaryFormat) {
    // Topo-sort the AND gates.
    llvm::MapVector<Object, std::pair<Object, Object>> graph;
    for (auto [lhs, rhs0, rhs1] : andGates) {
      assert(valueEquivalence.getOrInsertLeaderValue(lhs) == lhs);
      graph[lhs] = {valueEquivalence.getOrInsertLeaderValue(rhs0),
                    valueEquivalence.getOrInsertLeaderValue(rhs1)};
    }

    // Topo-sort the graph.
    SmallVector<Object> sortedAndGates;
    DenseSet<Object> visited;
    DenseSet<Object> temp; // For cycle detection

    // Helper function for DFS-based topological sort
    std::function<LogicalResult(const Object &)> visit =
        [&](const Object &obj) {
          // Skip if already visited
          if (visited.contains(obj))
            return success();

          // Check for cycles
          if (!temp.insert(obj).second)
            return failure();

          // Visit dependencies first
          auto it = graph.find(obj);
          if (it != graph.end()) {
            auto [rhs0, rhs1] = it->second;
            if (failed(visit(rhs0)) || failed(visit(rhs1)))
              return failure();
          }

          temp.erase(obj);
          visited.insert(obj);
          if (it != graph.end())
            sortedAndGates.push_back(obj);
          return success();
        };

    // Visit all nodes in the graph
    for (auto &[lhs, _] : graph) {
      if (!visited.contains(lhs)) {
        if (failed(visit(lhs)))
          return emitError("cycle detected in AND gate network");
      }
    }

    // Assign literals in topological order
    assert(sortedAndGates.size() == andGates.size());
    for (auto [index, lhs] : llvm::enumerate(sortedAndGates)) {
      auto [rhs0, rhs1] = graph[lhs];
      andGates[index] = {lhs, rhs0, rhs1};
      valueLiteralMap[lhs] = nextLiteral;
      // if (rhs0.first.getDefiningOp<aig::AndInverterOp>()) {
      //   if (!valueLiteralMap.count(rhs0)) {
      //     llvm::errs() << "lhs: " << lhs << "\n";
      //     llvm::errs() << "rhs0 not found: " << rhs0 << "\n";
      //   }
      //   assert(valueLiteralMap.count(rhs0));
      // }
      // if (rhs1.first.getDefiningOp<aig::AndInverterOp>()) {
      //   if (!valueLiteralMap.count(rhs1)) {
      //     llvm::errs() << "lhs: " << lhs << "\n";
      //     llvm::errs() << "rhs1 not found: " << rhs1 << "\n";
      //   }
      //   assert(valueLiteralMap.count(rhs1));
      // }
      // assert(valueLiteralMap.count(rhs1));
      nextLiteral += 2;
    }
  } else {
    // ASCII format - no need for topological sort
    for (auto [lhs, rhs0, rhs1] : andGates) {
      valueLiteralMap[lhs] = nextLiteral;
      LLVM_DEBUG(llvm::dbgs() << "  AND gate literal " << nextLiteral << ": "
                              << lhs << "\n");
      nextLiteral += 2;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Assigned " << valueLiteralMap.size()
                          << " literals\n");
  return success();
}

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

LogicalResult circt::exportAIGER(hw::HWModuleOp module, llvm::raw_ostream &os,
                                 const ExportAIGEROptions *options) {
  ExportAIGEROptions defaultOptions;
  if (!options)
    options = &defaultOptions;

  AIGERExporter exporter(module, os, *options);
  return exporter.exportModule();
}

llvm::cl::opt<bool>
    emitTextFormat("emit-text-format",
                   llvm::cl::desc("Export AIGER in text format"),
                   llvm::cl::init(false));
llvm::cl::opt<bool>
    includeSymbolTable("exclude-symbol-table",
                       llvm::cl::desc("Exclude symbol table from the output"),
                       llvm::cl::init(false));

void circt::registerToAIGERTranslation() {
  static mlir::TranslateFromMLIRRegistration toAIGER(
      "export-aiger", "Export AIG to AIGER format",
      [](mlir::ModuleOp module, llvm::raw_ostream &os) {
        auto ops = module.getOps<hw::HWModuleOp>();
        if (ops.empty()) {
          module.emitError("no HW module found in the input");
          return failure();
        }
        if (std::next(ops.begin()) != ops.end()) {
          module.emitError(
              "multiple HW modules found, expected single top module");
          return failure();
        }

        ExportAIGEROptions options;
        options.binaryFormat = !emitTextFormat;
        options.includeSymbolTable = !includeSymbolTable;

        return exportAIGER(*ops.begin(), os, &options);
      },
      [](DialectRegistry &registry) {
        registry.insert<aig::AIGDialect, hw::HWDialect, seq::SeqDialect,
                        comb::CombDialect>();
      });
}

// Helper method to write unsigned LEB128 encoded integers
void AIGERExporter::writeUnsignedLEB128(unsigned value) {
  do {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    if (value != 0)
      byte |= 0x80; // Set high bit if more bytes follow
    os.write(reinterpret_cast<char *>(&byte), 1);
  } while (value != 0);
}
