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
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Version.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
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

private:
  hw::HWModuleOp module;
  llvm::raw_ostream &os;
  const ExportAIGEROptions &options;

  // AIGER file data
  unsigned maxVarIndex = 0;
  unsigned numInputs = 0;
  unsigned numLatches = 0;
  unsigned numOutputs = 0;
  unsigned numAnds = 0;

  // Data structures for tracking variables and gates
  DenseMap<Value, unsigned> valueLiteralMap;
  SmallVector<Value> inputs;
  SmallVector<std::pair<Value, Value>> latches; // current, next
  SmallVector<Value> outputs;
  SmallVector<std::tuple<Value, Value, Value>> andGates; // lhs, rhs0, rhs1

  /// Analyze the module and collect information
  LogicalResult analyzeModule();

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
  unsigned getLiteral(Value value, bool inverted = false);

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
                          << " I=" << numInputs << " L=" << numLatches
                          << " O=" << numOutputs << " A=" << numAnds << "\n");

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
  os << " " << maxVarIndex << " " << numInputs << " " << numLatches << " "
     << numOutputs << " " << numAnds << "\n";

  return success();
}

LogicalResult AIGERExporter::writeInputs() {
  LLVM_DEBUG(llvm::dbgs() << "Writing inputs\n");

  if (options.binaryFormat) {
    // In binary format, inputs are implicit
    return success();
  }

  // Write input literals
  for (Value input : inputs) {
    unsigned literal = getLiteral(input);
    os << literal << "\n";
  }

  return success();
}

LogicalResult AIGERExporter::writeLatches() {
  LLVM_DEBUG(llvm::dbgs() << "Writing latches\n");

  // Write latch definitions
  for (auto [current, next] : latches) {
    unsigned currentLiteral = getLiteral(current);

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
  for (Value output : outputs) {
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
      auto andInvOp = lhs.getDefiningOp<aig::AndInverterOp>();
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
      }

      // In binary format, we need to write the delta values
      // Delta0 = lhs - rhs0
      // Delta1 = rhs0 - rhs1
      unsigned delta0 = lhsLiteral - rhs0Literal;
      unsigned delta1 = rhs0Literal - rhs1Literal;
      LLVM_DEBUG(llvm::dbgs()
                 << "Writing AND gate: " << lhsLiteral << " = " << rhs0Literal
                 << " & " << rhs1Literal << " (deltas: " << delta0 << ", "
                 << delta1 << ")\n");

      // Write deltas using variable-length encoding
      writeUnsignedLEB128(delta0);
      writeUnsignedLEB128(delta1);
    }
  } else {
    // Write AND gate definitions in ASCII format
    for (auto [lhs, rhs0, rhs1] : andGates) {
      unsigned lhsLiteral = getLiteral(lhs);

      // Get the AND-inverter operation to check inversion flags
      auto andInvOp = lhs.getDefiningOp<aig::AndInverterOp>();
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

  // TODO: Implement symbol table writing
  // Format: i<index> <symbol_name>
  //         l<index> <symbol_name>
  //         o<index> <symbol_name>

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

unsigned AIGERExporter::getLiteral(Value value, bool inverted) {
  // Handle constants
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

      // Look up input in literal map
      auto inputIt = valueLiteralMap.find(input);
      if (inputIt != valueLiteralMap.end()) {
        unsigned inputLiteral = inputIt->second;
        // Apply input inversion
        if (inputInverted)
          inputLiteral += 1;
        // Apply additional inversion if requested
        return inverted ? (inputLiteral ^ 1) : inputLiteral;
      }
    }
  }

  // Look up in the literal map
  auto it = valueLiteralMap.find(value);
  if (it != valueLiteralMap.end()) {
    unsigned literal = it->second;
    return inverted ? literal + 1 : literal;
  }

  // This should not happen if analysis was done correctly
  llvm_unreachable("Value not found in literal map");
}

LogicalResult AIGERExporter::analyzePorts(hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing module ports\n");

  // Analyze input ports
  for (auto [index, arg] :
       llvm::enumerate(hwModule.getBodyBlock()->getArguments())) {
    // Skip clock inputs for now (they're handled separately in latches)
    if (isa<seq::ClockType>(arg.getType()))
      continue;

    // All other inputs should be i1 (1-bit assumption)
    if (!arg.getType().isInteger(1))
      return emitError("input port must be i1 type, got ") << arg.getType();

    inputs.push_back(arg);
    LLVM_DEBUG(llvm::dbgs() << "  Input " << index << ": " << arg << "\n");
  }

  numInputs = inputs.size();

  // Analyze output ports by looking at hw.output operation
  auto walkResult = hwModule.walk([&](hw::OutputOp outputOp) {
    for (auto operand : outputOp.getOperands()) {
      // All outputs should be i1 (1-bit assumption)
      if (!operand.getType().isInteger(1)) {
        emitError("output port must be i1 type, got ") << operand.getType();
        return WalkResult::interrupt();
      }

      outputs.push_back(operand);
      LLVM_DEBUG(llvm::dbgs() << "  Output: " << operand << "\n");
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();

  numOutputs = outputs.size();

  LLVM_DEBUG(llvm::dbgs() << "Found " << numInputs << " inputs, " << numOutputs
                          << " outputs\n");
  return success();
}

LogicalResult AIGERExporter::analyzeOperations(hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing operations\n");

  auto walkResult = hwModule.walk([&](Operation *op) {
    if (auto andInvOp = dyn_cast<aig::AndInverterOp>(op)) {
      // Handle AIG AND-inverter operations
      if (andInvOp.getInputs().size() == 1) {
        // Single input = inverter (not counted as AND gate in AIGER)
        LLVM_DEBUG(llvm::dbgs() << "  Found inverter: " << andInvOp << "\n");
      } else if (andInvOp.getInputs().size() == 2) {
        // Two inputs = AND gate
        andGates.push_back({andInvOp.getResult(), andInvOp.getInputs()[0],
                            andInvOp.getInputs()[1]});
        numAnds++;
        LLVM_DEBUG(llvm::dbgs() << "  Found AND gate: " << andInvOp << "\n");
      } else {
        // Variadic AND gates need to be lowered first
        emitError("variadic AND gates not supported, run aig-lower-variadic "
                  "pass first");
        return WalkResult::interrupt();
      }
    } else if (auto regOp = dyn_cast<seq::CompRegOp>(op)) {
      // Handle registers (latches in AIGER)
      latches.push_back({regOp.getResult(), regOp.getInput()});
      numLatches++;
      LLVM_DEBUG(llvm::dbgs() << "  Found latch: " << regOp << "\n");
    }
    // Ignore other operations (hw.output, etc.)
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Found " << numAnds << " AND gates, " << numLatches
                          << " latches\n");
  return success();
}

LogicalResult AIGERExporter::assignLiterals() {
  LLVM_DEBUG(llvm::dbgs() << "Assigning literals\n");

  unsigned nextLiteral =
      2; // Start from 2 (literal 0 = FALSE, literal 1 = TRUE)

  // Assign literals to inputs first
  for (Value input : inputs) {
    valueLiteralMap[input] = nextLiteral;
    LLVM_DEBUG(llvm::dbgs()
               << "  Input literal " << nextLiteral << ": " << input << "\n");
    nextLiteral += 2; // Even literals only (odd = inverted)
  }

  // Assign literals to latches (current state)
  for (auto [current, next] : latches) {
    valueLiteralMap[current] = nextLiteral;
    LLVM_DEBUG(llvm::dbgs()
               << "  Latch literal " << nextLiteral << ": " << current << "\n");
    nextLiteral += 2;
  }

  // Assign literals to AND gate outputs
  for (auto [lhs, rhs0, rhs1] : andGates) {
    valueLiteralMap[lhs] = nextLiteral;
    LLVM_DEBUG(llvm::dbgs()
               << "  AND gate literal " << nextLiteral << ": " << lhs << "\n");
    nextLiteral += 2;
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

llvm::cl::opt<bool> useTextFormat("use-agg",
                                  llvm::cl::desc("Export AIGER in text format"),
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
        options.binaryFormat = !useTextFormat;

        return exportAIGER(*ops.begin(), os, &options);
      },
      [](DialectRegistry &registry) {
        registry.insert<aig::AIGDialect, hw::HWDialect, seq::SeqDialect>();
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
