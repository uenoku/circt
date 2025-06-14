//===- ExportAIGER.h - AIGER file export -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for exporting AIGER files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_EXPORTAIGER_H
#define CIRCT_CONVERSION_EXPORTAIGER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class MLIRContext;
class TimingScope;
} // namespace mlir

namespace circt {
namespace hw {
class HWModuleOp;
} // namespace hw

/// Options for AIGER export.
struct ExportAIGEROptions {
  /// Whether to export in binary format (aig) or ASCII format (aag).
  /// Default is ASCII format.
  bool binaryFormat = false;

  /// Whether to include symbol table in the output.
  /// Default is true.
  bool includeSymbolTable = true;

  /// Whether to include comments in the output.
  /// Default is true.
  bool includeComments = true;

  /// Callback for unknown operations.
  /// If true, operand and result will extracted to outputs and inputs respectively.
  /// Clients are expected to record this information in their use case.
  bool handleUnknownOperation = false;
  // Return true if the operand should be added to the output, false otherwise.
  // If returned false, outputIndex will be invalid for the given operand.
  std::function<bool(mlir::OpOperand& operand, size_t bitPos, size_t outputIndex)> unknownOperationOperandHandler = nullptr;
  // Return true if the result should be added to the input, false otherwise.
  // If returned false, inputIndex will be invalid for the given result.
  std::function<bool(mlir::OpResult result, size_t bitPos, size_t inputIndex)> unknownOperationResultHandler = nullptr;
};

/// Export an MLIR module containing AIG dialect operations to AIGER format.
mlir::LogicalResult exportAIGER(hw::HWModuleOp module, llvm::raw_ostream &os,
                                const ExportAIGEROptions *options = nullptr);

/// Register the `export-aiger` MLIR translation.
void registerToAIGERTranslation();

} // namespace circt

#endif // CIRCT_CONVERSION_EXPORTAIGER_H
