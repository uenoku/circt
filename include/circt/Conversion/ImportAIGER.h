//===- ImportAIGER.h - AIGER file import -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for importing AIGER files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_IMPORTAIGER_H
#define CIRCT_CONVERSION_IMPORTAIGER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
class TimingScope;
} // namespace mlir

namespace circt {

/// Options for AIGER import.
struct ImportAIGEROptions {
  /// The name to use for the top-level module. If empty, a default name
  /// will be generated.
  std::string topLevelModule = "";
  
  /// Whether to preserve symbol names from the AIGER file.
  bool preserveSymbols = true;
  
  /// Whether to support binary AIGER format in addition to ASCII.
  bool supportBinary = true;
};

/// Parse an AIGER file and populate the given MLIR module with corresponding
/// AIG dialect operations.
mlir::LogicalResult
importAIGER(llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context,
            mlir::TimingScope &ts, mlir::ModuleOp module,
            const ImportAIGEROptions *options = nullptr);

/// Register the `import-aiger` MLIR translation.
void registerFromAIGERTranslation();

} // namespace circt

#endif // CIRCT_CONVERSION_IMPORTAIGER_H
