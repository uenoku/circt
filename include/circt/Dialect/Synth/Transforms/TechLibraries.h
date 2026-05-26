//===- TechLibraries.h - Built-in tech libraries ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_TECHLIBRARIES_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_TECHLIBRARIES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace synth {

/// Append one of the built-in technology libraries to `module`.
mlir::LogicalResult appendBuiltinTechLibrary(mlir::ModuleOp module,
                                             llvm::StringRef libraryName);

/// Append HW modules from an MLIR technology library file to `module`.
mlir::LogicalResult appendTechLibraryFile(mlir::ModuleOp module,
                                          llvm::StringRef filename);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_TECHLIBRARIES_H
