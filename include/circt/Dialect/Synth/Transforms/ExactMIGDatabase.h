//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for generating exact MIG cut-rewrite databases.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTMIGDATABASE_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTMIGDATABASE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <string>

namespace circt {
namespace synth {

struct ExactMIGDatabaseGenOptions {
  std::string databaseName = "MIG_EXACT";
  std::string satSolver = "auto";
  unsigned maxInputs = 4;
  int64_t conflictLimit = 100;
};

/// Populate `module` with canonical exact-MIG implementations encoded as
/// `hw.module`s carrying `hw.techlib.info` and cut-rewrite metadata.
LogicalResult emitExactMIGDatabase(mlir::ModuleOp module,
                                   const ExactMIGDatabaseGenOptions &options);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTMIGDATABASE_H
