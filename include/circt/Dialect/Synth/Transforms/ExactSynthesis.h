//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for exact-synthesizing truth-table database entries.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESIS_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESIS_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <string>

namespace circt {
namespace synth {

struct ExactSynthesisRunOptions {
  std::string satSolver = "auto";
  std::string cadicalConfig = "default";
  int64_t conflictLimit = 100;
};

struct ExactSynthesisDatabaseGenOptions : ExactSynthesisRunOptions {
  unsigned maxInputs = 4;
};

/// Exact-synthesize each `hw.module` in `module` from a truth-table body into
/// a concrete database implementation for the requested backend family.
llvm::LogicalResult exactSynthesizeTruthTable(
    circt::hw::HWModuleOp module, llvm::StringRef kind,
    const ExactSynthesisRunOptions &options);

/// Populate `module` with canonical exact-synthesis implementations encoded as
/// `hw.module`s whose bodies are replayed by the cut-rewrite loader.
llvm::LogicalResult
emitExactSynthesisDatabase(mlir::ModuleOp module, llvm::StringRef kind,
                           const ExactSynthesisDatabaseGenOptions &options);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESIS_H
