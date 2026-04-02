//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESISIMPL_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESISIMPL_H

#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"

namespace circt {
namespace synth {

struct LoadedCutRewriteEntry {
  virtual ~LoadedCutRewriteEntry() = default;

  std::string moduleName;
  NPNClass npnClass;
  double area = 0.0;
  SmallVector<DelayType> delay;

  virtual FailureOr<Operation *> rewrite(OpBuilder &builder,
                                         CutEnumerator &enumerator,
                                         const Cut &cut) const = 0;
};

struct LoadedCutRewriteDatabase {
  std::string kind;
  std::vector<std::unique_ptr<LoadedCutRewriteEntry>> entries;
  unsigned maxInputSize = 0;
};

FailureOr<OwningOpRef<mlir::ModuleOp>>
parseCutRewriteDBFile(StringRef dbFile, mlir::MLIRContext *context);

LogicalResult
loadExactSynthesisDatabaseFromModule(mlir::ModuleOp dbModule,
                                     LoadedCutRewriteDatabase &database);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESISIMPL_H
