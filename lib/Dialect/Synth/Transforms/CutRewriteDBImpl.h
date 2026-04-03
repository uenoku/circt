//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_CUTREWRITEDBIMPL_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_CUTREWRITEDBIMPL_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>
#include <utility>

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
  mlir::OwningOpRef<mlir::ModuleOp> backingModule;
  std::vector<std::unique_ptr<LoadedCutRewriteEntry>> entries;
  unsigned maxInputSize = 0;
};

struct CutRewriteModuleMetadata {
  NPNClass npnClass;
  double area = 0.0;
  SmallVector<DelayType> delay;
  std::string inverterKind = "aig";
};

FailureOr<std::pair<double, SmallVector<DelayType>>>
getAreaAndDelayFromTechInfo(hw::HWModuleOp module);

FailureOr<NPNClass> getNPNClassFromModule(hw::HWModuleOp module);

FailureOr<std::unique_ptr<LoadedCutRewriteEntry>>
parseCutRewriteEntry(hw::HWModuleOp module,
                     const CutRewriteModuleMetadata &metadata);

FailureOr<OwningOpRef<mlir::ModuleOp>>
parseCutRewriteDBFile(StringRef dbFile, mlir::MLIRContext *context);

LogicalResult
loadCutRewriteDatabaseFromModule(mlir::ModuleOp dbModule,
                                 LoadedCutRewriteDatabase &database);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_CUTREWRITEDBIMPL_H
