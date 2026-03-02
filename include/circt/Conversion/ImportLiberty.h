//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the registration for the Liberty file importer.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_IMPORTLIBERTY_H
#define CIRCT_CONVERSION_IMPORTLIBERTY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"

namespace circt::liberty {

/// Import a Liberty source into an existing MLIR module.
mlir::LogicalResult importLiberty(llvm::SourceMgr &sourceMgr,
                                  mlir::MLIRContext *context,
                                  mlir::ModuleOp module);

/// Register the Liberty importer in the translation registry.
void registerImportLibertyTranslation();

} // namespace circt::liberty

#endif // CIRCT_CONVERSION_IMPORTLIBERTY_H
