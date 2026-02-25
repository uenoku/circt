//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for exporting SCA (Symbolic Computer Algebra)
// proof scripts for verifying arithmetic circuits using Gröbner basis reduction.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_EXPORTSCAPROOF_H
#define CIRCT_CONVERSION_EXPORTSCAPROOF_H

#include "circt/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hw {
class HWModuleOp;
} // namespace hw

/// Export two hw.modules (spec and impl) as a Singular CAS script that
/// verifies their equivalence using Gröbner basis reduction.
mlir::LogicalResult exportSCAProof(hw::HWModuleOp specModule,
                                   hw::HWModuleOp implModule,
                                   llvm::raw_ostream &os);

/// Register the `export-sca-proof` MLIR translation.
void registerExportSCAProofTranslation();

} // namespace circt

#endif // CIRCT_CONVERSION_EXPORTSCAPROOF_H
