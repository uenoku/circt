//===- Passes.h - ESI pass entry points -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIPASSES_H
#define CIRCT_DIALECT_ESI_ESIPASSES_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace circt {
namespace hw {
class HWDesignOp;
} // namespace hw
namespace esi {

/// This should eventually become a set of functions to define the various
/// platform-specific lowerings.
struct Platform {
  static constexpr char cosim[] = "cosim";
  static constexpr char fpga[] = "fpga";
};

#define GEN_PASS_DECL
#include "circt/Dialect/ESI/ESIPasses.h.inc"

std::unique_ptr<OperationPass<>> createESIVerifyConnectionsPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESIPhysicalLoweringPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESIBundleLoweringPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESIPortLoweringPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESITypeLoweringPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESItoHWPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESIConnectServicesPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESICleanMetadataPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESIBuildManifestPass();
std::unique_ptr<OperationPass<hw::HWDesignOp>> createESIAppIDHierPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/ESI/ESIPasses.h.inc"

} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_ESIPASSES_H
