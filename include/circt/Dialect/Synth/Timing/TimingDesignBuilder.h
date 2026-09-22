//===- TimingDesignBuilder.h - Build TimingDesign from mapped HW IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Elaborate a technology-mapped hierarchy into a flattened backend-neutral
// timing graph: hierarchy expansion with per-instance copies, bit-level
// connectivity through transparent wiring ops, cell and port binding
// validation, and diagnostics for unsupported constructs. See
// docs/StaticTimingAnalysisLibraryIntegration.md.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TIMING_TIMINGDESIGNBUILDER_H
#define CIRCT_DIALECT_SYNTH_TIMING_TIMINGDESIGNBUILDER_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Timing/TimingDesign.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace circt {
namespace synth {
namespace timing {

/// Builds a TimingDesign from mapped HW IR. The design must be technology
/// mapped: every timed leaf instance binds to a library cell; remaining
/// operations must be transparent wiring (extract, concat, replicate,
/// constants, module forwarding). Anything else is a validation error with
/// an actionable diagnostic; unsupported constructs never fall back to
/// optimistic zero delay.
class TimingDesignBuilder {
public:
  TimingDesignBuilder(mlir::MLIRContext *ctx, mlir::StringAttr topModule,
                      const llvm::DenseSet<mlir::StringAttr> &libraryCells);

  /// Elaborate and validate. Returns failure on any validation error;
  /// diagnostics remain available through getDiagnostics.
  mlir::FailureOr<TimingDesign> build(mlir::ModuleOp top);
  const TimingDesignDiagnostics &getDiagnostics() const { return diagnostics; }

private:
  mlir::MLIRContext *ctx;
  mlir::StringAttr topModule;
  llvm::DenseSet<mlir::StringAttr> libraryCells;
  TimingDesign design;
  TimingDesignDiagnostics diagnostics;
  bool failed = false;
  // Running node counter: every pin gets a unique dense id.
  uint32_t nextNode = 0;

  uint32_t freshNode() { return nextNode++; }

  // Active design modules on the current elaboration path (recursion guard).
  llvm::DenseSet<mlir::StringAttr> activeModules;

  uint32_t freshNet();
  void diagnose(llvm::StringRef message,
                const std::optional<TimingObjectKey> &object = std::nullopt);
  void fail(llvm::StringRef message,
            const std::optional<TimingObjectKey> &object = std::nullopt);

  // Bit width of a value (integers and clocks). Nullopt when unsupported.
  std::optional<unsigned> bitWidth(mlir::Value value);

  // Elaborate one module instantiation. Block arguments alias the given
  // parent nets; internal values get fresh per-instantiation nets. Returns
  // the result nets in (result, bit) order, or nullopt on failure.
  std::optional<SmallVector<SmallVector<uint32_t>>>
  elaborateModule(hw::HWModuleOp module,
                  ArrayRef<mlir::StringAttr> instancePath,
                  ArrayRef<SmallVector<uint32_t>> argNets, bool isTop);
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TIMING_TIMINGDESIGNBUILDER_H
