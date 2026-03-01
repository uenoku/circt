//===- Liberty.h - Liberty Data Bridge for Timing Analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_LIBERTY_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_LIBERTY_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"
#include <optional>
#include <string>

namespace circt {
namespace synth {
namespace timing {

/// In-memory view of Liberty metadata imported into MLIR attributes.
///
/// This bridge reuses `import-liberty` output conventions by collecting
/// `synth.liberty.pin` attributes from HW modules.
class LibertyLibrary {
public:
  struct Pin {
    bool isInput = false;
    std::optional<double> capacitance;
    mlir::DictionaryAttr attrs;
  };

  struct Cell {
    hw::HWModuleOp module;
    llvm::StringMap<Pin> pins;
  };

  /// Build a Liberty view from imported attributes in a module.
  static FailureOr<LibertyLibrary> fromModule(mlir::ModuleOp module);

  /// Look up a cell by Liberty cell name.
  const Cell *lookupCell(llvm::StringRef cellName) const;

  /// Look up a pin within a Liberty cell.
  const Pin *lookupPin(llvm::StringRef cellName, llvm::StringRef pinName) const;

  /// Get input pin capacitance for a given cell/pin when available.
  std::optional<double> getInputPinCapacitance(llvm::StringRef cellName,
                                               llvm::StringRef pinName) const;

private:
  llvm::StringMap<Cell> cells;
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_LIBERTY_H
