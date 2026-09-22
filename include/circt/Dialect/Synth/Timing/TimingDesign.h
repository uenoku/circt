//===- TimingDesign.h - Backend-neutral flattened timing graph ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Immutable backend-neutral flattened netlist for timing analysis: cells,
// bit-level nets, primary ports, and diagnostics. Built from mapped HW IR
// by TimingDesignBuilder; consumed by timer backends. See
// docs/StaticTimingAnalysisLibraryIntegration.md.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TIMING_TIMINGDESIGN_H
#define CIRCT_DIALECT_SYNTH_TIMING_TIMINGDESIGN_H

#include "circt/Dialect/Synth/Timing/TimingResult.h"
#include "circt/Dialect/Synth/Timing/TimingTypes.h"

#include <cstdint>
#include <optional>

namespace circt {
namespace synth {
namespace timing {

/// One pin of a cell instance in the flattened graph.
struct TimingPinConnection {
  TimingNodeId node;
  mlir::StringAttr portName;
  unsigned portIndex = 0;
  bool isInput = true;
  unsigned bit = 0;
};

/// One cell instance: type, persistent identity, and pins.
struct TimingCellInstance {
  uint32_t id = 0;
  mlir::StringAttr cellType;
  TimingObjectKey instance;
  SmallVector<TimingPinConnection, 4> pins;
};

/// One bit-level net: optional single driver plus loads. A null driver
/// with constantValue set is a tied constant; a null driver without one
/// is undriven (a validation error unless explicitly allowed).
struct TimingNet {
  uint32_t id = 0;
  std::optional<TimingNodeId> driver;
  SmallVector<TimingNodeId, 4> loads;
  bool isConstant = false;
  bool constantValue = false;
};

/// One top-level port bit.
struct TimingPrimaryPort {
  TimingObjectKey port;
  bool isInput = true;
};

/// Diagnostics collected during graph construction. Unsupported
/// constructs are errors (request failure), never silent omissions.
struct TimingDesignDiagnostics {
  SmallVector<TimingDiagnostic, 0> unsupported;
  uint64_t numCells = 0;
  uint64_t numNets = 0;
  uint64_t numConstantNets = 0;
};

/// Immutable flattened timing design.
struct TimingDesign {
  mlir::StringAttr topModule;
  SmallVector<TimingCellInstance, 0> cells;
  SmallVector<TimingNet, 0> nets;
  SmallVector<TimingPrimaryPort, 0> ports;
  TimingDesignDiagnostics diagnostics;
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TIMING_TIMINGDESIGN_H
