//===- TimingTypes.h - Shared timing units and identity ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared value types for timing analysis: explicit time units, transition
// kinds, persistent object identity, typed constraints, and backend
// capabilities. Backend-neutral: no timer types appear here. See
// docs/StaticTimingAnalysisLibraryIntegration.md.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TIMING_TIMINGTYPES_H
#define CIRCT_DIALECT_SYNTH_TIMING_TIMINGTYPES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace circt {
namespace synth {
namespace timing {

/// Integral picoseconds. All time values cross backend boundaries in this
/// unit; floating point is converted once with checked rounding.
struct TimePs {
  int64_t value = 0;
};

/// Femtofarads for capacitance values.
struct CapacitanceFf {
  double value = 0.0;
};

/// Signal transition direction.
enum class TimingTransition { Rise, Fall };

/// Analysis mode. Only setup/max is supported initially.
enum class TimingAnalysisKind { SetupMax, HoldMin };

/// Dense handle valid only for one TimingDesign. Makes graph construction
/// and backend maps efficient.
struct TimingNodeId {
  uint32_t value = 0;
};

/// What a persistent object key refers to.
enum class TimingObjectKind { Instance, InstancePin, TopPort };

/// Persistent semantic key for timing objects. An instance pin is identified
/// by the explicit top-relative instance path, referenced module, port index,
/// port name, and bit. A top port has an empty instance path. Indices and
/// names are both recorded so importers can diagnose a renamed or reordered
/// interface instead of resolving the wrong object. Display names are
/// generated only for diagnostics and are never authoritative.
struct TimingObjectKey {
  TimingObjectKind kind = TimingObjectKind::TopPort;
  SmallVector<mlir::StringAttr> instancePath;
  mlir::StringAttr moduleName;
  unsigned portIndex = 0;
  mlir::StringAttr portName;
  unsigned bit = 0;

  bool operator==(const TimingObjectKey &other) const {
    return kind == other.kind && moduleName == other.moduleName &&
           instancePath == other.instancePath && portName == other.portName &&
           portIndex == other.portIndex && bit == other.bit;
  }
  bool operator!=(const TimingObjectKey &other) const {
    return !(*this == other);
  }
};

/// A clock constraint with an explicit source object.
struct TimingClock {
  mlir::StringAttr name;
  TimingObjectKey source;
  TimePs period;
  TimePs riseEdge;
  TimePs fallEdge;
};

/// Arrival/required/slew/load constraint on one top port.
struct TimingPortConstraint {
  TimingObjectKey port;
  std::optional<TimePs> lateArrival;
  std::optional<TimePs> lateRequired;
  std::optional<TimePs> riseSlew;
  std::optional<TimePs> fallSlew;
  std::optional<CapacitanceFf> load;
};

/// Timing constraints without Tcl or SDC syntax.
struct TimingConstraints {
  SmallVector<TimingClock> clocks;
  SmallVector<TimingPortConstraint> ports;
  TimingAnalysisKind analysis = TimingAnalysisKind::SetupMax;
};

/// Backend capability report. Must reflect the actual implementation.
struct TimingCapabilities {
  bool setupMax = false;
  bool holdMin = false;
  bool multipleClocks = false;
  bool generatedClocks = false;
  bool timingExceptions = false;
  bool latches = false;
  bool interconnectDelay = false;
};

/// How much detail to extract. Bounds traces without limiting aggregates.
struct TimingQuery {
  unsigned maxPaths = 100;
  unsigned maxPathsPerEndpoint = 1;
  bool includePathPoints = false;
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TIMING_TIMINGTYPES_H
