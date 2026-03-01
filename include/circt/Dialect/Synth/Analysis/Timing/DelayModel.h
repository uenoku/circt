//===- DelayModel.h - Pluggable Delay Model Interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DelayModel interface for computing arc delays.
// Concrete implementations include UnitDelayModel and AIGLevelDelayModel.
// The interface is designed to be extensible for future NLDM/CCCS models.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_DELAYMODEL_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_DELAYMODEL_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>

namespace mlir {
class ModuleOp;
}

namespace circt {
namespace synth {
namespace timing {

/// Context passed to delay model for each arc computation.
struct DelayContext {
  mlir::Operation *op;      // The defining operation
  mlir::Value inputValue;   // Input driving this arc
  mlir::Value outputValue;  // Output of this arc
  int32_t inputIndex = -1;  // Operand index for this arc (if known)
  int32_t outputIndex = -1; // Result index for this arc (if known)
  double inputSlew = 0.0;   // Input slew (transition time)
  double outputLoad = 0.0;  // Output load capacitance
};

/// Result of delay computation.
struct DelayResult {
  int64_t delay;           // Arc delay
  double outputSlew = 0.0; // Output slew to propagate to fanout
};

/// Generic waveform point for waveform-based timing models (e.g. CCS).
struct WaveformPoint {
  double time = 0.0;
  double value = 0.0;
};

/// Abstract base class for delay models.
class DelayModel {
public:
  virtual ~DelayModel() = default;

  /// Compute the delay for a given arc context.
  virtual DelayResult computeDelay(const DelayContext &ctx) const = 0;

  /// Get the name of this delay model.
  virtual llvm::StringRef getName() const = 0;

  /// Whether this model uses slew propagation.
  virtual bool usesSlewPropagation() const { return false; }

  /// Return effective input capacitance for the arc consumer pin.
  ///
  /// This lets analyses compute output load from fanout caps without hardcoding
  /// Liberty knowledge in graph traversal. Models that do not use load can
  /// keep the default zero value.
  virtual double getInputCapacitance(const DelayContext &ctx) const {
    (void)ctx;
    return 0.0;
  }

  /// Whether this model provides waveform-level propagation.
  ///
  /// CCS-style models can override this and `computeOutputWaveform` while
  /// scalar models keep the default behavior.
  virtual bool usesWaveformPropagation() const { return false; }

  /// Optional waveform propagation hook for waveform-capable models.
  ///
  /// Returns true when `outputWaveform` is produced, false when unsupported.
  virtual bool computeOutputWaveform(
      const DelayContext &ctx, llvm::ArrayRef<WaveformPoint> inputWaveform,
      llvm::SmallVectorImpl<WaveformPoint> &outputWaveform) const {
    return false;
  }
};

/// Unit delay model: returns 1 for all logic ops, 0 for wiring ops.
class UnitDelayModel : public DelayModel {
public:
  DelayResult computeDelay(const DelayContext &ctx) const override;
  llvm::StringRef getName() const override { return "unit"; }
};

/// AIG-level delay model: AIG=1, variadic AND/OR/XOR=log2(N),
/// bit manipulation=0, default=1.
class AIGLevelDelayModel : public DelayModel {
public:
  DelayResult computeDelay(const DelayContext &ctx) const override;
  llvm::StringRef getName() const override { return "aig-level"; }
};

/// Bootstrap NLDM-oriented delay model.
///
/// This model consumes per-op/per-arc delay attributes when available and
/// falls back to AIG-level heuristics otherwise. It is intended as a bridge
/// until full Liberty LUT interpolation is wired.
class NLDMDelayModel : public DelayModel {
public:
  NLDMDelayModel();
  explicit NLDMDelayModel(std::unique_ptr<class LibertyLibrary> liberty);
  ~NLDMDelayModel() override;

  DelayResult computeDelay(const DelayContext &ctx) const override;
  llvm::StringRef getName() const override { return "nldm"; }
  double getInputCapacitance(const DelayContext &ctx) const override;

private:
  AIGLevelDelayModel fallback;
  std::unique_ptr<class LibertyLibrary> liberty;
};

/// Create the default delay model (AIGLevelDelayModel).
std::unique_ptr<DelayModel> createDefaultDelayModel();

/// Create the bootstrap NLDM-oriented delay model.
std::unique_ptr<DelayModel> createNLDMDelayModel();

/// Create the NLDM-oriented delay model and wire imported Liberty metadata
/// from `module` when available.
std::unique_ptr<DelayModel> createNLDMDelayModel(mlir::ModuleOp module);

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_DELAYMODEL_H
