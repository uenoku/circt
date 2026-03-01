# CCS Pilot Waveform-Derived Metrics

This note explains how `CCSPilotDelayModel` derives timing quantities from
typed CCS pilot waveform metadata, and how to enable waveform-derived delay.

## Scope

Applies to:

- `CCSPilotDelayModel` in
  `lib/Dialect/Synth/Analysis/Timing/DelayModel.cpp`
- typed CCS pilot metadata imported by `import-liberty`
- timing report waveform details (`circt-sta --show-waveform-details`)

This is a pilot path for CCS-style waveform coupling. It is intentionally
simplified and not yet full signoff CCS.

## Metadata Path

Importer emits output-pin attributes:

- `synth.ccs.pilot.arcs = [#synth.ccs_pilot_arc<...>]`

from Liberty timing groups containing:

- `output_current_rise`
- `output_current_fall`

Typed attribute fields:

- `relatedPin`, `toPin`
- rise: `currentRiseTimes`, `currentRiseValues`
- fall: `currentFallTimes`, `currentFallValues`
- receiver data: `synth.ccs.pilot.receivers = [#synth.ccs_pilot_receiver<...>]`
  carrying `receiver_capacitance{1,2}_{rise,fall}` tables.
- optional vector-set payload in `#synth.ccs_pilot_arc` for nested
  `vector(ccs_template)` forms (`selector indices`, `reference times`,
  `vector times/values`).

Key files:

- `include/circt/Dialect/Synth/SynthAttributes.td`
- `lib/Conversion/ImportLiberty/ImportLiberty.cpp`
- `include/circt/Dialect/Synth/Analysis/Timing/Liberty.h`
- `lib/Dialect/Synth/Analysis/Timing/Liberty.cpp`

## Waveform Selection

When computing propagated waveform:

- rising input waveform selects rise current table
- falling input waveform selects fall current table
- if preferred table is unavailable, model falls back to the other table

Waveform edge direction is inferred from input waveform endpoints.

For multi-input cells, arc selection is pin-specific (`relatedPin` / operand
index), so different inputs can produce different waveform-derived delay/slew.

For nested CCS vector sets, current pilot policy prefers structured bilinear
interpolation on regular `(index_1, index_2)` selector grids, then falls back
to axis-linear interpolation on sparse row/column slices, and finally to
inverse-distance blending for irregular sets. Interpolated `reference_time` is
applied as a time-axis offset before threshold extraction.

When blended vectors use different `index_3` sample grids, pilot decode now
uses `index_3` as a true time axis and performs linear resampling onto the
selected template waveform before blending values.

## Load-Aware Timing Stretch

Pilot model applies a load stretch to waveform time axis:

`stretch = max(0.5, 1.0 + 0.5 * (outputLoad - 0.5))`

Waveform sample times are transformed as:

`t_ps = arcDelay_ps + sampleTime_lib * timeScale_ps * stretch`

This approximates receiver/load sensitivity while keeping implementation simple.

When receiver tables are available, CCS pilot computes an effective receiver
capacitance from `receiver_capacitance{1,2}_{rise,fall}` at
`(inputSlew, outputLoad)` and blends it with output load before stretch.

## Threshold Extraction

From each propagated waveform, model/report extracts:

- `t50`: 50% crossing time
- `slew10-90`: |t90 - t10|

Crossings are linearly interpolated between neighboring waveform points.

## Delay Behavior Modes

Default mode:

- arc `delay` comes from NLDM delegate
- `outputSlew` is overwritten by waveform-derived `slew10-90` when available

Waveform-delay mode (opt-in):

- if module attr `synth.ccs.pilot.waveform_delay` is truthy,
  `delay` is set to waveform-derived `t50`
- `outputSlew` remains waveform-derived `slew10-90`

Supported truthy forms:

- `true` (bool)
- nonzero integer
- string `"true"` or `"1"`

Mixed delegation mode:

- module attr `synth.timing.model = "mixed-ccs-pilot"`
- module attr `synth.ccs.pilot.cells = ["CELL_A", ...]`
- cells listed in `synth.ccs.pilot.cells` use CCS pilot behavior;
  others use NLDM behavior.

Waveform-coupled convergence heuristics:

- enabled by default via `enableWaveformCoupledConvergence`
- active for waveform-capable models
- applies:
  - conservative adaptive damping when explicit mode is disabled
  - default relative epsilon `0.05` when unset
  - initial damping cap at `0.8`
  - minimum of 2 slew iterations

## Reporting

With waveform details enabled (`--show-waveform-details`), each arc line prints:

- waveform points `(t=..., v=...)`
- `t50=...`
- `slew10-90=...`

Implemented in:

- `lib/Dialect/Synth/Analysis/PrintTimingAnalysis.cpp`
- `lib/Dialect/Synth/Analysis/Timing/TimingReport.cpp`

## Test Coverage

Unit tests (`unittests/Dialect/Synth/TimingAnalysisTest.cpp`):

- `CCSPilotDelayModelProducesWaveform`
- `CCSPilotDelayModelUsesFallWaveformForFallingEdge`
- `CCSPilotWaveformStretchesWithOutputLoad`
- `CCSPilotDelayUsesWaveformThresholdWhenEnabled`
- `CCSPilotDelayFallsBackToNLDMDelayWhenDisabled`
- `CCSPilotMultiInputArcsProduceDifferentDelay`
- `ReportTimingIncludesWaveformDetailsWhenEnabled`

E2E waveform details:

- `test/circt-sta/ccs-pilot-timing-report.mlir`
- `test/circt-sta/ccs-pilot-waveform-delay-multi-input.mlir`
- `test/circt-sta/mixed-ccs-pilot-timing-report.mlir`
- `test/circt-sta/ccs-pilot-waveform-delay-report.mlir`
- `test/circt-sta/mixed-ccs-pilot-critical-path-policy.mlir`
- `test/circt-sta/ccs-pilot-vector-interpolation-report.mlir`
- `test/circt-sta/ccs-pilot-vector-grid-interpolation-report.mlir`
- `test/circt-sta/ccs-pilot-vector-axis-linear-report.mlir`
- `test/circt-sta/ccs-pilot-vector-template-semantics-report.mlir`

## Limitations and Next Steps

Current pilot does not yet model:

- full CCS current-source equations
- receiver waveform iteration per fanout branch
- signoff-equivalent threshold and arc characterization policies

Next implementation steps are tracked in `docs/TimingAnalysis/Roadmap.md`.
