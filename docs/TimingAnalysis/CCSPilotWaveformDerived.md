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

## Load-Aware Timing Stretch

Pilot model applies a load stretch to waveform time axis:

`stretch = max(0.5, 1.0 + 0.5 * (outputLoad - 0.5))`

Waveform sample times are transformed as:

`t_ps = arcDelay_ps + sampleTime_lib * timeScale_ps * stretch`

This approximates receiver/load sensitivity while keeping implementation simple.

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
- `ReportTimingIncludesWaveformDetailsWhenEnabled`

E2E waveform details:

- `test/circt-synth/ccs-pilot-timing-report.mlir`

## Limitations and Next Steps

Current pilot does not yet model:

- full CCS current-source equations
- receiver waveform iteration per fanout branch
- signoff-equivalent threshold and arc characterization policies

Next implementation steps are tracked in `docs/TimingAnalysis/Roadmap.md`.
