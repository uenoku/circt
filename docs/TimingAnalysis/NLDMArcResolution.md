# NLDM Cell/Arc Resolution in CIRCT TimingAnalysis

This document explains how CIRCT resolves a timing arc from IR operations to
typed NLDM metadata, and where the lookup can fail.

## Overview

For each timing arc traversal, the delay model needs to answer:

- which Liberty cell does this operation represent?
- which input pin and output pin are active for this arc?
- which NLDM arc table should be used?

The flow is intentionally metadata-driven and reuses ImportLiberty output.

## Source of Truth

NLDM data is imported once and attached to IR via typed Synth attributes:

- module/library scope: `#synth.nldm_time_unit<...>`
- pin scope: `#synth.nldm_arc<...>` entries in `synth.nldm.arcs`
- mapping hint on instances: `synth.liberty.cell = "..."`

Relevant importer:

- `lib/Conversion/ImportLiberty/ImportLiberty.cpp`

Relevant consumers:

- `include/circt/Dialect/Synth/Analysis/Timing/Liberty.h`
- `lib/Dialect/Synth/Analysis/Timing/Liberty.cpp`
- `include/circt/Dialect/Synth/Analysis/Timing/DelayModel.h`
- `lib/Dialect/Synth/Analysis/Timing/DelayModel.cpp`

## Resolution Steps

At a high level, lookup proceeds in this order:

1. Determine operation context and arc indices.
2. Resolve cell mapping for the op (commonly from `synth.liberty.cell`).
3. Resolve input/output pin names for the arc indices.
4. Find typed NLDM arc payload matching `(related_pin, to_pin, sense/type)`.
5. Interpolate delay/slew tables at `(inputSlew, outputLoad)`.

The arc context (`op + inputIndex + outputIndex`) is propagated by timing graph
construction and used in arrival/RAT/path recomputation.

## Hierarchical Behavior

When instance-level Liberty metadata is present, hierarchical timing graph build
treats the instance as a cell-level timing abstraction (black-box timing arcs),
instead of always elaborating internals. This preserves cell arc identity and
avoids losing NLDM context.

Key files:

- `include/circt/Dialect/Synth/Analysis/Timing/TimingGraph.h`
- `lib/Dialect/Synth/Analysis/Timing/TimingGraph.cpp`

## Failure Modes and Defaults

Current behavior is intentionally strict for typed NLDM:

- legacy dictionary fallback paths are removed
- no AIG-level fallback for unmatched NLDM arcs
- unmatched typed arcs default to zero delay unless explicit overrides exist

This makes missing mapping/metadata visible in reports/tests instead of silently
switching models.

## Debug Checklist

If delay is unexpectedly zero:

1. Confirm `synth.liberty.cell` exists on the timing-relevant op/instance.
2. Confirm output pin carries `synth.nldm.arcs` typed entries.
3. Confirm arc indices map to expected input/output pin names.
4. Confirm tables contain samples (non-empty index/value payloads).
5. Check timing report fields (`Delay Model`, path delays, convergence fields).

Useful tests:

- `unittests/Dialect/Synth/TimingAnalysisTest.cpp`
  - `NLDMDelayModelResolvesCellPinMapping`
  - `NLDMDelayModelReadsTimingArcTableValue`
  - `NLDMDelayModelInterpolatesOverSlewAndLoad`
  - `NLDMDelayModelInterpolatesOutputSlew`
- `test/circt-synth/nldm-timing-report.mlir`
- `test/circt-synth/nldm-timing-report-chain.mlir`

## Related Docs

- `docs/TimingAnalysis/Roadmap.md`
- `docs/TimingAnalysis/SlewConvergence.md`
