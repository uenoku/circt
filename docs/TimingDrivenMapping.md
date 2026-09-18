# Timing-driven mapping: load-aware delay + buffer insertion

## Status: Stage 1 landed (opt-in `--load-aware-delay`, default off);
Stage 2 / buffers remain.

## 8. Load-aware delay

Today matching uses one intrinsic number per arc: `DeriveMappingCost`
averages the whole NLDM table (all slews × all loads) into picoseconds,
and `TechMapper` copies only `intrinsic` (`TechMapper.cpp`: "sensitivity,
polarity, and input capacitance fields are carried in the attribute for
future load-aware mapping"). The other half already exists:
per-input-pin `capacitance` is stored in `mapping_cost.inputCaps`, and the
analysis side has an NLDM interpolator (`TimingAnalysis` + NLDM delay
model per the circt-sta-implementation skill).

Design (two stages):

- **Stage 1 — linear sensitivity (cheap, no iteration).** LANDED
  (opt-in `--load-aware-delay`, default off; zero default regression
  proven: fifo control 9003/36.802ns identical to pre-load-aware).
  `DeriveMappingCost` fits per-arc `sensitivity` (least-squares
  dDelay/dLoad over NLDM load columns at mid-slew; index_2 is output
  load on both PDKs, verified sky130/ASAP7; degenerate tables get 0).
  At match time, `delay = intrinsic + min(sensitivity × output_load,
  intrinsic)` (per-arc penalty cap: broadcasts are a buffering problem,
  not a selection signal), with `output_load` = min(fanout,64) ×
  library-mean input cap, applied only to roots with fanout ≤ 8
  (trusted-fanout gate). Unbounded penalties (fanout-2000 reset at
  27× past the fitted range) and even in-range penalties on shared
  roots otherwise dominate selection and arrival-ordered pruning.
  Measured: add32 opt-in 494→464 (-6%) and 2.26→1.82ns (-20%);
  fifo opt-in 9003→8572 (-4.8%) BUT worst path 36.8→87ns (same
  addr_bit0 endpoints: subject-fanout ≠ cover load after
  sharing/duplication, and arrival-only penalties lack criticality
  context — needs Stage 2/backward pass to fix; stays opt-in).
  Tests: `derive-mapping-cost.mlir` (fitted 308/100 case) +
  `tech-mapper-load-aware.mlir` (area/timing split + flag-off control).
- **Stage 2 — cover-iterated loads (ABC-style).** `output_load` = Σ input
  caps of the cells driven in the *current* cover; iterate mapping 2–3×,
  recomputing loads from the previous cover (same loop shape as
  area-flow reselection). Optionally share the analysis-side NLDM
  interpolator for (slew, load) → delay instead of the linear fit, with
  input-slew estimates from a forward arrival pass over the current cover.

Watch-outs: phase-flipped variants and supergates multiply the patterns
whose arcs need sensitivities (all derivable the same way); wide
AO/OA cells have the steepest load slopes, so ASAP7 gains most.

## 9. Buffer insertion

Nothing buffers high-fanout nets today; the fifo scoreboard is dominated
by a reset-fanout path (36.8ns arrival). New post-map pass
(`synth-buffer-high-fanout` sketch):

- Rank nets by total driven capacitance (Σ fanout input caps from
  `mapping_cost`, plus wire estimate).
- Above `--max-fanout-cap`, insert `clkbuf` (positive) chains sized by
  load; single level first, chain for very high fanout later.
- Never touch clock/reset nets with logic cells (no `isobufsrc`
  repurposing — item 16 showed why that reading was wrong; use real
  buffers), and never buffer synchronizer inputs.
- Close the loop with the STA scoreboard: buffers cost area/leakage, so
  land the threshold where worst-slack stops improving.

Requires 8 (sizing by load) for anything beyond a fixed threshold, and
the SDC work (docs/SdcEmission.md) so inserted buffers on clock trees
are constrained, not just logic nets.
