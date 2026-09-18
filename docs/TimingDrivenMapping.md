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
  ChipTop opt-in: 2080890→1989878 (-4.4%, flops exact); mechanistic
  win (mux2i_1→mux2i_4 drive shift on loaded nets) with delay cost
  on both axes (sync 6.78→10.32ns, async +3.22→-3.12) — selection
  responds to loads, worst paths need required-time gating.
  fifo flag combos (opt-ins compose safely): load-aware+buffers =
  8752/8.21ns (buffers fix the 87ns pathology; buf-only 9256/4.45ns
  still best delay). fifo ASAP7 buffers: 13055/32.13ns → 13267/0.556ns
  (58×; auto-picked HB1xp67_ASAP7_75t_L — PDK-agnostic discovery works).
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

## 9. Buffer insertion — LANDED v1 (opt-in `--buffer-high-fanout`,
default off; `--max-fanout`, default 32)

`synth-buffer-high-fanout` runs post-map (after TieCells, before strip
so liberty pin metadata is available). Per-copy fanout counting
*through hierarchy*: local gate loads are chunked behind one buffer
instance per ≤max-fanout chunk, and output ports with large downstream
loads get one buffer between driver and port. Ranking by subject-graph
refcount alone misses broadcasts (the addr_bit0 flop shows 8 local
loads but ~500 through-port memory-mux forest loads — 11.9pF at STA);
per-copy downstream loads are memoized over the instance DAG, no
multiplicity needed (every flat copy sees the same loads). The buffer
cell is auto-detected (smallest-area simulated identity cell:
sky130 `buf_1`), so the pass stays PDK-agnostic. Nets driving
clock=true pins are never touched (unknown targets conservatively
count as clock); tie-cell/constant drivers need no buffers.
Deterministic (file/IR order only; fifo A/B and ChipTop pre/post-speed-opt
bit-identical). Speed: instance index keeps ChipTop at 1m23s vs 45s
baseline (was 3m39s before indexing).

Measured (same binary): fifo 9003/36.802ns → 9256/4.451ns (**8.3×**,
+253 bufs, flops exact, MET with +5.47 slack at period 10); add32
untouched (0 buffers, no high fanout); ChipTop 2080890→2143663 (+3.0%,
flops exact), worst slack +3.22→+3.36 MET (different path; neutral to
slightly positive where broadcasts don't dominate).

Follow-ups: buffer trees/chains for very high fanout (single level
today), load-sized selection (needs item 8 Stage 2), synchronizer-input
markers (no reliable detection today; STA judges), default-on decision
once area/delay tradeoff is closed on more designs.
