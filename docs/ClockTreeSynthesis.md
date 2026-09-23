//===- ClockTreeSynthesis.md - CTS plan for the CIRCT/OpenROAD flow ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Clock tree synthesis (CTS) plan: what CIRCT provides, what OpenROAD
// (TritonCTS) consumes, and how timing closes the loop. Clocks are ideal
// until CTS; pre-CTS signoff uses margins, post-CTS uses propagated
// network timing.
//
//===----------------------------------------------------------------------===//

# CTS plan

## Current state

- `synth-emit-sdc` traces flop clock pins to top ports and emits
  `create_clock` plus default I/O delays. No uncertainty, no latency, no
  generated clocks; unresolvable clocks become comments.
- The OpenTimer backend creates ideal clocks on source pins (zero network
  delay, 1ps source slew). Required times assume ideal capture.
- `--buffer-high-fanout` buffers data broadcasts but explicitly skips
  clock networks (clock pins must stay unbuffered pre-CTS for TritonCTS
  to own the tree).

## What CTS needs from CIRCT

1. **Complete clock definitions**: every flop clock pin traces to exactly
   one `create_clock` source (port, divided, or generated). Comb-generated
   and gated clocks need explicit `create_generated_clock` / gating cells,
   not comments.
2. **Uncertainty budgets**: `set_clock_uncertainty` (setup/hold) covering
   pre-CTS skew estimates plus jitter, so pre-CTS optimization doesn't
   chase false precision. Values come from library characterization
   corners and target skew, not guesses.
3. **Latency exclusions**: `set_clock_latency -source` for off-chip
   insertion delay where known; zero by default (documented).
4. **Buffer footprints**: clock-capable buffers (CK* cells where the
   library provides them) must stay available to the mapper but unused
   pre-CTS; a `dont_use`/`dont_touch` discipline keeps TritonCTS's
   choices open.
5. **SDC for TritonCTS**: the emitted SDC must parse in OpenSTA/OpenROAD
   without edits (no CIRCT-only constructs), including false paths for
   asynchronous resets and CDC synchronizers.

## Timing modes

- **Pre-CTS (ideal)**: zero network delay + uncertainty margins. Used for
  synthesis judging (current OpenTimer backend) and optimization. Pessimism
  is bounded by the uncertainty budget and documented per design.
- **Post-CTS (propagated)**: real network delays from OpenROAD (SPEF or
  STA report). The backend must accept propagated arrivals/requireds per
  clock pin (extension to `TimingConstraints`: per-pin clock latency) and
  skip ideal-clock creation when they are present. Differential harness
  compares pre- vs post-CTS to validate margin adequacy.
- **Hold**: currently setup/max only. CTS affects hold (skew directly);
  hold/min support follows propagated clocks (documented ordering).

## Feedback loop

1. Synthesize with ideal clocks + margins (current flow).
2. OpenROAD CTS + route; OpenSTA propagated signoff.
3. If violations trace to synthesis choices (bad logic depth, missed
   buffering, wrong drive), feed endpoint slacks back as probes/regions
   (existing refinement flow) with propagated clocks attached.
4. If violations trace to CTS/route (congestion, detours), adjust
   placement/CTS options, not synthesis. The differential (pre- vs
   post-CTS) attributes blame: logic-depth deltas implicate synthesis,
   network-delay deltas implicate CTS.

## Phases

- **Phase 1**: uncertainty + generated-clock SDC emission; async/C DC
  false paths; OpenROAD-parseable SDC tests.
- **Phase 2**: propagated-clock constraints in `TimingConstraints` and the
  OpenTimer backend (per-pin latency, no ideal creation); pre/post-CTS
  differential in the harness.
- **Phase 3**: hold/min support; skew-aware refinement loop driven by
  post-CTS STA.

## Non-goals

CIRCT does not build clock trees (no TritonCTS reimplementation). It
provides complete definitions, honest margins, and a timing backend that
understands both ideal and propagated clocks.
