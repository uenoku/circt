# circt-synth synthesis-flow status

## Goal
Make `circt-synth` reproduce the target synthesis flow in
`circt-sfc-experiments/OpenROAD-flow-scripts/flow`:
a liberty-mapped netlist plus SDC, for the
`fifo`, `Rocket`, `ChipTop`/`SmallBoomV4Config` examples, on sky130hd
and ASAP7.

## Branch / commits (branch: `dev/hidetou/openroad`)
- `3272e2df96` — this file (handoff doc)
- `25597653a8` — llhd-lower-sync-processes (conditional-drive sync processes)
- `fed3cbaffe`, `05ff4c76d9` — strip-assert-like, liberty integration (prior)
- `cd62d85841` — WIP e2e fixes: llhd-strip-init-drives, comb And/Or truth-table
  sim, wide-flop explosion, mem→flop fallback
- `72399efba7` — cherry-pick of upstream PR #10193, squashed to one commit
  (timing-preserving area-flow reselection; 12 commits)
- `13142044fa` — cherry-pick of upstream PR #11106 (RegOfVecToMem masked writes)
- `73ea58ca23` — WIP: NPN canonical-form cache, word-level truth-table
  transforms, array_inject O(N) lowering
- `2cb6121e75` — masked-write flop fallback, array-flop explosion + tests
- `5f1adb30ce` — phase-aware tech mapping (inverter materialization) + 6-input
  pattern cap + tests
- `59a898a069` — WIP: NPN cache hit/miss stats instrumentation
- Cherry-picked PRs kept as their own commits with original messages intact.

## Build & test
```
cmake -G Ninja llvm/llvm -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_EXTERNAL_PROJECTS=circt \
  -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD -DLLVM_ENABLE_LLD=ON
ninja -C build bin/circt-opt bin/circt-synth bin/circt-verilog
ninja -C build check-circt   # full suite; see pre-existing failures below
```
Targeted lit (run from `build/test` against the build-tree mirror paths):
```
cd build/test && /usr/bin/python3.12 /home/uenoku/dev/circt/llvm/build/bin/llvm-lit -s \
  Dialect/Synth/tech-mapper-phase.mlir Dialect/LLHD/Transforms/strip-init-drives.mlir
```
Re-verified 2026-09-14 ~01:20: `tech-mapper-phase` + `strip-init-drives` +
`tech-mapper` 3/3 green; broader `Dialect/Synth` + `Dialect/LLHD` sweep
53/57 with only the 4 known pre-existing `tech-mapper-external*` failures.
`ninja check-circt` result: **1643 passed, 7 failed — all 7 pre-existing
and environmental** (verify before/after any change):
- `Conversion/ImportVerilog/libraries.sv` — `circt-translate` lacks `-l`
  flag (Slang CLI drift)
- `Dialect/OM/class-slice{,-errors}.mlir` — OM dialect, untouched by us
- `Dialect/Synth/tech-mapper-external{,-constant,-error,-multi}.mlir` —
  stale `lib-files` pass option (fails at arg parsing)
- Also clang-format C++ (`clang-format -i`) per AGENTS.md.

## End-to-end recipes
PDK libs:
- sky130hd: `circt-sfc-experiments/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib`
- ASAP7 LVT_FF set (assemble once): exactly these 5 files —
  `asap7sc7p5t_SIMPLE_LVT_FF_nldm_201020.lib`,
  `asap7sc7p5t_INVBUF_LVT_FF_nldm_201020.lib`,
  `asap7sc7p5t_AO_LVT_FF_nldm_201020.lib`,
  `asap7sc7p5t_OA_LVT_FF_nldm_201020.lib` (all extracted with `7z x` from
  `/home/uenoku/dev/asap7/asap7sc7p5t_27/LIB/NLDM/*.lib.7z`),
  plus `asap7sc7p5t_SEQ_LVT_FF_nldm_220123.lib` (all under `~/Downloads/`).
  Pass comma-separated via one `--liberty a,b,c,d,e` (no spaces).
  INV/BUF live in a separate INVBUF lib (ORFS uses it too); no MUX2
  cells exist in these libs (muxes decompose — correct, verbose:
  ChipTop ASAP7 has 681K INVs).
  NOTE: `~/Downloads/asap7sc7p5t_AO_SLVT_FF_nldm_201020.lib` also exists
  but is NOT part of the set (SLVT ≠ LVT).

Stubs needed for elaboration (recreate under /tmp; /tmp is wiped
between sessions!):
```systemverilog
// /tmp/plusarg_stub.sv
module plusarg_reader #(parameter FORMAT = "", parameter [31:0] DEFAULT = 0,
                        parameter [127:0] WIDTH = 1) (output [WIDTH-1:0] out);
  assign out = DEFAULT[WIDTH-1:0];
endmodule
// /tmp/io_stub.sv  (ChipTop only; plain ports keep llhd.ref out of the IR;
//                   oe/ie are tied 1'h1 at all 76 sites, so passthrough is sound)
module GenericDigitalInIOCell(input pad, output i, input ie);
  assign i = pad;
endmodule
module GenericDigitalOutIOCell(output pad, input o, input oe);
  assign pad = o;
endmodule
module EICG_wrapper(output out, input in, input test_en, input en);
  assign out = in & (en | test_en);
endmodule
```

Flows (all verified; timings on 32-core/62GB box):
```
# fifo → sky130: ~3 s, 15,641 instances, 2073 flops exact (2048 RAM + 25)
./build/bin/circt-verilog --top=fifo circt-sfc-experiments/example/fifo/fifo.v -o /tmp/e2e/fifo.mlir
./build/bin/circt-synth --liberty <sky130.lib> --strip-lib-to-extern --top fifo /tmp/e2e/fifo.mlir -o /tmp/e2e/fifo_synth.mlir
# fifo → ASAP7 (5 libs): ~4 s, same flop count
# ChipTop → ASAP7 (5 libs + same tmp/sram-macros-chiptop.mlir — macros are
# liberty-independent): ~33 s, 1,974,297 instances, 226,157 flops EXACT-match
# with sky130 (seq path is PDK-independent), 4 SRAM blackboxes, 0 leftovers
# → 717MB Verilog, 456 modules, 0 always-blocks, parse clean.
# (1.97M vs 1.15M sky130 instances = MUX2-less decomposition + 681K INVs.)
./build/bin/circt-synth --liberty <a>,<b>,<c>,<d>,<e> --sram-macros tmp/sram-macros-chiptop.mlir --strip-lib-to-extern --top ChipTop /tmp/e2e/chiptop.mlir -o /tmp/e2e/chiptop_asap7.mlir
# Rocket → sky130: ~3 min, 266,566 instances, ~72k flops (54.8k mem + 15.1k reg, reconciled)
./build/bin/circt-verilog -D SYNTHESIS --ignore-unknown-modules --top=RocketSystem <Rocket.v> /tmp/plusarg_stub.sv -o /tmp/e2e/rocket.mlir
./build/bin/circt-synth --liberty <sky130.lib> --strip-lib-to-extern --top RocketSystem /tmp/e2e/rocket.mlir -o /tmp/e2e/rocket_synth.mlir
# Verilog out (NOTE: --export-verilog prints Verilog to stdout; -o gets passthrough MLIR):
./build/bin/circt-opt --export-verilog /tmp/e2e/rocket_synth.mlir -o /dev/null > /tmp/e2e/rocket_net.v
./build/bin/circt-verilog --parse-only /tmp/e2e/rocket_net.v   # 1.8M lines, 211 modules, 0 always-blocks: clean
```
ChipTop: `circt-verilog ... --top=ChipTop <536 gen-collateral/*.sv> stubs` → 0 LLHD, 61 firmems (~2M bits), 11.8k firregs. Root-caused 2026-09-14: no SRAM macros are provided, so `SeqMemMapper` exploded all 61 firmems to flops; the 16384x64 alone becomes ~1M flops + ~2M muxes (~5–10M AIG nodes in one module, serial passes). A single-module probe confirmed it (9+ min single-core, still running). Fix: `synth-seq-mem-mapper` now hard-errors above `max-fallback-bits` (default 65536; `circt-synth --max-mem-fallback-bits`). ChipTop now fails in ~1 s listing exactly 4 oversized memories (16384x64, 8192x64, 1024x136, 512x256); fifo (2Kb) and Rocket (≤32Kb single) still explode as before.
- **ChipTop COMPLETE end-to-end (2026-09-14 ~02:37): ~30 s total**
  (0.5 s seq-mem-map, 16 s lowering, 11 s tech-map, 1 s strip) → 177MB
  mapped MLIR: 1,149,473 instances, 226,157 flops (210Kb exploded mem +
  regs, reconciles), 4 SRAM macro blackboxes, 0 firmem/firreg/aig
  leftovers → 461MB Verilog (`circt-opt --export-verilog`, 23 s), 456
  modules, 0 always-blocks, `circt-verilog --parse-only` clean. Two
  blockers fell: (i) silent flop explosion (fail-fast + macros);
  (ii) `x & ~x` constant nodes from late covering with no trailing
  cleanup → "No matching cut" (TechMapper has no constant patterns) —
  fixed by trailing CSE+canonicalize in the optimization pipeline, plus
  poison-chain notes on the error.

## What's done (with measurements)
- llhd lowering gaps: sync-process mapper, init-drive stripper, pure-op cloning
  (`bitcast`/`array_get`), `--strip-init-drives` + `--strip-assert-like` flags.
- Liberty import: full sky130hd (347 modules) + ASAP7 AO/OA/SIMPLE/INVBUF/SEQ.
- Truth-table simulator handles 1-bit comb And/Or (unconverted lib bodies).
- Seq mapping: wide-flop + array-flop explosion, mem→flop fallback incl.
  masked writes with per-lane enables, SRAM-macro path preserved.
- Perf: permutation-aware NPN matching — patterns pre-enumerate all pin
  permutations at load (keyed by N-canonical tables), per-cut matching is
  negation-only canonicalization (both output polarities) + hash lookup.
  Replaced: 4-input NPN table, canonical-form memoization, semi-canonical
  prefilter (~470 lines deleted). Exactness preserved end-to-end (every
  binding mechanically simulated in assertion builds). Word-level
  (single-u64) transforms for n≤6 with closed-form var masks. fifo
  25 min-hang → ~2.6 s; ChipTop tech-map 6+ min → ~11 s (sky130).
  Selections differ benignly from pre-flip on symmetric cells (more
  same-class pin assignments explored, best-arrival wins; e.g. fifo
  mux2_4 1602→1 with AOI+INV absorbing) — flop counts exact everywhere
  (rocket 71965=71965, ASAP7 226235=226235).
- Correctness: phase-aware matching (negated NPN bindings materialize
  inverter cells with area/delay folded in; `TechMapper.cpp` match/rewrite).
- Robustness: pattern/cut cap raised 6→9 (`kMaxPatternInputs`). sky130
  unaffected (nothing wider; bound follows the library via
  `maxMatchedInputs`). ASAP7's ~21 wide AO/OA/AOI/OAI cells (7–9 inputs)
  map via offline witness tables (below).
- **9-input scaling**: multi-word (≤9 input) word-level swap/negate
  transforms (single-word fast path untouched for n≤6; differential
  unit-tested vs naive reference incl. the pos-5 cross-word case);
  `enumeratePatternEntries` shared by load-time indexing and offline
  generation; wide (7+) patterns use checked-in offline witness tables
  (`lib/Dialect/Synth/Transforms/WidePatternTables.{h,cpp}`, 15 ASAP7
  functions, 2835 key groups, ≤16 entries/key — sound: dropping only
  loses pin-diversity candidates, never completeness) with mechanical
  per-entry verification at load (corrupt data is fatal, never silent).
  ASAP7 ChipTop: 800+ wide instances across all 15 functions (pin order
  agreement proven by hits), flops exact (226235), Verilog exports clean.
  Raising `max-cut-size-per-root` to 16 gains only 1.7x more wide matches
  for 4x time and MORE total instances (2.01M vs 1.96M) — wide cuts get
  pruned by design under timing priority; default settings stand.
- Fail-fast memory fallback: `synth-seq-mem-mapper{max-fallback-bits=65536}`
  (default 64Kb; `circt-synth --max-mem-fallback-bits`) — memories over
  the limit with no macro match are a hard error naming geometry, bits,
  and the fix (was: silent explosion → 76-min ChipTop hang). fifo (2Kb)
  and Rocket (≤32Kb single) still explode as before.
- SRAM macro replacement: `circt-synth --sram-macros=<file.mlir>` splices
  `hw.module.extern` + `#synth.sram.macro` externs into the design for
  `SeqMemMapper` (design symbols win; warns on empty files). ChipTop's 4
  oversized SRAMs covered by `tmp/sram-macros-chiptop.mlir` (placeholder
  names, not PDK cells).

## Open problems (for relaunch)
1. **Symmetry-pruned exact NPN is BROKEN — reverted, do not reapply blindly.**
   Attempted: pairwise-symmetry union-find per negation mask + skip
   permutations violating class order (claimed exact via coset argument).
   Result: `tech-mapper.mlir` + `tech-mapper-phase.mlir` failed with
   "No matching cut found" on previously-matching cuts. The coset proof
   must have a hole (prime suspect: static symmetries vs. the dynamic
   prefix context, or the skip predicate direction). Superseded by the
   permutation-aware matching (no per-cut permutation search at all).
   The Lazy Man's paper (§ cut rewriting / choice networks) remains useful
   background; also compare symmetry analysis plus canonicalization.
2. **Per-pattern NPN re-matching** is obsolete as a scaling wall: the
   permutation-aware matching (see What's done) moved the factorial to
   load time; per-cut matching is O(2^n) + hash lookups.
- **Supergates** (multi-level collapse, incl. phase-flipped variants with
  an inner inverter): `synth-gen-supergates` pass (opt-in via
  `circt-synth --gen-supergates`), ported from `ueno/dev/hidetou/tech-lib`
  and adapted to N-canon matching (raw-table dedup keeping pin diversity;
  compose-first generation builds only dedup winners; per-(pattern,key)
  entry cap 8 bounds matching). Supergates flow into TechMapper as
  ordinary patterns (mapping_cost); instance-aware truth tables support
  their bodies. fifo sky130: 15,641 → 9,927 instances (37% fewer, 377
  supergate uses across 35 distinct) in 9.8 s (vs 2.6 s). ChipTop sky130
  with supergates: 1,433,470 → 1,280,895 instances (11% fewer, 172K
  supergate uses across 1547 distinct, flops exact), 0.7 s generation +
  108 s tech-map; Verilog exports clean (6059 modules).
3. **SDC emission** (`write_sdc` equivalent) untouched; `constraint.sdc`
   files are flow inputs. Now the TOP remaining flow gap: netlist + SDC
   are the stated goal and only the netlist exists.
4. Tie constants emit as Verilog literals (valid; reference parity = `conb_1`).
5. `lpflow_isobufsrc` picked as generic buffers (~10k in fifo ASAP7 run) —
   functionally fine, flow dont-uses some lpflow cells; QoR question.
6. Upstream PRs #10193 (area-flow, cherry-picked+squashed) and #11106
   (RegOfVecToMem masked writes, cherry-picked) are OPEN upstream —
   rebase/re-cherry-pick if they land.
7. `circt-verilog` covers single-read memories only in RegOfVecToMem;
   multi-read/write arrays fall back to synth-side flop explosion
   (correct; macros unreachable for them). See `rf_31x64`, `head_*`,
   `tail_*`, `regfile_*` modules.
8. **Wide cells (7–9 inputs): DONE.** `kMaxPatternInputs` 6→9 with the
   bound following the library (`maxMatchedInputs`, so sky130 flows are
   untouched); multi-word transforms; offline witness tables for all 15
   distinct ASAP7 wide functions with load-time mechanical verification;
   800+ wide instances on ChipTop ASAP7 at default settings. Cut-retention
   tuning (`max-cut-size-per-root` 16) explored and rejected: 1.7x more
   wide matches for 4x time and higher totals.

## Key files
- `lib/Support/TruthTable.cpp` (word transforms, `computeNCanonicalForm`),
  `include/circt/Support/TruthTable.h`
- `lib/Dialect/Synth/Transforms/TechMapper.cpp` (phase-aware
  match/rewrite, inverter selection, `kMaxPatternInputs`)
- `lib/Dialect/Synth/Transforms/CutRewriter.cpp` (N-canon pattern map,
  `makeNPNBinding` + debug `verifyNPNBinding`, offline loader,
  instance-aware truth tables)
- `lib/Dialect/Synth/Transforms/GenSupergates.cpp` (supergate generation
  with phase-flipped variants)
- `lib/Dialect/Synth/Transforms/WidePatternTables.{h,cpp}` (offline
  witness data + accessor)
- `lib/Dialect/Synth/Transforms/SeqMemMapper.cpp` (`explodeToFlops` masked)
- `lib/Dialect/Synth/Transforms/SeqTechMapper.cpp` (wide/array explode)
- `lib/Dialect/LLHD/Transforms/LowerSyncProcesses.cpp`,
  `StripInitDrives.cpp`, `lib/Dialect/HW/Transforms/HWAggregateToComb.cpp`
  (array_inject O(N) lowering)
- Tests: `test/Dialect/Synth/tech-mapper-phase.mlir` (new),
  `test/Dialect/LLHD/Transforms/strip-init-drives.mlir` (new),
  `test/Dialect/Synth/seq-mem-mapper-fallback-limit{,-ok}.mlir` (new),
  `test/circt-synth/sram-macros.mlir` +
  `test/circt-synth/Inputs/sram-macros.mlir` (new). Flip covered by
  binding self-verification (active in assertion builds) + E2E counts
  (flops exact everywhere) + full lit suites.
- Tests for 9-input scaling: `test/Dialect/Synth/tech-mapper-wide-skip.mlir`
  (wide pattern without offline table is skipped, maps via smaller cells),
  `NCanonicalFormTest.MatchesNaiveReference` (randomized differential to
  n=9), `BinaryTruthTableTest.MultiWordPermutation` (every swap position
  incl. cross-word pos=5 + random perms, n=7..9).
- Tests for supergates: `test/Dialect/Synth/gen-supergates.mlir`
  (generation incl. phase-flipped inner_inv, budgets, duplicate-input
  merging), `test/Dialect/Synth/tech-mapper-supergates.mlir`
  (TechMapper consumes generated supergates as ordinary patterns).

## Suggested reading for relaunch
- https://scispace.com/pdf/lazy-man-s-logic-synthesis-43gevit82r.pdf
  (Mishchenko et al., Lazy Man's Logic Synthesis): choice networks,
  priority cuts, area-flow — direct background to our CutRewriter,
  the cherry-picked area-flow reselection, and any supergate work.
