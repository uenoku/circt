# circt-synth as Yosys replacement — status

## Goal
Make `circt-synth` a drop-in replacement for Yosys in
`circt-sfc-experiments/OpenROAD-flow-scripts/flow`:
a liberty-mapped netlist (`1_2_yosys.v` equivalent) plus SDC, for the
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
- ASAP7 LVT_FF set (assemble once):
  - `~/Downloads/asap7sc7p5t_{AO_SLVT→extract LVT,OA_LVT from as7p5t_27/LIB/NLDM,...}` —
    concretely: SIMPLE_LVT + INVBUF_LVT + AO_LVT + OA_LVT from
    `/home/uenoku/dev/asap7/asap7sc7p5t_27/LIB/NLDM/*.lib.7z` (extract with
    `7z x`), plus `~/Downloads/asap7sc7p5t_SEQ_LVT_FF_nldm_220123.lib`.
    INV/BUF live in a separate INVBUF lib (ORFS uses it too); no MUX2
    cells exist in these libs (muxes decompose — correct, verbose).

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
- Perf: 4-input NPN lookup table (`NPNTable` ctor, 222 representatives —
  4-input canonicalization is O(1), same as ABC's mapper tables) plus NPN
  canonical-form memoization (`NPNTable::getCanonicalForm`) — fifo NPN
  stats BEFORE the prefilter: 119307 hits / 592 misses. Word-level
  (single-u64) transforms for n≤6 with closed-form var masks (no per-call
  bit loops). fifo 25 min-hang → ~1–3 s; addr_gen 25 s → 0.95 s.
- **Semi-canonical NPN prefilter** (`BinaryTruthTable::getNPNSignature`,
  `CutRewritePatternSet::hasFamilyWithSignature`): cheap
  negation+permutation-invariant signature (arity, normalized weight,
  sorted normalized cofactor pairs) per cut; exact O(n!·2^n)
  canonicalization runs only on signature hits. fifo NPN stats AFTER:
  57024 hits / 117 misses (5x fewer exact searches). Exactness PRESERVED
  (unlike kitty's lossy flip-swap): fifo outputs byte-identical across 4
  binaries, rocket deterministic run-to-run. ChipTop tech-map: 6+ min
  (failing) → 11 s (succeeding) — combined with the constant-fold fix;
  NPN-stats + fifo timing are the clean attribution. ABC reference: same
  4-input tables, cofactor trick for n=5, exhaustive for n=6 (no better
  exact algorithm there either — Gray-code incremental enumeration
  evaluated and deferred at 11 s).
- Correctness: phase-aware matching (negated NPN bindings materialize
  inverter cells with area/delay folded in; `TechMapper.cpp` match/rewrite).
- Robustness: 6-input pattern/cut cap (`kMaxPatternInputs`,
  `TechMapper.cpp:254`) — 9-input AO333/OA333 skipped, logic covers
  through smaller cells (ABC's default cut size).
- Fail-fast memory fallback: `synth-seq-mem-mapper{max-fallback-bits=65536}`
  (default 64Kb; `circt-synth --max-mem-fallback-bits`) — memories over
  the limit with no macro match are a hard error naming geometry, bits,
  and the fix (was: silent explosion → 76-min ChipTop hang). fifo (2Kb)
  and Rocket (≤32Kb single) still explode as before.
- SRAM macro replacement: `circt-synth --sram-macros=<file.mlir>` splices
  `hw.module.extern` + `#synth.sram.macro` externs into the design for
  `SeqMemMapper` (design symbols win; warns on empty files). ChipTop's 4
  oversized SRAMs covered by `tmp/sram-macros-chiptop.mlir` (placeholder
  names, not PDK cells): with macros, pre-TechMapper stages drop from
  60+ min to ~16 s; TechMapper over the remaining ~210Kb of small flops
  is the current long pole (suspected NPN — see below).

## Open problems (for relaunch)
1. **Symmetry-pruned exact NPN is BROKEN — reverted, do not reapply blindly.**
   Attempted: pairwise-symmetry union-find per negation mask + skip
   permutations violating class order (claimed exact via coset argument).
   Result: `tech-mapper.mlir` + `tech-mapper-phase.mlir` failed with
   "No matching cut found" on previously-matching cuts. The coset proof
   must have a hole (prime suspect: static symmetries vs. the dynamic
   prefix context, or the skip predicate direction). Reverted to plain
   exhaustive search; suite green. Cache + word ops carry scaling for now.
   The Lazy Man's paper (§ cut rewriting / choice networks) is the right
   background before retrying; also compare ABC's `Abc_NpnSymmetries` +
   canonicalization approach.
2. **Per-pattern NPN re-matching** is the next scaling wall after
   canonicalization (~0.1 s/cut with 400-pattern libs observed earlier;
   mitigated by caps). Options: share one match per NPN class across
   same-function size variants; ABC supergates (`super_gates.c`) for
   multi-level collapse — NOT implemented (each lib cell = one pattern).
   UPDATE 2026-09-14: semi-canonical prefilter implemented (see What's
   done) — exact search now runs only on signature hits; per-pattern
   match cost itself (binding + arrival, O(n) over same-class drive
   variants) measured negligible. CLOSED unless profiles say otherwise.
3. **SDC emission** (`write_sdc` equivalent) untouched; `constraint.sdc`
   files are flow inputs. Now the TOP remaining flow gap: netlist + SDC
   are the stated goal and only the netlist exists.
4. Tie constants emit as Verilog literals (valid; Yosys parity = `conb_1`).
5. `lpflow_isobufsrc` picked as generic buffers (~10k in fifo ASAP7 run) —
   functionally fine, flow dont-uses some lpflow cells; QoR question.
6. Upstream PRs #10193 (area-flow, cherry-picked+squashed) and #11106
   (RegOfVecToMem masked writes, cherry-picked) are OPEN upstream —
   rebase/re-cherry-pick if they land.
7. `circt-verilog` covers single-read memories only in RegOfVecToMem;
   multi-read/write arrays fall back to synth-side flop explosion
   (correct; macros unreachable for them). See `rf_31x64`, `head_*`,
   `tail_*`, `regfile_*` modules.

## Key files
- `lib/Support/TruthTable.cpp` (`getCanonicalForm:540`, word transforms,
  cache stats `dumpCacheStats:554`), `include/circt/Support/TruthTable.h`
- `lib/Dialect/Synth/Transforms/TechMapper.cpp` (phase-aware
  match/rewrite ~117-200, inverter selection, `kMaxPatternInputs:254`)
- `lib/Dialect/Synth/Transforms/CutRewriter.cpp` (`Cut::getNPNClass:477`)
- `lib/Dialect/Synth/Transforms/SeqMemMapper.cpp` (`explodeToFlops` masked)
- `lib/Dialect/Synth/Transforms/SeqTechMapper.cpp` (wide/array explode)
- `lib/Dialect/LLHD/Transforms/LowerSyncProcesses.cpp`,
  `StripInitDrives.cpp`, `lib/Dialect/HW/Transforms/HWAggregateToComb.cpp`
  (array_inject O(N) lowering)
- Tests: `test/Dialect/Synth/tech-mapper-phase.mlir` (new),
  `test/Dialect/LLHD/Transforms/strip-init-drives.mlir` (new),
  `test/Dialect/Synth/seq-mem-mapper-fallback-limit{,-ok}.mlir` (new),
  `test/circt-synth/sram-macros.mlir` +
  `test/circt-synth/Inputs/sram-macros.mlir` (new). Prefilter + mask +
  pipeline-cleanup covered by output-identity (fifo byte-identical
  across 4 binaries; rocket deterministic) + full lit suites.

## Suggested reading for relaunch
- https://scispace.com/pdf/lazy-man-s-logic-synthesis-43gevit82r.pdf
  (Mishchenko et al., Lazy Man's Logic Synthesis): choice networks,
  priority cuts, area-flow — direct background to our CutRewriter,
  the cherry-picked area-flow reselection, and any supergate work.
