# Multi-output cell mapping (emap-style)

## Goal

Map adder structures to multi-output library cells (`fa_*`, `ha_*` on
sky130; `FAx1_*` on ASAP7) instead of discrete gates. Today these cells
are present in both PDKs but usage is exactly zero: the mapper rejects
multi-output patterns upstream.

## Why: measured opportunity (sky130, current flow, authoritative counts)

| Design | Bits | Instances | Cells/bit | FA ideal | Tax |
|---|---|---|---|---|---|
| `add32` (`a+b+cin`) | 32 | 494 | 15.4 | ~1 + carry | ~15x |
| `inc16` (enabled counter) | 16 | 116 | 7.3 | ~1.5–2 | ~4x |
| fifo `addr_gen` counter | 8 | 67 (8 flops + 59 comb) | 7.4 comb | ~1.5–2 | ~4x |

Every ALU/AGU/counter bit in Rocket/ChipTop pays this tax. Microbenchmarks
live in `/tmp/e2e/momap/` (adders.v + synth + resource reports); rerun with
`circt-verilog`/`circt-synth` per status.md recipes.

Caveats: the flow runs timing-driven mapping, and discrete gates can win on
critical paths, so FA use must be cost-driven, not blanket. Datapath
lowering deliberately builds compressor trees for multipliers; extraction
must target what that flow leaves behind (ripple chains, counters, final
adders), not fight it.

## Reference: mockturtle `emap` (`mockturtle/.../algorithms/emap.hpp`)

Extended technology mapper; multi-output is opt-in (`map_multioutput`,
default off) and, per its own comment, effectively restricted to half/full
adders:

- Enumerate ≤3-leaf cuts per node (`max_multioutput_cut_size = 3`,
  `max_multioutput_output_size = 2`).
- Hash cuts by leaf set; nodes sharing a support form candidate pairs.
- Look up the truth-table pair in the library multi-function table
  (`get_multi_function_id`): XOR3+MAJ3 on a shared triple → full adder,
  XOR2+AND2 on a shared pair → half adder.
- Cover extraction is sharing-aware: joint area accounting for the pair
  plus "remove overlapping multi-output cuts" so shared logic is not
  double-covered. Stats report `multioutput_gates` separately.

The sibling `extract_adders` takes the other route: structural recovery of
adder chains pre-mapping instead of pair matching during covering.

## CIRCT gap analysis

- Pattern side: `CutRewriter::run` hard-errors on patterns with >1 output
  (`CutRewriter.cpp`, "does not support patterns with multiple outputs
  yet"); truth-table simulation (`getTruthTableImpl`), NPN verification
  (`verifyNPNBinding`), and cut/pattern compatibility all assume
  `numOutputs == 1`.
- Network side: mapping works on single-output AIG nodes; rewrite
  materializes one instance per root (`TechMapper.cpp`).
- Dialect: no half/full-adder ops exist in `synth`/`comb`.
- Selection: area-flow/delay accounting assumes one gate per cover node;
  a shared 2-output gate needs joint accounting or double-counting.

## Recommended approach: extraction first (B), pair matching later (A)

**Phase B — `extract_adders`-style pre-pass (small blast radius).**
Detect XOR3/MAJ3 (and XOR2/AND2) on shared support in the post-optimization
AIG, replace with new `synth.full_adder` / `synth.half_adder` ops (3→2 and
2→2 outputs), and map those ops 1:1 to fa/ha cells the way `SeqTechMapper`
maps flops 1:1 today: cut enumeration treats them as barriers (roots and
leaves of the mapping network, never cut through), and TechMapper gets a
direct rule instead of an NPN pattern. No cover-extraction changes, no
sharing accounting: each adder bit is one gate by construction.

Concretely:
1. `SynthOps.td`/`SynthOps.cpp`: `FullAdderOp` (`(i1,i1,i1)->(sum:i1,
   carry:i1)`), `HalfAdderOp` (`(i1,i1)->(sum,carry)`), pure, with
   folders/verifier.
2. `ExtractAdders.cpp` (`synth-extract-adders`, ModuleOp): run after the
   optimization pipeline, before mapping. For each 3-input XOR node, check
   whether the same triple feeds a majority (3×AND2 on the input pairs +
   combining AND, modulo phases — mirror mockturtle's complemented-edge
   handling); same for 2-input XOR+AND pairs. Rewrite to adder ops.
   Statistics: `numFullAdders`, `numHalfAdders`.
3. `TechMapper.cpp` (+ CutRewriter network build): barrier treatment +
   direct cell rule keyed off `mapping_cost` of the fa/ha lib modules
   (works for any PDK exposing them; sky130 `fa_1`, ASAP7 `FAx1_*`).
4. Tests: `test/Dialect/Synth/extract-adders.mlir` (structural incl.
   phases), `tech-mapper-adders.mlir` (fa cell selected, both outputs
   wired), LEC old-vs-new on the microbenchmarks (`circt-lec`), STA
   scoreboard (`tmp/sta_scoreboard.py`) for delay parity, flop-count
   invariance.
5. `circt-synth.cpp`: run extraction as part of the mapping stage
   (flag-gated at first: `--disable-adder-extraction` default-off amongst
   `disable-*`? or opt-in `--enable-...`? House style has both; recommend
   opt-in until STA parity is demonstrated, then default-on).

**Phase A — emap-style pair matching (only if B leaves general
multi-output cells on the table).** Requires: 2-output pattern loading
(truth tables per output), pair enumeration over shared ≤3-leaf supports,
joint area/delay in the CutRewriter cost model, overlap-free cover
extraction, and 2-output rewrite in TechMapper. Strictly harder than B
for the same HA/FA payoff; justified only by non-adder multi-output cells
(which neither PDK has in practice — audit before starting).

## Open questions

- Interaction with datapath compressor trees: extraction placement (after
  datapath lowering? only outside multiplier trees?) needs experiments on
  Rocket/ChipTop, guided by the STA scoreboard, not instance counts.
- Selection policy: FA-on-critical-path vs discrete gates; needs load-aware
  delay (task 8) for a fair fight.
- Supergate interplay (task 10): FA cells as supergate leaves.
- ASAP7 FA cell timing/area audit (`FAx1_*` drive variants).
- Bit-level naming through extraction for STA path stability.
