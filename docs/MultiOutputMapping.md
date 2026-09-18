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

## Recommended approach: pass-through bundles (supersedes adder-specific ops)

An earlier revision of this doc proposed dedicated `synth.full_adder` /
`synth.half_adder` ops with 1:1 barrier mapping. That was set aside: a
generic pass-through grouping op covers adders and any future
multi-output family with one mechanism, and fallback is free erase
instead of a lowering. Concretely (`synth.create_bundle`, landed):

```mlir
%o0, %o1 = synth.create_bundle %sum, %carry : (i1, i1) -> (i1, i1)
```

Pure N-in/N-out grouping, result `i` equals operand `i`, pairwise same
types (verifier-enforced). Carries no function data; the joint function
is derived by simulating producer cones at match time (ground truth),
and no canonicalization may forward through it before mapping.

1. **Extraction** (`synth-extract-bundles`, ModuleOp, opt-in
   `--extract-bundles`): reuse `CutEnumerator` (k=3) + cut truth tables +
   N-canonical AND2/XOR2/MAJ3/XOR3 IDs + leaf-set hashing over shared
   support (mockturtle `extract_adders` structure, adapted). Propose
   bundles only; map disposes. Statistics double as the opportunity
   measurement. MFFC/overlap filtering: skip already-bundled roots (v1).
2. **Joint matching**: brute-force joint NPN (input perm/neg + output
   perm/neg; 384 combos for FA) against multi-output library cells,
   reusing the truth-table simulator. Single-output flow untouched.
3. **Value-based rewrite**: replace uses of the bundled values with the
   cell outputs (not op-based like cut rewriting); DCE cleans producers.
   Unmatched bundles are erased (discrete logic stays) — never leak
   downstream (fail-fast if any survive mapping).
4. Tests: extraction detection (structural incl. phases) +
   `tech-mapper-bundles.mlir` (fa cell selected, both outputs wired) +
   adder microbenchmarks (`/tmp/e2e/momap`) via STA scoreboard.
5. Cut enumeration treats bundles as barriers (roots/leaves only, never
   cut through) — same treatment flops already get.

Extraction + joint matching land as one atomic unit: unverified
extraction is untestable, so they are not split across commits.


## v1 results (landed, opt-in `--extract-bundles`)

- Passes: `synth-extract-bundles` + `synth-map-bundles` (in mapping
  section when the flag is on; MapBundles runs standalone in tests).
- Joint tables by fresh cone simulation at match time (no persistent cut
  state — survives intervening passes; combinational-only domain:
  `comb`, `hw.constant`, `synth.choice`, single-output instances).
  Cheapest-area approval (`synth.liberty.cell` or `mapping_cost`, minus
  `dont_use`); free erase fallback; flops/hierarchy/SRAM untouched.
- Determinism: index-based + discovery-order canonicalization only.
  Pointer-order sorts caused ASLR run-to-run variance (caught by A/B
  md5); ChipTop A/B now bit-identical.
- Measured (same binary, authoritative flattening counts): add32
  494→411 (-17%, 47 ha_1, delay 2.26→2.40ns); fifo 9003→8988 (-15);
  inc16 0 bundles (counter structures out of scope); ChipTop
  2080890→2077933 (-2957, -0.14%, 9261 ha_1 flattened, flops exact
  422417). Compile tax +5s on ChipTop. Extraction finds 14972 ChipTop
  bundles (1 FA + 14971 HA); 5902 map, rest erased (negated phases etc).
- Delay caveat: selection is area-only and load-blind; stays opt-in
  until load-aware delay (task 8). ChipTop worst reported path moved
  +3.22 MET to -22.73 on an async-crossing path with a 27ns single-stage
  load pathology and no bundle cells on it (TechMapper reselection
  variance); sync parity needs SDC clock groups + item 8.
- Next: negated-phase pairs, FA coverage, MFFC economics.

## Open questions

- Interaction with datapath compressor trees: extraction placement (after
  datapath lowering? only outside multiplier trees?) needs experiments on
  Rocket/ChipTop, guided by the STA scoreboard, not instance counts.
- Selection policy: FA-on-critical-path vs discrete gates; needs load-aware
  delay (task 8) for a fair fight.
- Supergate interplay (task 10): FA cells as supergate leaves.
- ASAP7 FA cell timing/area audit (`FAx1_*` drive variants).
- Bit-level naming through extraction for STA path stability.
