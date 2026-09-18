 Here's the full remaining list:

 Measurement (gates everything else)

 1. STA scoreboard — DONE (runner: tmp/sta_scoreboard.py, ns-normalized,
    multi-endpoint). fifo sky130 baseline/supergates/wide16 + ASAP7 baseline;
    ChipTop sky130 baseline + dedup/tie A/B (2.99→3.22ns, neutral-to-better).
    Unblocked by `disallowReplicate` (no Yosys round-trip). Supergates+ASAP7
    synthesis errors (missing __supergate_NNNN) — open.
 2. Authoritative counting — DONE: hw.instance per cell, flattening-
    equivalent. ChipTop 2,080,760 flat vs 1,229,085 textual; <unknown> is
    wiring only, zero unmapped gates.
 3. Yosys baselines — fifo DONE: Yosys/ABC 3,478 cells / 23.84ns vs ours
    9,003 / 36.80ns (logic ~5x gap: ABC delay mapping + mux4). Reference
    nets/logs in /tmp/e2e/momap. ChipTop ORFS run outstanding (hours; needs
    ORFS checkout + PDK SRAM collateral per item 14).
 4. LEC validation — SKIPPED per user (hard-blocked, findings recorded):
    needs unstripped MLIR + --verify-each=false, then blocked on synth.ff
    legalization in circt-lec. `synth-sort-topologically` kept for revival
    (off by default: 5 min on ChipTop-scale blocks).

 Inverters (20%/34% of instances, 7%/14% of area)

 5. Post-map inverter dedup — DONE: `synth-dedup-instances` (leaf comb cells
    only; flops/hierarchy/SRAMs never merge; --disable-dedup). fifo −12.4%,
    ChipTop −13.5% (clkinv −42%), flops exact, fifo STA bit-identical.
 6. ASAP7 MUX check — DONE: no MUX cell in any 7p5t flavor/corner/release
    (AO/OA/SIMPLE/INVBUF/SEQ only). MUX2-less decomposition is inherent.
 7. Inversion-absorbing selection — DONE (verified): negated terminals fold
    a full inverter area+delay into match cost; NAND/NOR/AOI/OAI win
    automatically.

 Timing-driven (needs STA first)

 8. Load-aware delay — DESIGNED (docs/TimingDrivenMapping.md): sensitivity
    fit in DeriveMappingCost + match-time delay=intrinsic+sensitivity×load,
    then cover-iterated loads. Attr fields already carried for this.
 9. Buffer insertion — DESIGNED (same doc): post-map clkbuf on
    total-cap-ranked nets, STA-closed threshold. Needs 8 + SDC first.
 10. Supergate tuning — DONE (measured): maxGates=3 default confirmed
    (sg3 = baseline area, −1.15ns; sg2 +38 inst same delay); per-key cap
    stays 8 (cap4+sg −190 inst but base+cap4 +35 — thin evidence, keep
    safe default). Both knobs now CLI-tunable (--supergate-max-gates,
    --max-entries-per-key).

 Area

 11. Tie cells — DONE: `synth-tie-cells` (auto-detect 0-input
    function-"1"/"0" liberty cells, shared instance per polarity per
    module). ChipTop 164 literals → 89 conb_1 + 4 glue assigns; STA clean.
 12. Stronger pre-map constprop — CLOSED (no gap): 41 pretech constants,
    all consumed by mapping (0 leftovers, 0 No-matching-cut); residual is
    glue/regs. Revisit only if matching failures reappear.

 Flow completeness

 13. SDC emission — IMPLEMENTED (v1): `synth-emit-sdc` pass + --sdc-output
    --sdc-clock-period/--sdc-input-delay/--sdc-output-delay/
    --sdc-clock-gates/--sdc-clock-sources flags. Exact hierarchical descent
    (follows real driver nets, not name pairing); EICG gate rule;
    FixedClock* source recording; shared-context unions reported, not
    constrained. fifo: 1 clock, STA-equivalent to hand TCL. ChipTop: 1
    certain clock (clock_uncore, 10.6K flops) + 4 source pins + 8 shared
    candidates; emitted SDC loads in STA. See docs/SdcEmission.md.
 14. Real PDK SRAM names — PATH FOUND: ASAP7 compiler collateral exists
    (asap7_sram_0p0/generated); needs exact-geometry compilation for
    16384x64/8192x64/1024x136/512x256 (pre-generated banks are small).
    sky130 platform has no SRAM macros visible. ORFS-integration work.
 15. RegOfVecToMem multi-read/write arrays — SCOPED: synth-side
    SeqMemMapper already matches N-port macros; gap is solely the
    `seq.regofvectomem` import pass (single-read). Bounded, delicate
    (read-during-write/write-ordering semantics). Not started.
 16. lpflow buffer selection — DONE (measured + explained): the 30 sites
    are and2b-function (A & ~SLEEP), not buffers; isobufsrc_1 (6.26/339ps)
    correctly beats dedicated and2b_1 (7.51/419ps); dont-use substitutes
    nor2b_1 neutrally. No mapping bug; isolation-cell handling is an ORFS
    integration concern.

 Multi-output mapping (see docs/MultiOutputMapping.md)

 17. emap-style FA/HA mapping — DONE v1 (pass-through bundle design,
    supersedes full_adder/half_adder ops): `synth.create_bundle` op +
    `synth-extract-bundles` (CutEnumerator k=3, AND2/XOR2/MAJ3/XOR3 IDs,
    leaf-set pairing, opt-in `--extract-bundles`) + `synth-map-bundles`
    (fresh cone simulation for joint tables — no persistent cut state,
    survives intervening passes; cheapest-area cell wins; free erase
    fallback; flops/hierarchy/SRAM untouched). Determinism: index-based
    + discovery-order canonicalization only (pointer sorts caused ASLR
    run-to-run variance; verified bit-identical ChipTop A/B). Measured
    (same binary, flattening counts): add32 494→411 (-17%, 47 ha_1,
    delay 2.26→2.40ns +6% area/delay tradeoff of area-only selection);
    fifo 9003→8988 (-15, 2 mapped); ChipTop 2080890→2077933 (-2957,
    -0.14%, 9261 ha_1 flattened / 6320 textual, flops exact 422417).
    Compile tax +5s on ChipTop (50s vs 45s). Delay caveat: selection is
    area-only and load-blind; ChipTop worst reported path moved
    +3.22 MET (out_arb, sync) to -22.73 on an async-crossing in_demux
    path with a 27ns single-stage load pathology and zero ha cells on
    it (TechMapper reselection variance, not bundle cells) — sync
    parity needs SDC clock groups + item 8. Stays opt-in until item 8
    (load-aware delay) lands. Remaining: negated-phase pairs (mostly
    erased today: 9070/14972 on ChipTop), FA coverage (0-1 bundles),
    MFFC economics.

 Suggested order (updated)

 Scoreboard/counting/dedup/tie/SDC/bundles done. Remaining implementation order:
 load-aware (8) → buffers (9)
 → multi-RW (15), with Yosys/ORFS parity (3) and LEC revival (4) as
 background tracks. Items 6/7/10/11/12/16/17 closed; 14 needs PDK/ORFS time.
