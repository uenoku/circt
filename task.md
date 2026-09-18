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
 3. Yosys baselines — fifo DONE (fresh reproduce: 3,447 cells /
    23.839ns vs recorded 3,478/23.84): Yosys/ABC beats our baseline on
    both axes (2.6x area, 1.5x delay); our buffered (9,256/4.45ns)
    beats Yosys 5.4x on delay at 2.7x cells. Yosys worst path is ALSO
    broadcast-launched (17.7ns flop-Q stage) — ABC suffers the same
    disease, milder (tighter selection → smaller decoder forest).
    add32: Yosys 226/6.65ns vs ours 464/1.82ns (load-aware) — ABC wins
    area 2x with no FAs on either side; we win delay 3.6x; ABC -D is a
    no-op in this flow (identical md5). inc16: Yosys 70/1.75ns vs ours
    116/1.83ns — delay parity, ABC area 1.7x. Remaining area gaps
    (fifo 2.6x, add32 2x, inc16 1.7x) point at ABC rewriting/balancing
    + mux4 (fifo) as the next selection/structure work: mux census
    (sky130hd fifo) is ours 2228 mux2i_1 + 17 mux2i_4 + 336 mux4_2
    vs Yosys 4 mux2_1 + 36 mux2i_1 + 575 mux4_2 — ABC collapses the
    decoder forest ~4x into 4:1s (mux4_2 area 22.5 beats 3x mux2i_1
    30.0). Our matcher DOES pick mux4_2 on clean 4:1 trees (probed),
    so the gap is pre-map shape/restructuring (shared selects, phases,
    interleaved logic defeat 6-input collapse), not matching power —
    next direction is mux restructuring/SOP-balancing evaluation on
    forests: PROBED --enable-sop-balancing on fifo → 9705/+7.8% area,
    37.08ns (mux4_2 336→1, drive-4 mux2s 0→454) — SOP balancing
    anti-collapses (spreads into 2-input shapes); mux collapse needs
    dedicated mux-tree restructuring, not SOP balancing. Scoped next
    direction (not started): a pre-map mux-restructuring pass that
    reassociates shared-select mux trees into collapsible 4:1 cones
    (phase-normalized selects via existing negation machinery,
    forest-local, STA-judged on fifo decoder; ~4x mux-count prize).
    Reference
    nets/logs in /tmp/e2e/momap (yosys_*_fresh/fifo/add32/inc16).
    Buffered ours is now 9,256 / 4.45ns
    (better than Yosys/ABC on delay, 2.7x on area — the gap is closed
    on timing, open on area). ChipTop ORFS run outstanding: no OpenROAD
    binary installed (no docker image; source build is hours). ORFS
    groundwork landed (docs/ORFSBringup.md): fifo needs no SRAM macros
    (flop-array memories) so it is the first target; netlist + emitted
    SDC regenerate from flags, SDC verified loading in STA (all MET).
    ChipTop ORFS additionally needs item-14 macros.
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

 8. Load-aware delay — STAGE 1 LANDED (opt-in --load-aware-delay,
    default off; docs/TimingDrivenMapping.md): sensitivity fit in
    DeriveMappingCost (least-squares dDelay/dLoad over NLDM load
    columns at mid-slew; index_2 = load on both PDKs) + match-time
    delay=intrinsic+min(sens×load, intrinsic), load=min(fanout,64)
    × mean input cap, trusted-fanout gate ≤8 (pointer-free,
    deterministic; ChipTop-scale determinism inherited from flow).
    Zero default regression proven (fifo control 9003/36.802ns
    identical). Measured opt-in: add32 494→464 (-6%), 2.26→1.82ns
    (-20%); fifo 9003→8572 (-4.8%) BUT worst path 36.8→87ns
    (subject-fanout ≠ cover load + arrival-only blindness; needs
    Stage 2 cover-iteration + backward criticality pass — scoped
    future work, stays opt-in until then). ChipTop opt-in: 2080890→
    1989878 (-4.4%, -91K, flops exact 422417); the win is mechanistic
    (mux2i_1→mux2i_4 drive shift -455K/+347K on loaded nets + broad
    reselection: 56 cells down, 88 up) BUT delay regresses on both
    axes (sync out_arb 6.78→10.32ns, async in_demux +3.22→-3.12) —
    drive-strength selection responds to loads yet worst paths lose
    without required-time gating. fifo combos: load-aware+buffers =
    8752/8.21ns (buffers fix the 87ns pathology; buf-only 9256/4.45ns
    still best delay — flags compose safely, trade-offs differ).
    ChipTop combo (LA+buffers): 2080890→2050077 (-1.5%, -31K, flops
    exact, 60K bufs); worst -3.12 same async path (buffers can't fix
    non-broadcast paths; LA delay cost persists, LA area win partly
    kept; buf-only still best delay). Opt-ins compose safely at scale.
    Tests: fitted-sensitivity case + area/timing split + flag-off control.
 9. Buffer insertion — LANDED v1 (opt-in --buffer-high-fanout,
    default off; --max-fanout default 32; docs/TimingDrivenMapping.md):
    `synth-buffer-high-fanout` (post-map, pre-strip): per-copy fanout
    through hierarchy (local chunking + driver-to-port buffers for
    through-port forests); auto-detected identity buffer cell (buf_1);
    clock-pin nets never touched; tie drivers skipped; deterministic
    (A/B identical); 1m23s on ChipTop vs 45s baseline. Measured:
    fifo 9003/36.802ns → 9256/4.451ns (8.3×, +253 bufs, flops exact,
    MET); add32 untouched (0 bufs); ChipTop +3.0% area, worst slack
    +3.22→+3.36 MET (neutral-positive). Follow-ups: trees/chains,
    load-sizing (needs 8/Stage-2), sync markers, default-on decision.
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
 15. RegOfVecToMem multi-read/write arrays — CLOSED (no gap on current
    designs): census of /tmp/e2e/chiptop.mlir finds 9 array-typed
    firregs, all in top ChipTop, 7,836 bits total (1.9% of 422K flops),
    depths 2–52 (LUTRAM-scale; no SRAM macro fits 2-deep arrays, and
    multi-port read-during-write semantics are delicate). Upstream
    RegOfVecToMem already extracted every single-read array (the rest
    of memory is seq.firmem); the 9 leftovers are flop-appropriate even
    under a perfect multi-read extension (firmem→SeqMemMapper→flop
    fallback without a fitting macro = same flops). Revisit only if a
    design shows large multi-read arrays.
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

 Implementation program complete. Scoreboard/counting/dedup/tie/SDC/
 bundles/load-aware/buffers done; 6/7/10/11/12/15/16/17 closed by
 measurement or implementation. Remaining tracks are external/
 background: Yosys/ORFS parity (3, ChipTop ORFS is hours), LEC
 revival (4, skipped per user), PDK SRAM compilation (14). Item 8
 stays opt-in pending Stage-2/criticality; item 9 stays opt-in
 pending trees/sizing/more designs.
