
 Here's the full remaining list:

 Measurement (gates everything else)

 1. STA scoreboard — DONE unblocked: new `disallowReplicate` lowering option
    makes ExportVerilog emit OpenSTA-parseable netlists directly (no Yosys
    round-trip). fifo matrix measured (sky130 baseline/supergates/wide16 +
    ASAP7 baseline, arrivals normalized to ns) and ChipTop sky130 baseline
    links + report_checks clean (worst arrival 25.13ns, 1 MET + 1 VIOLATED
    path @10ns). Scoreboard runner: tmp/sta_scoreboard.py. Supergates +
    ASAP7 synthesis itself errors (missing __supergate_NNNN module) — open.
 2. Authoritative counting — DONE: ResourceUsageAnalysis counts hw.instance
    per cell through the hierarchy (flattening-equivalent, no regexes).
    Validated on fifo (grep 10194 vs analysis 10277 = extra addr_gen copy)
    and ChipTop (1,899,760 flattened vs 1,149,473 textual). Remaining
    <unknown> bucket is wiring only (extract/concat/replicate), zero
    unmapped gates.
 3. Yosys baselines — fifo manual synth + ChipTop ORFS run for area/delay parity reference.
 4. LEC validation — old-vs-new / ours-vs-Yosys via circt-lec once clean netlists flow.

 Inverters (20%/34% of instances, 7%/14% of area)

 5. Post-map inverter dedup — ~60K/201K exact duplicates (30%), created after the last CSE; merge for ~5% instance reduction.
 6. ASAP7 MUX check — 34% INV is driven by MUX2-less decomposition; look for a mux cell in other Vt flavors/full libs.
 7. Inversion-absorbing selection — verify cost model already prefers NAND/NOR/AOI/OAI over AND/OR+INV.

 Timing-driven (needs STA first)

 8. Load-aware delay — matching uses intrinsic only; NLDM tables already parsed.
 9. Buffer insertion — nothing buffers high-fanout nets today.
 10. Supergate tuning — maxGates=3 default? per-key cap value? Measure via STA, not instance counts.

 Area

 11. Tie cells (conb_1) instead of Verilog literals.
 12. Stronger pre-map constprop (beyond x & ~x).

 Flow completeness

 13. SDC emission — top remaining gap (netlist exists, constraints don't).
 14. Real PDK SRAM names — placeholders work structurally; binding is later work.
 15. RegOfVecToMem multi-read/write arrays — still flop-explode; macros unreachable.
 16. lpflow buffer selection — functionally fine, QoR open.

 Suggested order

 STA scoreboard → inverter dedup → Yosys baselines + LEC → ASAP7 MUX check → SDC emission → timing-driven work (8–10) guided by real numbers. Items
 1–2 unblock everything downstream, which is why they're in flight.
