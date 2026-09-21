// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-report=-})' | FileCheck %s --check-prefix=REPORT

// Hierarchical timing: the datapath region lives in child @alu while the
// probe sits in parent @top. Hierarchical summaries connect them: the
// region reaches the parent probe with structural slack 0 (critical in
// both directions across the instance edge) and is selected.
// REPORT: {"schema": "circt.synth.datapath-refinement-report.v1", "regions": [{"id": "@alu:0", "module": "alu", "policy": "reduce-delay", "reason": "structural pressure", "violated": false, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@top:1"], "endpoints": []}]
hw.module @alu(in %a : i8, in %b : i8, in %c : i8, out s : i8) {
  %add = comb.add %a, %b, %c : i8
  hw.output %add : i8
}
hw.module @fd1(in %D : i1, in %CLK : i1 {synth.liberty.pin = {clock = "true"}}, out Q : i1) attributes {synth.liberty.cell = {area = 6.0 : f64}} {
  %q, %qn = synth.ff %D clock %CLK edge rise : i1, i1
  hw.output %q : i1
}
hw.module @top(in %clk : i1, in %a : i8, in %b : i8, in %c : i8, in %d : i1, out q : i1, out s : i8) {
  %sum = hw.instance "u_alu" @alu(a: %a: i8, b: %b: i8, c: %c: i8) -> (s: i8)
  %p = synth.timing_probe %sum : i8 {required_time = 10000 : i64}
  %q0 = hw.instance "f0" @fd1(D: %d: i1, CLK: %clk: i1) -> (Q: i1)
  hw.output %q0, %p : i1, i8
}
