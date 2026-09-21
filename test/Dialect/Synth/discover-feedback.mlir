// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{pressure-margin-levels=-1 timing-feedback=%S/Inputs/refine-feedback-violated.json refinement-report=-})' | FileCheck %s
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{pressure-margin-levels=-1 refinement-policy=baseline refinement-report=-})' | FileCheck %s --check-prefix=NOFEEDBACK

// Feedback-driven selection with structural selection disabled
// (pressure-margin-levels=-1): the multiplier region feeding the violated
// endpoint u_alu/f0 is selected with the timer's required time; the relaxed
// adder region keeps the baseline. The unresolvable endpoint is reported.
// CHECK: {"id": "@alu:0", "module": "alu", "policy": "reduce-delay", "reason": "violating endpoint cone", "violated": true, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@alu:5"], "endpoints": ["u_alu/f0"]}, {"id": "@alu:7", "module": "alu", "policy": "baseline", "reason": "relaxed (no pressure)", "violated": false, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@alu:10"], "endpoints": []}
// NOFEEDBACK: {"id": "@alu:0", "module": "alu", "policy": "baseline", "reason": "relaxed (no pressure)", "violated": false, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@alu:5"], "endpoints": []}, {"id": "@alu:7", "module": "alu", "policy": "baseline", "reason": "relaxed (no pressure)", "violated": false, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@alu:10"], "endpoints": []}
hw.module @fd1(in %D : i1, in %CLK : i1 {synth.liberty.pin = {clock = "true"}}, out Q : i1) attributes {synth.liberty.cell = {area = 6.0 : f64}} {
  %q, %qn = synth.ff %D clock %CLK edge rise : i1, i1
  hw.output %q : i1
}
hw.module @alu(in %clk : i1, in %a : i8, in %b : i8, in %c : i8, out x : i1, out y : i1) {
  %m = comb.mul %a, %b : i8
  %e0 = comb.extract %m from 0 : (i8) -> i1
  %pe = synth.timing_probe %e0 : i1 {required_time = 10000 : i64}
  %f0 = hw.instance "f0" @fd1(D: %pe: i1, CLK: %clk: i1) -> (Q: i1)
  %s = comb.add %a, %b, %c : i8
  %e1 = comb.extract %s from 7 : (i8) -> i1
  %ps = synth.timing_probe %e1 : i1 {required_time = 10000 : i64}
  %f1 = hw.instance "f1" @fd1(D: %ps: i1, CLK: %clk: i1) -> (Q: i1)
  hw.output %f0, %f1 : i1, i1
}
hw.module @top(in %clk : i1, in %a : i8, in %b : i8, in %c : i8, out x : i1, out y : i1) {
  %x, %y = hw.instance "u_alu" @alu(clk: %clk: i1, a: %a: i8, b: %b: i8, c: %c: i8) -> (x: i1, y: i1)
  hw.output %x, %y : i1, i1
}
