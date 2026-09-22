// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-import-timing-feedback{timing-feedback=%S/Inputs/refine-feedback-import-v2.json feedback-report=-})' 2>&1 | FileCheck %s

// CHECK: warning: cannot resolve v2 feedback endpoint 'u_sub/f9/fd1:D[0]' (missing, skipped)
// Structured v2 import: endpoint keys resolve from the configured top with
// classified failures; probes are created/updated identically to v1.
// CHECK: {"schema": "circt.synth.timing-feedback-report.v1", "resolved": [{"endpoint": "u_sub/f0/fd1:D[0]", "required_ps": 8000, "action": "updated"}, {"endpoint": "u_sub/f1/fd1:D[0]", "required_ps": 9000, "action": "created"}], "unresolved": ["u_sub/f9/fd1:D[0]:missing"]}
// CHECK: %[[P:.+]] = synth.timing_probe %d : i1 {required_time = 8000 : i64}
// CHECK: hw.instance "f0" @fd1(D: %[[P]]: i1, CLK: %clk: i1)
// CHECK: %[[P:.+]] = synth.timing_probe %d : i1 {required_time = 9000 : i64}
// CHECK: hw.instance "f1" @fd1(D: %[[P]]: i1, CLK: %clk: i1)
hw.module @fd1(in %D : i1, in %CLK : i1 {synth.liberty.pin = {clock = "true"}}, out Q : i1) attributes {synth.liberty.cell = {area = 6.0 : f64}} {
  %q, %qn = synth.ff %D clock %CLK edge rise : i1, i1
  hw.output %q : i1
}
hw.module @sub(in %clk : i1, in %d : i1, out p : i1, out q : i1) {
  %t = synth.timing_probe %d : i1 {required_time = 10000 : i64}
  %q0 = hw.instance "f0" @fd1(D: %t: i1, CLK: %clk: i1) -> (Q: i1)
  %q1 = hw.instance "f1" @fd1(D: %d: i1, CLK: %clk: i1) -> (Q: i1)
  hw.output %q0, %q1 : i1, i1
}
hw.module @top(in %clk : i1, in %d : i1, out p : i1, out q : i1) {
  %p, %q = hw.instance "u_sub" @sub(clk: %clk: i1, d: %d: i1) -> (p: i1, q: i1)
  hw.output %p, %q : i1, i1
}
