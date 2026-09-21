// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-import-timing-feedback{timing-feedback=%S/Inputs/refine-feedback-import.json feedback-report=-})' 2>&1 | FileCheck %s
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-import-timing-feedback{timing-feedback=%S/Inputs/refine-feedback-legacy.json})' 2>&1 | FileCheck %s --check-prefix=LEGACY
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-import-timing-feedback{timing-feedback=%S/Inputs/refine-feedback-bad.json})' 2>&1 | FileCheck %s --check-prefix=PERMISSIVE
// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-import-timing-feedback{timing-feedback=%S/Inputs/refine-feedback-bad.json timing-feedback-strict})' 2>&1 | FileCheck %s --check-prefix=STRICT
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-import-timing-feedback{timing-feedback=%S/Inputs/refine-feedback-units.json top-module-name=top})' 2>&1 | FileCheck %s --check-prefix=UNITS
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-import-timing-feedback{timing-feedback=%S/Inputs/refine-feedback-units.json top-module-name=other})' 2>&1 | FileCheck %s --check-prefix=TOPMISMATCH

// The versioned record tightens the period probe (10000 -> 8000ps) and the
// legacy record creates a probe where none existed. Bad records warn
// (permissive) or fail (strict).
// CHECK: {"schema": "circt.synth.timing-feedback-report.v1", "resolved": [{"endpoint": "u_sub/f0", "required_ps": 8000, "action": "updated"}, {"endpoint": "u_sub/f1", "required_ps": 9000, "action": "created"}], "unresolved": ["u_sub/missing"]}
// CHECK: %[[P:.+]] = synth.timing_probe %d : i1 {required_time = 8000 : i64}
// CHECK: hw.instance "f0" @fd1(D: %[[P]]: i1, CLK: %clk: i1)
// LEGACY: %[[P:.+]] = synth.timing_probe %d : i1 {required_time = 9000 : i64}
// LEGACY: hw.instance "f1" @fd1(D: %[[P]]: i1, CLK: %clk: i1)
// PERMISSIVE: warning: timing feedback: ignoring record for endpoint 'u_sub/f0': inconsistent required/arrival/slack
// STRICT: error: timing feedback: ignoring record for endpoint 'u_sub/f0'
// UNITS: %[[P:.+]] = synth.timing_probe %d : i1 {required_time = 8000 : i64}
// UNITS: hw.instance "f0" @fd1(D: %[[P]]: i1, CLK: %clk: i1)
// TOPMISMATCH: warning: timing feedback: top 'top' does not match expected 'other'
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
