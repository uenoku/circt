// RUN: circt-opt --pass-pipeline='builtin.module(synth-size-critical{sta-report=%S/Inputs/size-critical-violated.rpt top-module-name=top})' %s | FileCheck %s
// Upsizes cells on STA-violated paths to the next stronger drive of the
// same function (exact truth table, identical ports, minimal area step).
// CHECK: hw.instance "g0" @AND2STRONG(
// CHECK-NOT: @AND2WEAK(
hw.module @AND2WEAK(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %y = comb.and %A, %B : i1
  hw.output %y : i1
}
hw.module @AND2STRONG(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.0 : f64, arcs = [#synth.linear_timing_arc<80, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<80, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %y = comb.and %A, %B : i1
  hw.output %y : i1
}
hw.module @top(in %a : i1, in %b : i1, out y : i1) {
  %y = hw.instance "g0" @AND2WEAK(A: %a: i1, B: %b: i1) -> (Y: i1)
  hw.output %y : i1
}
