// RUN: circt-opt --pass-pipeline='builtin.module(convert-comb-to-synth, synth-tech-mapper, synth-bubble-pushing)' %s | FileCheck %s
// Single-fanout AND+INV absorbs into cheaper NAND by exact truth-table match.
// CHECK: hw.instance "mapped_{{.+}}" @NAND2(
// CHECK-NOT: @INV(
// CHECK-NOT: @AND2(
hw.module @AND2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %y = comb.and %A, %B : i1
  hw.output %y : i1
}
hw.module @NAND2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %c = hw.constant true
  %0 = comb.xor %A, %c : i1
  %1 = comb.xor %B, %c : i1
  %y = comb.or %0, %1 : i1
  hw.output %y : i1
}
hw.module @INV(in %A : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<50, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %c = hw.constant true
  %y = comb.xor %A, %c : i1
  hw.output %y : i1
}
hw.module @top(in %a : i1, in %b : i1, out y : i1) {
  %c = hw.constant true
  %0 = comb.xor %a, %c : i1
  %1 = comb.xor %b, %c : i1
  %y = comb.or %0, %1 : i1
  hw.output %y : i1
}
