// RUN: circt-opt --pass-pipeline='builtin.module(synth-gen-supergates,synth-tech-mapper)' %s | FileCheck %s

// Supergates generated inline are consumed by TechMapper as ordinary
// patterns: mapping succeeds, no aig ops remain, and the supergate modules
// themselves keep their costs (patterns, not design logic).
hw.module @AND2(in %a : i1, in %b : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

hw.module @INV(in %a : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.5 : f64, arcs = [#synth.linear_timing_arc<50, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @top(
// CHECK-NOT: synth.aig.and_inv
// CHECK: hw.instance {{.*}} @AND2(
// CHECK: hw.output
// Supergate modules survive with costs intact (usable as patterns).
// CHECK: hw.module @__supergate_{{[0-9]+}}
// CHECK-SAME: synth.mapping_cost
// CHECK-SAME: synth.supergate = true
hw.module @top(in %a : i1, in %b : i1, in %c : i1, out result : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c : i1
  hw.output %1 : i1
}
