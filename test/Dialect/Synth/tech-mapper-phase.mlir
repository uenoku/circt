// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=area test=true max-cuts-per-root=8})' %s | FileCheck %s

// Phase-aware matching: cuts whose NPN binding carries input negation
// match by materializing inverter cells (previously rejected outright).
// Here only a plain AND2 and an inverter exist, so and(!a, b) must map
// to AND2 fed by an inverter.

// CHECK-LABEL: hw.module @myand2(
hw.module @myand2(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @myinv(
hw.module @myinv(in %a : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.5 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @phase_and_not(
// CHECK: %[[INV:.+]] = hw.instance {{.*}} @myinv(a: %a: i1) -> (result: i1)
// CHECK: %[[OUT:.+]] = hw.instance {{.*}} @myand2(a: %[[INV]]: i1, b: %b: i1) -> (result: i1)
// CHECK: hw.output %[[OUT]] : i1
hw.module @phase_and_not(in %a : i1, in %b : i1, out result : i1) {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

// A plain AND still maps directly with no inverters.
// CHECK-LABEL: hw.module @phase_and(
// CHECK-NOT: @myinv
// CHECK: hw.instance {{.*}} @myand2(a: %a: i1, b: %b: i1) -> (result: i1)
hw.module @phase_and(in %a : i1, in %b : i1, out result : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
