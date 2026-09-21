// RUN: circt-opt %s --verify-each --pass-pipeline='builtin.module(synth-gen-supergates{max-gates=3})' | FileCheck %s

// Library cells with distinct areas so cheaper same-function compositions
// outcompete earlier winners after those winners were already referenced
// as inner cells by newer candidates.
hw.module @AND2(in %A : i1, in %B : i1, out X : i1) attributes {synth.liberty.cell = {area = 10.0 : f64}, synth.mapping_cost = #synth.mapping_cost<area = 10.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = comb.and %A, %B : i1
  hw.output %0 : i1
}
hw.module @OR2(in %A : i1, in %B : i1, out X : i1) attributes {synth.liberty.cell = {area = 9.0 : f64}, synth.mapping_cost = #synth.mapping_cost<area = 9.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = comb.or %A, %B : i1
  hw.output %0 : i1
}
hw.module @NOR2(in %A : i1, in %B : i1, out X : i1) attributes {synth.liberty.cell = {area = 8.0 : f64}, synth.mapping_cost = #synth.mapping_cost<area = 8.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %t = comb.or %A, %B : i1
  %f = hw.constant true
  %0 = comb.xor %t, %f : i1
  hw.output %0 : i1
}
hw.module @NAND2(in %A : i1, in %B : i1, out X : i1) attributes {synth.liberty.cell = {area = 7.0 : f64}, synth.mapping_cost = #synth.mapping_cost<area = 7.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %t = comb.and %A, %B : i1
  %f = hw.constant true
  %0 = comb.xor %t, %f : i1
  hw.output %0 : i1
}
hw.module @INV(in %A : i1, out X : i1) attributes {synth.liberty.cell = {area = 1.0 : f64}, synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %f = hw.constant true
  %0 = comb.xor %A, %f : i1
  hw.output %0 : i1
}

// Outcompeted winners referenced as inner cells are retargeted to the
// surviving winner for the same function instead of dangling: surviving
// supergates nest references to other surviving supergates, every
// referenced module is defined, and winners carry costs.
// CHECK: hw.module @__supergate_{{[0-9]+}}
// CHECK: hw.instance "inner" @__supergate_{{[0-9]+}}(in0: %in0: i1, in1: %in1: i1) -> (Y: i1)
// CHECK: attributes {synth.mapping_cost = #synth.mapping_cost<area = {{[0-9.e+-]+}} : f64, {{.*}}synth.supergate = true
