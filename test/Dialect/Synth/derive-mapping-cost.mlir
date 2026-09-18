// RUN: circt-opt --pass-pipeline='builtin.module(synth-derive-mapping-cost)' %s | FileCheck %s
// RUN: circt-opt --pass-pipeline='builtin.module(synth-derive-mapping-cost{dont-use=a_nouse})' %s | FileCheck %s --check-prefix=DONTUSE

// One input, one arc: mean(0.1, 0.2, 0.3)ns = 0.2ns -> 200ps, positive.
// CHECK-LABEL: hw.module @buf(
// CHECK-SAME: synth.mapping_cost = #synth.mapping_cost<area = 5.000000e+00 : f64, arcs = [#synth.linear_timing_arc<200, 0, <positive>>], input_caps = {a = 1.500000e-03 : f64}>
hw.module @buf(in %a : i1 {synth.liberty.pin = {capacitance = 1.5e-03 : f64}}, out z : i1) attributes {synth.liberty.cell = {area = 5.0 : f64}, synth.timing.nldm = [#synth.nldm_timing_arc<"z", "a", "positive_unate", <[0.01], [0.001], [0.1, 0.2]>, <[0.01], [0.001], [0.3]>, <[0.01], [0.001], [0.4]>, <[0.01], [0.001], [0.5]>>]} {
  hw.output %a : i1
}

// Non-unate sense maps to positive; two arcs on one pin pool values:
// mean(1.0, 3.0) = 2.0ns -> 2000ps. No pin capacitance: no input_caps.
// CHECK-LABEL: hw.module @xor2(
// CHECK-SAME: synth.mapping_cost = #synth.mapping_cost<area = 9.000000e+00 : f64, arcs = [#synth.linear_timing_arc<2000, 0, <positive>>, #synth.linear_timing_arc<2000, 0, <positive>>]>
hw.module @xor2(in %a : i1, in %b : i1, out z : i1) attributes {synth.liberty.cell = {area = 9.0 : f64}, synth.timing.nldm = [#synth.nldm_timing_arc<"z", "a", "non_unate", <[0.01], [0.001], [1.0]>, <[0.01], [0.001], [3.0]>, <[0.01], [0.001], [0.0]>, <[0.01], [0.001], [0.0]>>, #synth.nldm_timing_arc<"z", "b", "non_unate", <[0.01], [0.001], [1.0]>, <[0.01], [0.001], [3.0]>, <[0.01], [0.001], [0.0]>, <[0.01], [0.001], [0.0]>>]} {
  %0 = comb.xor %a, %b : i1
  hw.output %0 : i1
}

// Sensitivity fit: middle slew row (index1 rows 0,1 -> row 1) over load
// columns [1.0,2.0,4.0]: rise (0.15,0.25,0.45) and fall (0.25,0.35,0.55)
// both fit slope 0.1ns/unit -> 100ps/unit. Intrinsic: mean of all 12
// delay values = 0.308333ns -> 308ps.
// CHECK-LABEL: hw.module @sens(
// CHECK-SAME: synth.mapping_cost = #synth.mapping_cost<area = 7.000000e+00 : f64, arcs = [#synth.linear_timing_arc<308, 100, <positive>>]>
hw.module @sens(in %a : i1, out z : i1) attributes {synth.liberty.cell = {area = 7.0 : f64}, synth.timing.nldm = [#synth.nldm_timing_arc<"z", "a", "positive_unate", <[0.01, 0.02], [1.0, 2.0, 4.0], [0.1, 0.2, 0.4, 0.15, 0.25, 0.45]>, <[0.01, 0.02], [1.0, 2.0, 4.0], [0.2, 0.3, 0.5, 0.25, 0.35, 0.55]>, <[0.01, 0.02], [1.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]>, <[0.01, 0.02], [1.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]>>]} {
  hw.output %a : i1
}

// Missing arc for an input: no cost.
// CHECK-LABEL: hw.module @partial(
// CHECK-NOT: synth.mapping_cost
hw.module @partial(in %a : i1, in %b : i1, out z : i1) attributes {synth.liberty.cell = {area = 9.0 : f64}, synth.timing.nldm = [#synth.nldm_timing_arc<"z", "a", "positive_unate", <[0.01], [0.001], [1.0]>, <[0.01], [0.001], [1.0]>, <[0.01], [0.001], [0.0]>, <[0.01], [0.001], [0.0]>>]} {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}

// Multi-output and stateful modules are skipped. Externs are skipped.
// CHECK-LABEL: hw.module @two_out(
// CHECK-NOT: synth.mapping_cost
hw.module @two_out(out x : i1, out y : i1) attributes {synth.liberty.cell = {area = 2.0 : f64}} {
  %true = hw.constant true
  %false = hw.constant false
  hw.output %true, %false : i1, i1
}

// CHECK-LABEL: hw.module @ff_cell(
// CHECK-NOT: synth.mapping_cost
hw.module @ff_cell(in %d : i1, in %clk : i1, out q : i1) attributes {synth.liberty.cell = {area = 20.0 : f64}, synth.timing.nldm = [#synth.nldm_timing_arc<"q", "clk", "non_unate", <[0.01], [0.001], [1.0]>, <[0.01], [0.001], [1.0]>, <[0.01], [0.001], [0.0]>, <[0.01], [0.001], [0.0]>>]} {
  %q, %qn = synth.ff %d clock %clk edge rise : i1, i1
  hw.output %q : i1
}

// CHECK-LABEL: hw.module.extern @blackbox
hw.module.extern @blackbox(in %a : i1, out z : i1)

// dont_use marks and skips, with or without other costs. Default run keeps
// the cost; the dont-use run marks.
// CHECK-LABEL: hw.module @a_nouse(
// CHECK-SAME: synth.mapping_cost = #synth.mapping_cost<
// DONTUSE-LABEL: hw.module @a_nouse(
// DONTUSE-SAME: synth.dont_use
// DONTUSE-NOT: synth.mapping_cost
hw.module @a_nouse(in %a : i1, out z : i1) attributes {synth.liberty.cell = {area = 5.0 : f64}, synth.timing.nldm = [#synth.nldm_timing_arc<"z", "a", "positive_unate", <[0.01], [0.001], [1.0]>, <[0.01], [0.001], [1.0]>, <[0.01], [0.001], [0.0]>, <[0.01], [0.001], [0.0]>>]} {
  hw.output %a : i1
}
