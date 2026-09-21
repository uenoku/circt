// RUN: circt-opt --synth-map-bundles %s | FileCheck %s

// Fake 2-output full-adder cell (comb body).
// CHECK-LABEL: hw.module @fa_cell
hw.module @fa_cell(in %a: i1, in %b: i1, in %c: i1, out s: i1, out co: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %ab = comb.and %a, %b : i1
  %ac = comb.and %a, %c : i1
  %bc = comb.and %b, %c : i1
  %t = comb.or %ab, %ac : i1
  %co = comb.or %t, %bc : i1
  %xab = comb.xor %a, %b : i1
  %s = comb.xor %xab, %c : i1
  hw.output %s, %co : i1, i1
}

// A bundle over matching sum/carry becomes one cell instance.
// CHECK-LABEL: hw.module @top
// CHECK: hw.instance {{.*}} @fa_cell
// CHECK-NOT: create_bundle
hw.module @top(in %a: i1, in %b: i1, in %c: i1, out s: i1, out co: i1) {
  %n1 = synth.xor_inv %a, %b : i1
  %s = synth.xor_inv %n1, %c : i1
  %m = synth.majority %a, %b, %c : i1
  %o0, %o1 = synth.create_bundle %s, %m : (i1, i1) -> (i1, i1)
  hw.output %o0, %o1 : i1, i1
}

// A bundle with no matching cell is erased, discrete logic stays.
// CHECK-LABEL: hw.module @top_erase
// CHECK-NOT: create_bundle
// CHECK: hw.output
hw.module @top_erase(in %a: i1, in %b: i1, out x: i1, out y: i1) {
  %x = synth.xor_inv %a, %b : i1
  %y = synth.aig.and_inv %a, %b : i1
  %o0, %o1 = synth.create_bundle %x, %y : (i1, i1) -> (i1, i1)
  hw.output %o0, %o1 : i1, i1
}

// Complemented-output full-adder cell (xnor sum + minority carry) plus a
// discrete inverter for output-negation matching.
// CHECK-LABEL: hw.module @fa_cell_neg
hw.module @fa_cell_neg(in %a: i1, in %b: i1, in %c: i1, out s: i1, out co: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 9.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %xab = comb.xor %a, %b : i1
  %x = comb.xor %xab, %c : i1
  %f = hw.constant true
  %s = comb.xor %x, %f : i1
  %ab = comb.and %a, %b : i1
  %ac = comb.and %a, %c : i1
  %bc = comb.and %b, %c : i1
  %t = comb.or %ab, %ac : i1
  %o = comb.or %t, %bc : i1
  %co = comb.xor %o, %f : i1
  hw.output %s, %co : i1, i1
}

// CHECK-LABEL: hw.module @inv_cell
hw.module @inv_cell(in %a: i1, out y: i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>} {
  %f = hw.constant true
  %y = comb.xor %a, %f : i1
  hw.output %y : i1
}

// A bundle whose sum polarity differs maps with one output inverter.
// CHECK-LABEL: hw.module @top_neg
// CHECK: hw.instance {{.*}} @fa_cell_neg
// CHECK: hw.instance {{.*}} @inv_cell
// CHECK-NOT: create_bundle
hw.module @top_neg(in %a: i1, in %b: i1, in %c: i1, out s: i1, out co: i1) {
  %n1 = synth.xor_inv %a, %b : i1
  %s = synth.xor_inv %n1, %c : i1
  %m = synth.majority not %a, not %b, not %c : i1
  %o0, %o1 = synth.create_bundle %s, %m : (i1, i1) -> (i1, i1)
  hw.output %o0, %o1 : i1, i1
}
