// RUN: circt-opt --pass-pipeline='builtin.module(convert-comb-to-synth, synth-tech-mapper)' %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt --pass-pipeline='builtin.module(convert-comb-to-synth, synth-tech-mapper{reserve-ingredient-cuts=2 max-cuts-per-root=12})' %s | FileCheck %s --check-prefix=RESERVE

// Ingredient-cut reservation protects the small building-block cuts that
// combine into larger covers (here: minority cones over shared ANDs inside
// a Wallace multiplier) from bottom-up priority-cut truncation crowding.
// Default retention finds 5 MAJI covers; reserving 2 ingredient slots (with
// a wider 12-cut limit) recovers a 6th MAJI and drops one discrete cell.

// DEFAULT-COUNT-5: hw.instance "mapped_{{[0-9]+}}" @MAJI(
// RESERVE-COUNT-6: hw.instance "mapped_{{[0-9]+}}" @MAJI(

hw.module @AND2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %y = comb.and %A, %B : i1
  hw.output %y : i1
}

hw.module @OR2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %y = comb.or %A, %B : i1
  hw.output %y : i1
}

hw.module @XOR2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %y = comb.xor %A, %B : i1
  hw.output %y : i1
}

hw.module @INV(in %A : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %f = hw.constant true
  %y = comb.xor %A, %f : i1
  hw.output %y : i1
}

hw.module @MAJI(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.and %A, %B : i1
  %ac = comb.and %A, %C : i1
  %bc = comb.and %B, %C : i1
  %o1 = comb.or %ab, %ac : i1
  %o2 = comb.or %o1, %bc : i1
  %f = hw.constant true
  %y = comb.xor %o2, %f : i1
  hw.output %y : i1
}

hw.module @NAND2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %o = comb.and %A, %B : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @NOR2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %o = comb.or %A, %B : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @XNOR2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %o = comb.xor %A, %B : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @AND3(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.and %A, %B : i1
  %y = comb.and %ab, %C : i1
  hw.output %y : i1
}

hw.module @OR3(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.or %A, %B : i1
  %y = comb.or %ab, %C : i1
  hw.output %y : i1
}

hw.module @NAND3(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.and %A, %B : i1
  %o = comb.and %ab, %C : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @NOR3(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.or %A, %B : i1
  %o = comb.or %ab, %C : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @AND4(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.and %A, %B : i1
  %cd = comb.and %C, %D : i1
  %y = comb.and %ab, %cd : i1
  hw.output %y : i1
}

hw.module @OR4(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.or %A, %B : i1
  %cd = comb.or %C, %D : i1
  %y = comb.or %ab, %cd : i1
  hw.output %y : i1
}

hw.module @AOI21x1(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.and %A, %B : i1
  %o = comb.or %ab, %C : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @OAI21x1(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.or %A, %B : i1
  %o = comb.and %ab, %C : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @AOI22x1(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.and %A, %B : i1
  %cd = comb.and %C, %D : i1
  %o = comb.or %ab, %cd : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @OAI22x1(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.or %A, %B : i1
  %cd = comb.or %C, %D : i1
  %o = comb.and %ab, %cd : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @AOI21x2(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.and %A, %B : i1
  %o = comb.or %ab, %C : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @OAI21x2(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.or %A, %B : i1
  %o = comb.and %ab, %C : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @AOI22x2(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.and %A, %B : i1
  %cd = comb.and %C, %D : i1
  %o = comb.or %ab, %cd : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @OAI22x2(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.5 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %ab = comb.or %A, %B : i1
  %cd = comb.or %C, %D : i1
  %o = comb.and %ab, %cd : i1
  %f = hw.constant true
  %y = comb.xor %o, %f : i1
  hw.output %y : i1
}

hw.module @top(in %a : i6, in %b : i6, out y : i6) {
  %p = comb.mul %a, %b : i6
  hw.output %p : i6
}
