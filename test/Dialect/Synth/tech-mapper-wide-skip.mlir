// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper)' %s | FileCheck %s

// A 9-input cell with no offline witness table is skipped as a pattern;
// logic still maps through smaller cells via smaller cuts. (Wide logic
// arrives binarized by lowering passes; wide patterns match wide cuts of
// small gates.)
hw.module @and2(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

hw.module @wide9(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, in %f : i1, in %g : i1, in %h : i1, in %i : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 5.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c : i1
  %2 = synth.aig.and_inv %1, %d : i1
  %3 = synth.aig.and_inv %2, %e : i1
  %4 = synth.aig.and_inv %3, %f : i1
  %5 = synth.aig.and_inv %4, %g : i1
  %6 = synth.aig.and_inv %5, %h : i1
  %7 = synth.aig.and_inv %6, %i : i1
  hw.output %7 : i1
}

// CHECK-LABEL: hw.module @top(
// CHECK-NOT: @wide9(
// CHECK-COUNT-8: hw.instance {{.*}} @and2(
hw.module @top(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, in %f : i1, in %g : i1, in %h : i1, in %i : i1, out result : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c : i1
  %2 = synth.aig.and_inv %1, %d : i1
  %3 = synth.aig.and_inv %2, %e : i1
  %4 = synth.aig.and_inv %3, %f : i1
  %5 = synth.aig.and_inv %4, %g : i1
  %6 = synth.aig.and_inv %5, %h : i1
  %7 = synth.aig.and_inv %6, %i : i1
  hw.output %7 : i1
}
