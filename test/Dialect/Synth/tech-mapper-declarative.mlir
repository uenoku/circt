// RUN: circt-opt --synth-tech-mapper='test=true' %s | FileCheck %s

synth.cut_rewrite_pattern (%a: i1, %b: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>
  ]>
} {
  %0 = synth.dot not %b, %b, not %a : i1
  synth.yield %0 : i1
}

// CHECK-LABEL: hw.module @rewrite_xor
// CHECK-NEXT: %[[DOT:.+]] = synth.dot not %b, %b, not %a {test.arrival_times = [1]} : i1
// CHECK-NEXT: hw.output %[[DOT]] : i1
hw.module @rewrite_xor(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.xor_inv %a, %b : i1
  hw.output %0 : i1
}
