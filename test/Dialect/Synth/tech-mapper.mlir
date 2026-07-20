// RUN: circt-opt --synth-tech-mapper='strategy=area test=true' %s | FileCheck %s --check-prefixes=CHECK,AREA
// RUN: circt-opt --synth-tech-mapper='strategy=timing test=true' %s | FileCheck %s --check-prefixes=CHECK,TIMING

hw.module @xor_cell(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.dot not %b, %b, not %a : i1
  hw.output %0 : i1
}

hw.module @and_cell(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

hw.module @fast3_cell(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %c, not %0 : i1
  hw.output %1 : i1
}

hw.module @majority_cell(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %0 = synth.majority %a, %b, %c : i1
  hw.output %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>
  ]>
} {
  %0 = hw.instance "mapped" @xor_cell(a: %a: i1, b: %b: i1) -> (y: i1)
  synth.yield %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1) -> i1 attributes {
  allow_negation = true,
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>
  ]>
} {
  %0 = hw.instance "mapped" @and_cell(a: %a: i1, b: %b: i1) -> (y: i1)
  synth.yield %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1, %c: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 10.0 : f64, arcs = [
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>
  ]>
} {
  %0 = hw.instance "mapped" @fast3_cell(a: %a: i1, b: %b: i1, c: %c: i1) -> (y: i1)
  synth.yield %0 : i1
}

// This cheaper pattern cannot add the input negation needed by
// @negation_policy.
synth.cut_rewrite_pattern (%a: i1, %b: i1, %c: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>
  ]>
} {
  %0 = hw.instance "mapped" @majority_cell(a: %a: i1, b: %b: i1, c: %c: i1) -> (y: i1)
  synth.yield %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1, %c: i1) -> i1 attributes {
  allow_negation = true,
  cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
    #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>
  ]>
} {
  %0 = hw.instance "mapped" @majority_cell(a: %a: i1, b: %b: i1, c: %c: i1) -> (y: i1)
  synth.yield %0 : i1
}

// CHECK-LABEL: hw.module @xor_cell
// CHECK: synth.dot

// CHECK-LABEL: hw.module @rewrite_xor
// CHECK-NEXT: %[[CELL:.+]] = hw.instance "mapped" @xor_cell(a: %a: i1, b: %b: i1) -> (y: i1) {test.arrival_times = [1]}
// CHECK-NEXT: hw.output %[[CELL]] : i1
hw.module @rewrite_xor(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.xor_inv %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @test_strategy
// AREA: hw.instance "mapped" @and_cell
// AREA: hw.instance "mapped" @and_cell
// AREA-NOT: hw.instance "mapped" @fast3_cell
// AREA: hw.output
// TIMING: hw.instance "mapped" @fast3_cell
// TIMING-NOT: hw.instance "mapped" @and_cell
// TIMING: hw.output
hw.module @test_strategy(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %c, not %0 : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @timing_chain
// CHECK: hw.instance "mapped" @and_cell
// CHECK: hw.instance "mapped" @and_cell
// CHECK: hw.instance "mapped" @and_cell
// CHECK: test.arrival_times = [2]
hw.module @timing_chain(in %a : i1, in %b : i1, in %c : i1, in %d : i1,
                        out y : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %c, %d : i1
  %2 = synth.aig.and_inv %0, %1 : i1
  hw.output %2 : i1
}

// CHECK-LABEL: hw.module @negation_policy
// CHECK: synth.aig.and_inv not
// CHECK: hw.instance "mapped" @majority_cell
hw.module @negation_policy(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %0 = synth.majority not %a, %b, %c : i1
  hw.output %0 : i1
}
