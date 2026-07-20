// REQUIRES: z3

// RUN: circt-synth %s -o %t1.mlir
// RUN: cat %t1.mlir | FileCheck %s
// RUN: circt-opt %t1.mlir --hw-flatten-modules=hw-inline-public -o %t1.inline.mlir
// RUN: circt-lec.sh %t1.inline.mlir %s -c1=mul -c2=mul
// RUN: circt-lec.sh %t1.inline.mlir %s -c1=dot_test -c2=dot_test

// RUN: circt-synth %s -o %t.lut.mlir --top mul --lower-to-k-lut 6
// RUN: cat %t.lut.mlir | FileCheck %s --check-prefix=LUT
// RUN: circt-opt -lower-comb %t.lut.mlir -o %t2.mlir
// RUN: circt-lec.sh %t2.mlir %s -c1=mul -c2=mul

// Set delay for binary and inv op to 5 so that others will be prioritized
hw.module @and_inv(in %a : i1, in %b : i1, out result : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

hw.module @and_inv_n(in %a : i1, in %b : i1, out result : i1) {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

hw.module @and_inv_nn(in %a : i1, in %b : i1, out result : i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}

hw.module @nand_nand(in %a : i1, in %b : i1, in %c : i1, in %d : i1,
                     out result : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %c, %d : i1
  %2 = synth.aig.and_inv not %0, not %1 : i1
  hw.output %2 : i1
}

hw.module @some(in %a : i1, in %b : i1, out result : i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  %1 = synth.aig.and_inv %a, %b : i1
  %2 = synth.aig.and_inv not %0, not %1 : i1
  hw.output %2 : i1
}

hw.module @dot_lib(in %x : i1, in %y : i1, in %z : i1, out result : i1) {
  %0 = synth.dot %z, not %x, not %y : i1
  hw.output %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<5, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<5, 0, #synth.polarity<positive>>]>
} {
  %0 = hw.instance "mapped" @and_inv(a: %a: i1, b: %b: i1) -> (result: i1)
  synth.yield %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<5, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<5, 0, #synth.polarity<positive>>]>
} {
  %0 = hw.instance "mapped" @and_inv_n(a: %a: i1, b: %b: i1) -> (result: i1)
  synth.yield %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<5, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<5, 0, #synth.polarity<positive>>]>
} {
  %0 = hw.instance "mapped" @and_inv_nn(a: %a: i1, b: %b: i1) -> (result: i1)
  synth.yield %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1, %c: i1, %d: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 3.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>
} {
  %0 = hw.instance "mapped" @nand_nand(a: %a: i1, b: %b: i1, c: %c: i1, d: %d: i1) -> (result: i1)
  synth.yield %0 : i1
}

synth.cut_rewrite_pattern (%a: i1, %b: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>
} {
  %0 = hw.instance "mapped" @some(a: %a: i1, b: %b: i1) -> (result: i1)
  synth.yield %0 : i1
}

synth.cut_rewrite_pattern (%x: i1, %y: i1, %z: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>]>
} {
  %0 = hw.instance "mapped" @dot_lib(x: %x: i1, y: %y: i1, z: %z: i1) -> (result: i1)
  synth.yield %0 : i1
}

hw.module @dot_test(in %x : i1, in %y : i1, in %z : i1, out result : i1) {
    %0 = synth.dot %x, not %y, not %z : i1
    hw.output %0 : i1
}

// Make sure @mul is mapped to the cells referenced by the declarative patterns.
// CHECK-LABEL: hw.module @mul
// CHECK-NOT: comb.and
// CHECK-NOT: comb.xor
// CHECK-DAG: hw.instance {{".+"}} @and_inv
// CHECK-DAG: hw.instance {{".+"}} @some
// LUT: hw.module @mul
// LUT: comb.truth_table
// LUT-NOT: synth.aig.and_inv
// LUT-NOT: comb.and
// LUT-NOT: comb.xor
// LUT-NOT: hw.instance
hw.module @mul(in %arg0: i4, in %arg1: i4, out add: i4) {
  %0 = comb.mul %arg0, %arg1 : i4
  hw.output %0 : i4
}
