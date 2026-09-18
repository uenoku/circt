// RUN: circt-opt --synth-extract-bundles --mlir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS
// RUN: circt-opt --synth-extract-bundles %s | FileCheck %s

// A full-adder bit: xor3 sum + majority carry on a shared triple.
// STATS: 1 full-adders
// CHECK-LABEL: hw.module @fa_bit
hw.module @fa_bit(in %a: i1, in %b: i1, in %cin: i1, out s: i1, out co: i1) {
  %n1 = synth.xor_inv %a, %b : i1
  // CHECK: synth.create_bundle
  %s = synth.xor_inv %n1, %cin : i1
  %co = synth.majority %a, %b, %cin : i1
  hw.output %s, %co : i1, i1
}

// A half-adder bit: xor2 + and2 on a shared pair.
// STATS: 1 half-adders
// CHECK-LABEL: hw.module @ha_bit
hw.module @ha_bit(in %a: i1, in %b: i1, out s: i1, out co: i1) {
  %s = synth.xor_inv %a, %b : i1
  %co = synth.aig.and_inv %a, %b : i1
  hw.output %s, %co : i1, i1
}
