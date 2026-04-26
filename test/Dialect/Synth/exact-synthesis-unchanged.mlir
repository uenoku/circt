// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.xor_inv:3})' | FileCheck %s

// CHECK-LABEL: hw.module @unchanged_tt
// CHECK: %[[TT:.+]] = comb.truth_table %a, %b -> [false, false, false, true]
// CHECK-NEXT: hw.output %[[TT]] : i1
hw.module @unchanged_tt(in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.truth_table %a, %b -> [false, false, false, true]
  hw.output %0 : i1
}
