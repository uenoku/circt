// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:2})' | FileCheck %s

// CHECK-LABEL: hw.module @and_tt
// CHECK: %[[AND:.+]] = synth.aig.and_inv %b, %a : i1
// CHECK-NEXT: hw.output %[[AND]] : i1
hw.module @and_tt(in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.truth_table %a, %b -> [false, false, false, true]
  hw.output %0 : i1
}
