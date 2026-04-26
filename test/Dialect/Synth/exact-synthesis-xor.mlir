// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.xor_inv:2})' | FileCheck %s

// CHECK-LABEL: hw.module @xor_tt
// CHECK: %[[XOR:.+]] = synth.xor_inv not %b, not %a : i1
// CHECK-NEXT: hw.output %[[XOR]] : i1
hw.module @xor_tt(in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.truth_table %a, %b -> [false, true, true, false]
  hw.output %0 : i1
}
