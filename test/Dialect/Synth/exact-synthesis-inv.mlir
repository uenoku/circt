// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:2})' | FileCheck %s

// CHECK-LABEL: hw.module @inv_tt
// CHECK: %[[INV:.+]] = synth.aig.and_inv not %a : i1
// CHECK-NEXT: hw.output %[[INV]] : i1
hw.module @inv_tt(in %a : i1, out y : i1) {
  %0 = comb.truth_table %a -> [true, false]
  hw.output %0 : i1
}
