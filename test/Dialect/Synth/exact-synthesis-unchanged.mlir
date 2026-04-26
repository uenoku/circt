// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allow-and=false allow-xor=false allow-dot=false})' | FileCheck %s

// CHECK-LABEL: hw.module @unchanged_tt
// CHECK: %[[TT:.+]] = comb.truth_table %a, %b -> [false, true, true, false]
// CHECK-NEXT: hw.output %[[TT]] : i1
hw.module @unchanged_tt(in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.truth_table %a, %b -> [false, true, true, false]
  hw.output %0 : i1
}
