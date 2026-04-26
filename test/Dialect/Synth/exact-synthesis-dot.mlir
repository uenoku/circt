// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allow-and=false allow-xor=false})' | FileCheck %s

// CHECK-LABEL: hw.module @dot_tt
// CHECK: %[[DOT:.+]] = synth.dot{{.*}}: i1
// CHECK-NEXT: hw.output %[[DOT]] : i1
hw.module @dot_tt(in %x : i1, in %y : i1, in %z : i1, out y : i1) {
  %0 = comb.truth_table %x, %y, %z -> [false, true, false, true,
                                        true, false, false, false]
  hw.output %0 : i1
}
