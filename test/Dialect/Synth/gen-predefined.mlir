// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-gen-predefined{kind=npn max-inputs=2})' | FileCheck %s

// CHECK-LABEL: module {
// CHECK: hw.module @npn_i1_tt_0_v0
// CHECK: %[[TT0:.+]] = comb.truth_table %i0 -> [false, false]
// CHECK: hw.output %[[TT0]] : i1
// CHECK: hw.module @npn_i2_tt_6_v0
// CHECK: %[[TT1:.+]] = comb.truth_table %i1, %i0 -> [false, true, true, false]
// CHECK: hw.output %[[TT1]] : i1
module {
}
