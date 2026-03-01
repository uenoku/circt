// RUN: circt-translate --import-liberty %S/basic.lib | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: synth.nldm.time_unit = #synth.nldm_time_unit<1.000000e+03 : f64>
// CHECK: hw.module @BUF(
// CHECK-SAME: synth.nldm.arcs = [#synth.nldm_arc<"A", "Y", "positive_unate", [1.200000e-02, 1.800000e-02], [2.000000e-02]>]
