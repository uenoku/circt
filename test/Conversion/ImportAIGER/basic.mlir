// RUN: circt-translate --import-aiger %S/basic.aag | FileCheck %s

// CHECK-LABEL: hw.module @aiger_top
// CHECK-SAME: (in %input_0 : i1, in %input_1 : i1, out output_0 : i1)
// CHECK: hw.output %false : i1
