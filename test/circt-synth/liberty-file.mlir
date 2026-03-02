// RUN: circt-synth %s --until-before mapping --liberty-files %S/Inputs/liberty-file.lib,%S/Inputs/liberty-file2.lib | FileCheck %s

hw.module @top(in %a : i1, out y : i1) {
  hw.output %a : i1
}

// CHECK: hw.module @LIBINV
// CHECK-SAME: synth.liberty.library =
// CHECK: hw.module @LIBBUF
// CHECK-SAME: synth.liberty.area = 1.000000e+00 : f64
// CHECK-SAME: synth.liberty.library =
