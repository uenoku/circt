// RUN: circt-opt --synth-tech-mapper='external-library-files=%S/Inputs/external-techlib.mlir' %s | FileCheck %s

hw.module @top(in %a : i1, in %b : i1, out Y : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @top
// CHECK: %[[MAPPED:.*]] = hw.instance "mapped" @ext_and
// CHECK: hw.output %[[MAPPED]] : i1
// CHECK: hw.module @ext_and
