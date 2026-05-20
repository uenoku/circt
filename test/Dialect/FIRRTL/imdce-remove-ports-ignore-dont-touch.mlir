// RUN: circt-opt -pass-pipeline='builtin.module(firrtl-imdeadcodeelim{remove-ports-only ignore-dont-touch})' %s | FileCheck %s

firrtl.circuit "IgnoreDontTouch" {
  firrtl.module @IgnoreDontTouch() {
    // CHECK: firrtl.instance bar @Bar()
    %bar_sym, %bar_anno = firrtl.instance bar @Bar(out sym: !firrtl.uint<1>, out anno: !firrtl.uint<1>)
  }

  // CHECK-LABEL: firrtl.module private @Bar() {
  // CHECK-NEXT:  }
  firrtl.module private @Bar(out %sym: !firrtl.uint<1> sym @sym, out %anno: !firrtl.uint<1>) attributes {
    portAnnotations = [[], [{a = "a"}]]
  } {
  }
}
