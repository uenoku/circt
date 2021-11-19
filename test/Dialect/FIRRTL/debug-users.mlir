// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imconstprop)' --split-input-file  %s | FileCheck %s

firrtl.circuit "Test" {
  // CHECK-LABEL: firrtl.module @Test
  firrtl.module @Test(in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<2>, 2>, out %b: !firrtl.uint<2>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %r3 = firrtl.reg %clock  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %r3[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %r3[1] : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.add %0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    firrtl.connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
