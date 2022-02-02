// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(merge-connections))' %s | FileCheck %s

firrtl.circuit "Test"   {
  // circuit Test :
  //   module Test :
  //     input a : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     output b : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     b <= a
  // CHECK-LABEL: firrtl.module @Test(in %a: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>, out %b: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>) {
  // CHECK-NEXT:    firrtl.connect %b, %a : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>, !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
  // CHECK-NEXT:  }
  firrtl.module @Test(in %a: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>, out %b: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>) {
     %0 = firrtl.subindex %a[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %1 = firrtl.subindex %b[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %2 = firrtl.subfield %0(0) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.clock
     %3 = firrtl.subfield %1(0) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.clock
     firrtl.connect %3, %2 : !firrtl.clock, !firrtl.clock
     %4 = firrtl.subfield %0(1) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.uint<1>
     %5 = firrtl.subfield %1(1) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.uint<1>
     firrtl.connect %5, %4 : !firrtl.uint<1>, !firrtl.uint<1>
     %6 = firrtl.subindex %a[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %7 = firrtl.subindex %b[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %8 = firrtl.subfield %6(0) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.clock
     %9 = firrtl.subfield %7(0) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.clock
     firrtl.connect %9, %8 : !firrtl.clock, !firrtl.clock
     %10 = firrtl.subfield %6(1) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.uint<1>
     %11 = firrtl.subfield %7(1) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.uint<1>
     firrtl.connect %11, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // circuit Bar :
  //   module Bar :
  //     output a : {b: UInt<1>, c:UInt<1>}
  //     a.b <= UInt<1>(0)
  //     a.c <= UInt<1>(1)
  // CHECK-LABEL: firrtl.module @Constant(out %a: !firrtl.bundle<b: uint<1>, c: uint<1>>) {
  // CHECK-NEXT:    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  // CHECK-NEXT:    %0 = firrtl.bitcast %c1_ui2 : (!firrtl.uint<2>) -> !firrtl.bundle<b: uint<1>, c: uint<1>>
  // CHECK-NEXT:    firrtl.connect %a, %0 : !firrtl.bundle<b: uint<1>, c: uint<1>>, !firrtl.bundle<b: uint<1>, c: uint<1>>
  // CHECK-NEXT:  }
  firrtl.module @Constant(out %a: !firrtl.bundle<b: uint<1>, c: uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<b: uint<1>, c: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %a(1) : (!firrtl.bundle<b: uint<1>, c: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module @ConcatToVector(in %s1: !firrtl.uint<1>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
  // CHECK-NEXT: %0 = firrtl.cat %s2, %s1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK-NEXT: %1 = firrtl.bitcast %0 : (!firrtl.uint<2>) -> !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: firrtl.connect %sink, %1 : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: }
  firrtl.module @ConcatToVector(in %s1: !firrtl.uint<1>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %s1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %s2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Check that we don't use %s1 as a source value.
  // CHECK: firrtl.module @FailedToUseAggregate(in %s1: !firrtl.vector<uint<1>, 2>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>)
  // CHECK-NEXT:  %0 = firrtl.subindex %s1[0] : !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT:  %1 = firrtl.cat %s2, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK-NEXT:  %2 = firrtl.bitcast %1 : (!firrtl.uint<2>) -> !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT:  firrtl.connect %sink, %2 : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: }
  firrtl.module @FailedToUseAggregate(in %s1: !firrtl.vector<uint<1>, 2>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %s1[0] : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %s2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
