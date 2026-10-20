// RUN: circt-opt --firrtl-dedup %s | FileCheck %s
// RUN: circt-opt --firrtl-dedup='dedup-classes=false' %s | FileCheck %s --check-prefix=NOMINAL

// Deduplicating classes changes class-typed module ports. Rewrite their symbol
// references before comparing instance and module port types, rather than
// trying to adapt them with wires and bundle subfields.
firrtl.circuit "ClassPorts" {
  // CHECK: firrtl.class private @A()
  // CHECK-NOT: firrtl.class private @B()
  // NOMINAL: firrtl.class private @A()
  // NOMINAL: firrtl.class private @B()
  firrtl.class private @A() {}
  firrtl.class private @B() {}

  // CHECK: firrtl.module private @ForwardA(in %i: !firrtl.class<@A()>, out %o: !firrtl.class<@A()>)
  // CHECK-NOT: firrtl.module private @ForwardB
  // NOMINAL: firrtl.module private @ForwardA(in %i: !firrtl.class<@A()>, out %o: !firrtl.class<@A()>)
  // NOMINAL: firrtl.module private @ForwardB(in %i: !firrtl.class<@B()>, out %o: !firrtl.class<@B()>)
  firrtl.module private @ForwardA(in %i: !firrtl.class<@A()>, out %o: !firrtl.class<@A()>) {
    firrtl.propassign %o, %i : !firrtl.class<@A()>
  }
  firrtl.module private @ForwardB(in %i: !firrtl.class<@B()>, out %o: !firrtl.class<@B()>) {
    firrtl.propassign %o, %i : !firrtl.class<@B()>
  }

  // CHECK: firrtl.module @ClassPorts(in %ia: !firrtl.class<@A()>, in %ib: !firrtl.class<@A()>, out %oa: !firrtl.class<@A()>, out %ob: !firrtl.class<@A()>)
  // CHECK-NEXT: %a_i, %a_o = firrtl.instance a @ForwardA(in i: !firrtl.class<@A()>, out o: !firrtl.class<@A()>)
  // CHECK-NEXT: %b_i, %b_o = firrtl.instance b @ForwardA(in i: !firrtl.class<@A()>, out o: !firrtl.class<@A()>)
  // CHECK-NEXT: firrtl.propassign %a_i, %ia : !firrtl.class<@A()>
  // CHECK-NEXT: firrtl.propassign %b_i, %ib : !firrtl.class<@A()>
  // CHECK-NEXT: firrtl.propassign %oa, %a_o : !firrtl.class<@A()>
  // CHECK-NEXT: firrtl.propassign %ob, %b_o : !firrtl.class<@A()>
  // NOMINAL: firrtl.module @ClassPorts(in %ia: !firrtl.class<@A()>, in %ib: !firrtl.class<@B()>, out %oa: !firrtl.class<@A()>, out %ob: !firrtl.class<@B()>)
  // NOMINAL-NEXT: %a_i, %a_o = firrtl.instance a @ForwardA(in i: !firrtl.class<@A()>, out o: !firrtl.class<@A()>)
  // NOMINAL-NEXT: %b_i, %b_o = firrtl.instance b @ForwardB(in i: !firrtl.class<@B()>, out o: !firrtl.class<@B()>)
  firrtl.module @ClassPorts(in %ia: !firrtl.class<@A()>, in %ib: !firrtl.class<@B()>, out %oa: !firrtl.class<@A()>, out %ob: !firrtl.class<@B()>) {
    %a_i, %a_o = firrtl.instance a @ForwardA(in i: !firrtl.class<@A()>, out o: !firrtl.class<@A()>)
    %b_i, %b_o = firrtl.instance b @ForwardB(in i: !firrtl.class<@B()>, out o: !firrtl.class<@B()>)
    firrtl.propassign %a_i, %ia : !firrtl.class<@A()>
    firrtl.propassign %b_i, %ib : !firrtl.class<@B()>
    firrtl.propassign %oa, %a_o : !firrtl.class<@A()>
    firrtl.propassign %ob, %b_o : !firrtl.class<@B()>
  }
}
