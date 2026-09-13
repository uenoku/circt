// RUN: circt-opt --pass-pipeline='builtin.module(strip-assert-like)' %s | FileCheck %s

// CHECK-LABEL: hw.module @erase(
// CHECK-NOT: verif.assert
// CHECK-NOT: verif.assume
// CHECK-NOT: verif.cover
// CHECK-NOT: verif.clocked_assert
// CHECK: hw.output
hw.module @erase(in %clk : i1, in %a : i1, in %en : i1, out o : i1) {
  verif.assert %a label "p1" : i1
  verif.assume %a label "p2" : i1
  verif.cover %a label "p3" : i1
  verif.clocked_assert %a if %en, posedge %clk : i1
  hw.output %a : i1
}
