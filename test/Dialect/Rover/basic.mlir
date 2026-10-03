// RUN: circt-opt --rover-insert-graph %s | FileCheck %s

// CHECK-LABEL: hw.module @adder
hw.module @adder(in %a : i8, in %b : i8, out c : i8) {
  // CHECK: %[[G:.+]] = equivalence.graph -> (i8) {
  // CHECK:   %[[CA:.+]] = equivalence.class %a : i8
  // CHECK:   %[[CB:.+]] = equivalence.class %b : i8
  // CHECK:   %[[ADD:.+]] = comb.add %[[CA]], %[[CB]] : i8
  // CHECK:   equivalence.yield %[[ADD]] : i8
  // CHECK: }
  // CHECK: hw.output %[[G]] : i8
  %0 = comb.add %a, %b : i8
  hw.output %0 : i8
}
