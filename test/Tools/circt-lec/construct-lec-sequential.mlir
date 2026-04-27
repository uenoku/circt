// RUN: circt-opt --convert-to-arcs --construct-lec="first-module=seqA second-module=seqB insert-mode=none sequential-mode=arc-state" %s | FileCheck %s

hw.module @seqA(in %clk: !seq.clock, in %a: i8, in %b: i8, in %rst: i1, out out: i8) {
  %sum = comb.add %a, %b : i8
  %c0 = hw.constant 0 : i8
  %state = seq.compreg name "state" %sum, %clk reset %rst, %c0 : i8
  hw.output %state : i8
}

hw.module @seqB(in %clk: !seq.clock, in %a: i8, in %b: i8, in %rst: i1, out out: i8) {
  %sum = comb.add %b, %a : i8
  %c0 = hw.constant 0 : i8
  %state = seq.compreg name "state" %sum, %clk reset %rst, %c0 : i8
  hw.output %state : i8
}

// CHECK-NOT: hw.module @seqA
// CHECK-NOT: hw.module @seqB
// CHECK-NOT: arc.define

// CHECK: verif.lec first {
// CHECK: ^bb0(%{{.+}}: i8, %{{.+}}: i8, %{{.+}}: i1, %[[STATE:.+]]: i8):
// CHECK:   verif.yield %[[STATE]] : i8
// CHECK: } second {
// CHECK: ^bb0(%{{.+}}: i8, %{{.+}}: i8, %{{.+}}: i1, %[[STATE2:.+]]: i8):
// CHECK:   verif.yield %[[STATE2]] : i8

// CHECK: verif.lec first {
// CHECK: ^bb0(%[[A0:.+]]: i8, %[[B0:.+]]: i8, %[[RST0:.+]]: i1, %[[CUR0:.+]]: i8):
// CHECK:   %[[SUM0:.+]] = comb.add %[[A0]], %[[B0]] : i8
// CHECK:   %[[ZERO0:.+]] = hw.constant 0 : i8
// CHECK:   %[[NEXT0:.+]] = comb.mux %[[RST0]], %[[ZERO0]], %[[SUM0]] : i8
// CHECK:   verif.yield %[[NEXT0]] : i8
// CHECK: } second {
// CHECK: ^bb0(%[[A1:.+]]: i8, %[[B1:.+]]: i8, %[[RST1:.+]]: i1, %[[CUR1:.+]]: i8):
// CHECK:   %[[SUM1:.+]] = comb.add %[[B1]], %[[A1]] : i8
// CHECK:   %[[ZERO1:.+]] = hw.constant 0 : i8
// CHECK:   %[[NEXT1:.+]] = comb.mux %[[RST1]], %[[ZERO1]], %[[SUM1]] : i8
// CHECK:   verif.yield %[[NEXT1]] : i8
