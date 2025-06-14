// RUN: circt-synth %s -output-longest-path=%t -top counter && cat %t | FileCheck %s
// CHECK-LABEL: # Analysis result for "counter"
// CHECK-NEXT: Found 89 closed paths
// CHECK-NEXT: Maximum path delay: 46
hw.module @counter(in %a: i50, in %clk: !seq.clock, out result: i50) {
    %reg = seq.compreg %add, %clk : i50
    %add = comb.mul %reg, %a : i50
    hw.output %reg : i50
}
