// RUN: circt-synth %s -output-longest-path=%t -top counter && cat %t | FileCheck %s
// CHECK-LABEL: # Analysis result for "counter"
// CHECK-NEXT: Found 89 closed paths
// CHECK-NEXT: Maximum path delay: 46
hw.module @counter(in %a: i3, out result: i3) {
    %add = comb.mul %a, %a : i3
    hw.output %add : i3
}
