// RUN: circt-synth %s -print-longest-path -top counter -num-longest-path-fanout=32 | FileCheck %s
// CHECK-LABEL: # Longest Path Analysis result for "counter"
// CHECK-LABEL: Showing Levels
// CHECK:      Level = 1         . Count = 2         . 6.25      %
// CHECK-NEXT: Level = 3         . Count = 2         . 12.50     %
// CHECK-NEXT: Level = 7         . Count = 2         . 18.75     %
// CHECK-NEXT: Level = 11        . Count = 2         . 25.00     %
// CHECK-NEXT: Level = 13        . Count = 2         . 31.25     %
// CHECK-NEXT: Level = 17        . Count = 2         . 37.50     %
// CHECK-NEXT: Level = 19        . Count = 2         . 43.75     %
// CHECK-NEXT: Level = 23        . Count = 2         . 50.00     %
// CHECK-NEXT: Level = 25        . Count = 2         . 56.25     %
// CHECK-NEXT: Level = 27        . Count = 2         . 62.50     %
// CHECK-NEXT: Level = 30        . Count = 2         . 68.75     %
// CHECK-NEXT: Level = 32        . Count = 2         . 75.00     %
// CHECK-NEXT: Level = 35        . Count = 2         . 81.25     %
// CHECK-NEXT: Level = 37        . Count = 2         . 87.50     %
// CHECK-NEXT: Level = 38        . Count = 2         . 93.75     %
// CHECK-NEXT: Level = 48        . Count = 2         . 100.00    %
hw.module @passthrough(in %a: i16, out result: i16) {
    hw.output %a : i16
}

hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %add, %clk : i16
    %add = comb.mul %reg, %a : i16
    %result = hw.instance "passthrough" @passthrough(a: %add: i16) -> (result: i16)
    hw.output %result : i16
}
