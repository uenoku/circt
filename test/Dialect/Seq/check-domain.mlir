// RUN: circt-opt --seq-check-domain='module-name=Top' %s | FileCheck %s

// The checked signal is driven through combinational logic by a register in
// the requested clock domain. A private child module is flattened first.
// CHECK-LABEL: hw.module @Top
// CHECK: hw.instance
hw.module @Top(in %clock: !seq.clock, in %in: i8) {
  %value = hw.instance "child" @Child(clock: %clock: !seq.clock, in: %in: i8) -> (out: i8)
  seq.check_clock_domain %value, %clock : i8, !seq.clock
}

// CHECK-LABEL: hw.module @Top_flatten
// CHECK-NOT: hw.instance
hw.module private @Child(in %clock: !seq.clock, in %in: i8, out out: i8) {
  %reg = seq.compreg %in, %clock : i8
  %one = hw.constant 1 : i8
  %out = comb.add %reg, %one : i8
  hw.output %out : i8
}
