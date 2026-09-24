// RUN: circt-opt --seq-check-domain='module-name=Top skip-clock-gate' %s | FileCheck %s

hw.module.extern private @EICG_wrapper(
  in %in: !seq.clock, in %test_en: i1, in %en: i1, out out: !seq.clock
) attributes {verilogName = "EICG_wrapper"}

hw.module @Top(in %expected: !seq.clock, in %actual: !seq.clock,
              in %enable: i1, in %in: i8, out out: i8) {
  %gate = hw.instance "gate/impl" @EICG_wrapper(
    in: %actual: !seq.clock, test_en: %enable: i1, en: %enable: i1
  ) -> (out: !seq.clock)
  %reg = seq.compreg %in, %gate : i8
  seq.check_clock_domain %reg, %expected : i8, !seq.clock
  hw.output %reg : i8
}

// CHECK-LABEL: hw.module @Top
// CHECK: hw.instance "gate/impl" @EICG_wrapper
// CHECK-LABEL: hw.module @Top_flatten
// CHECK-NOT: hw.instance "gate/impl" @EICG_wrapper
// CHECK: seq.clock_gate
