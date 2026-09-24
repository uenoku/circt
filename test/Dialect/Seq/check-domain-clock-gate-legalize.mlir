// RUN: circt-opt --seq-check-domain='module-name=Top allow-crossing' %s | FileCheck %s

hw.module.extern private @EICG_wrapper(
  in %in: i1, in %test_en: i1, in %en: i1, out out: i1
) attributes {verilogName = "EICG_wrapper"}

hw.module @Top(in %clock: !seq.clock, in %enable: i1, in %in: i8,
              out out: i8) {
  %gate = hw.instance "gate/impl" @EICG_wrapper(
    in: %enable: i1, test_en: %enable: i1, en: %enable: i1
  ) -> (out: i1)
  %gated_clock = seq.to_clock %gate
  %reg = seq.compreg %in, %gated_clock : i8
  seq.check_clock_domain %reg, %clock : i8, !seq.clock
  hw.output %reg : i8
}

// CHECK-LABEL: hw.module @Top
// CHECK: hw.instance "gate/impl" @EICG_wrapper
// CHECK-LABEL: hw.module @Top_flatten
// CHECK-NOT: hw.instance "gate/impl" @EICG_wrapper
// CHECK: seq.clock_gate
