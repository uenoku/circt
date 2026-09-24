// RUN: circt-opt --seq-check-domain='module-name=Top' -verify-diagnostics %s

hw.module.extern private @EICG_wrapper(
  in %in: i1, in %test_en: i1, in %en: i1, out out: i1
) attributes {verilogName = "EICG_wrapper"}

hw.module @Top(in %expected: !seq.clock, in %actual: i1, in %in: i8,
              out out: i8) {
  %gate = hw.instance "gate/impl" @EICG_wrapper(
    in: %actual: i1, test_en: %actual: i1, en: %actual: i1
  ) -> (out: i1) {sv.namehint = "misleading/gate/name"}
  %clock = seq.to_clock %gate
  %reg = seq.compreg %in, %clock : i8
  // expected-error @below {{input depends on a sequential element clocked by result of 'seq.clock_gate'}}
  seq.check_clock_domain %reg, %expected : i8, !seq.clock
  hw.output %reg : i8
}
