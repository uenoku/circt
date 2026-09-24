// RUN: circt-opt --seq-check-domain='module-name=Top' -verify-diagnostics %s

hw.module.extern private @OpaqueClock(out out: !seq.clock)

hw.module @Top(in %expected: !seq.clock, in %in: i8, out out: i8) {
  %clock = hw.instance "clock/instance" @OpaqueClock() -> (out: !seq.clock) {
    sv.namehint = "misleading/clock/name"
  }
  %reg = seq.compreg %in, %clock : i8
  // expected-error @below {{value 'clock/instance' (defined by 'hw.instance')}}
  seq.check_clock_domain %reg, %expected : i8, !seq.clock
  hw.output %reg : i8
}
