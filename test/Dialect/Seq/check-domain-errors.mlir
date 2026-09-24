// RUN: circt-opt --seq-check-domain='module-name=Top' -verify-diagnostics %s

hw.module @Top(in %expected: !seq.clock, in %actual: !seq.clock, in %in: i8) {
  %reg = seq.compreg %in, %actual : i8
  // expected-error @below {{input depends on a sequential element clocked by module input 'actual', not the expected clock module input 'expected'}}
  seq.check_clock_domain %reg, %expected : i8, !seq.clock
}
