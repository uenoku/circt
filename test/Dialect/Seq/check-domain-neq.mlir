// RUN: circt-opt --seq-check-domain='module-name=Top' -verify-diagnostics %s

hw.module @Top(in %expected: !seq.clock, in %actual: !seq.clock, in %in: i8,
              out out: i8) {
  %different = seq.compreg %in, %actual : i8
  seq.check_clock_domain_neq %different, %expected : i8, !seq.clock

  %same = seq.compreg %in, %expected : i8
  // expected-error @below {{input depends on a sequential element clocked by module input 'expected', but it must not depend on the expected clock module input 'expected'}}
  seq.check_clock_domain_neq %same, %expected : i8, !seq.clock

  hw.output %same : i8
}
