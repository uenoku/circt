// RUN: circt-opt --seq-check-domain='module-name=Top' -verify-diagnostics %s

hw.module @Top(in %expected: !seq.clock, in %actual: !seq.clock, in %in: i8,
              in %in2: i8, out out: i8) {
  %reg = seq.compreg %in, %actual : i8
  // expected-error @below {{input depends on a sequential element clocked by module input 'actual', not the expected clock module input 'expected'}}
  seq.check_clock_domain %reg, %expected : i8, !seq.clock

  %actual_wire = hw.wire %actual sym @actual_clock name "actual_clock" : !seq.clock
  %named_reg = seq.compreg %in, %actual_wire : i8
  // expected-error @below {{input depends on a sequential element clocked by module input 'actual', not the expected clock module input 'expected'}}
  seq.check_clock_domain %named_reg, %expected : i8, !seq.clock

  // The checked value is the input of a sequential element through
  // combinational logic. Users must be considered as clock-domain boundaries
  // as well.
  %user_comb = comb.add %in2, %in2 : i8
  %user_comb2 = comb.add %user_comb, %in2 : i8
  %user_reg = seq.compreg %user_comb2, %actual : i8
  // expected-error @below {{input depends on a sequential element clocked by module input 'actual', not the expected clock module input 'expected'}}
  seq.check_clock_domain %user_comb, %expected : i8, !seq.clock
  hw.output %user_reg : i8
}
