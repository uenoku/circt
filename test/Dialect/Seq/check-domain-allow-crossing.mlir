// RUN: circt-opt --seq-check-domain='module-name=Top allow-crossing' %s -o %t

hw.module @Top(in %expected: !seq.clock, in %actual: !seq.clock, in %in: i8,
              out out: i8) {
  %reg = seq.compreg %in, %actual : i8
  seq.check_clock_domain %reg, %expected : i8, !seq.clock
  hw.output %reg : i8
}
