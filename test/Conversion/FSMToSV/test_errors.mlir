// RUN: circt-opt --pass-pipeline='builtin.module(hw.design(convert-fsm-to-sv))' -split-input-file -verify-diagnostics %s

hw.design {
fsm.machine @foo(%arg0: i1) -> (i1) attributes {initialState = "A"} {
  // expected-error@+1 {{'arith.constant' op is unsupported (op from the arith dialect).}}
  %true = arith.constant true
  fsm.state @A output  {
    fsm.output %true : i1
  } transitions {
    fsm.transition @A
  }

  fsm.state @B output  {
    fsm.output %true : i1
  } transitions {
    fsm.transition @A
  }
}
}