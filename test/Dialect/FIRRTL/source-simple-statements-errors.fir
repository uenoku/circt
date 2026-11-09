; RUN: circt-translate --import-firrtl --verify-diagnostics --split-input-file %s

FIRRTL version 3.0.0
circuit Test :
  module Test :
    input clock : Clock
    input enable : UInt<1>
    ; expected-error @below {{expected condition in 'stop'}}
    stop(clock, , 1)

;// -----

FIRRTL version 3.0.0
circuit Test :
  module Test :
    input clock : Clock
    input enable : UInt<1>
    ; expected-error @below {{expected integer literal}}
    stop(clock, enable, enable)

;// -----

FIRRTL version 3.0.0
circuit Test :
  module Test :
    input clock : Clock
    input enable : UInt<1>
    ; expected-error @below {{expected ')'}}
    stop(clock, enable, 1, 2)
