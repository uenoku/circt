; RUN: circt-translate --import-firrtl --verify-diagnostics --split-input-file %s

FIRRTL version 7.0.0
circuit Test :
  public module Test :
    input clock : Clock
    input enable : UInt<1>
    ; expected-error @below {{expected format string in printf}}
    printf(clock, enable, enable)

;// -----

FIRRTL version 7.0.0
circuit Test :
  public module Test :
    input a : Analog<1>
    ; expected-error @below {{expected operand in attach}}
    attach(a, )
