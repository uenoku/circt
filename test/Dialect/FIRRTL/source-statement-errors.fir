; RUN: circt-translate --import-firrtl --verify-diagnostics --split-input-file %s

FIRRTL version 5.0.0
circuit Test :
  public module Test :
    input clock : Clock
    input enable : UInt<1>
    ; expected-error @below {{fprintf are a FIRRTL 6.0.0+ feature}}
    fprintf(clock, enable, "out.txt", "hello")

;// -----

FIRRTL version 5.0.0
circuit Test :
  public module Test :
    input clock : Clock
    input enable : UInt<1>
    ; expected-error @below {{fflush are a FIRRTL 6.0.0+ feature}}
    fflush(clock, enable)

;// -----

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
