; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s

; Both import and export require FIRRTL 6.0.0 for boolean operations.
FIRRTL version 5.0.0
circuit Test :
  public module Test :
    input a : UInt<1>
    ; expected-error @below {{bool_and are a FIRRTL 6.0.0+ feature}}
    node x = bool_and(a, a)

;// -----

FIRRTL version 5.0.0
circuit Test :
  public module Test :
    input a : UInt<1>
    node x = asReset(a)

;// -----

FIRRTL version 6.0.0
circuit Test :
  public module Test :
    input a : Bool
    input b : Bool
    output out : Bool
    propassign out, bool_and(a, b)
