; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s

; Import intentionally requires 7.0.0, even though exporting bool_and is
; allowed from 6.0.0 (see source-primitive-versions.mlir).
FIRRTL version 6.0.0
circuit Test :
  public module Test :
    input a : Bool
    input b : Bool
    output out : Bool
    ; expected-error @below {{Boolean and are a FIRRTL 7.0.0+ feature}}
    propassign out, bool_and(a, b)

;// -----

FIRRTL version 6.0.0
circuit Test :
  public module Test :
    input a : Integer
    input b : Integer
    output out : Integer
    ; expected-error @below {{Integers are a FIRRTL 7.0.0+ feature}}
    propassign out, integer_and(a, b)

;// -----

FIRRTL version 7.0.0
circuit Test :
  public module Test :
    input a : Bool
    input b : Bool
    output out : Bool
    propassign out, bool_and(a, b)
