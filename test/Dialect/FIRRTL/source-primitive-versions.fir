; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s

; Import intentionally requires 7.0.0, even though exporting bool_and is
; allowed from 6.0.0 (see source-primitive-versions.mlir).
FIRRTL version 6.0.0
circuit Test :
  public module Test :
    input a : Bool
    input b : Bool
    output out : Bool
    ; expected-error @below {{bool_and are a FIRRTL 7.0.0+ feature}}
    propassign out, bool_and(a, b)
