; RUN: circt-translate --import-firrtl --verify-diagnostics --split-input-file %s

; The handwritten import-only registration retains its version gate.
FIRRTL version 6.0.0
circuit Test :
  public module Test :
    input x : UInt<1>
    ; expected-error @below {{tagExtract are a FIRRTL 7.0.0+ feature}}
    node result = tagExtract(x)

;// -----

FIRRTL version 7.0.0
circuit Test :
  public module Test :
    input x : {|A, B|}
    output result : UInt<1>
    connect result, tagExtract(x)
