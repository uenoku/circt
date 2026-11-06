// RUN: circt-translate --export-firrtl --verify-diagnostics --split-input-file %s

// Tag extraction remains import-only until enum source emission is supported.
firrtl.circuit "TagExtract" {
  firrtl.module @TagExtract(in %a : !firrtl.uint<0>, out %out : !firrtl.uint<1>) {
    %enum = firrtl.enumcreate A(%a) : (!firrtl.uint<0>) -> !firrtl.enum<A, B>
    // expected-error @below {{'firrtl.tagextract' op not supported as expression}}
    %0 = firrtl.tagextract %enum : !firrtl.enum<A, B>
    firrtl.matchingconnect %out, %0 : !firrtl.uint<1>
  }
}
