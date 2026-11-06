// RUN: circt-translate --export-firrtl --firrtl-version=6.0.0 %s | FileCheck %s
// RUN: circt-translate --export-firrtl --firrtl-version=5.0.0 --verify-diagnostics %s

firrtl.circuit "Test" {
  firrtl.module @Test(in %a : !firrtl.bool, in %b : !firrtl.bool,
                      in %r : !firrtl.uint<1>, out %out : !firrtl.bool) {
    // asReset has an import gate but no export gate.
    %reset = firrtl.asReset %r : (!firrtl.uint<1>) -> !firrtl.reset
    %named = firrtl.node %reset : !firrtl.reset
    // expected-error @below {{'firrtl.bool.and' op boolean and requires FIRRTL 6.0.0}}
    %0 = firrtl.bool.and %a, %b
    firrtl.propassign %out, %0 : !firrtl.bool
  }
}

// CHECK: FIRRTL version 6.0.0
// CHECK: node named = asReset(r)
// CHECK: propassign out, bool_and(a, b)
