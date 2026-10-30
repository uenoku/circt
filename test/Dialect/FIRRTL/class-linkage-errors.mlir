// RUN: circt-opt %s --split-input-file --verify-diagnostics

firrtl.circuit "Top" {
  // Use generic syntax to exercise MLIR's standard inherent-property validation.
  // expected-error @+1 {{Invalid attribute `linkage` in property conversion}}
  "firrtl.class"() <{sym_name = "Bad", linkage = "linkonce_odr"}> ({}) : () -> ()
  firrtl.module @Top() {}
}

// -----

firrtl.circuit "Top" {
  // expected-error @+2 {{expected ::circt::firrtl::ClassLinkage to be one of: external, linkonce_odr}}
  // expected-error @+1 {{failed to parse ClassLinkageAttr parameter 'value'}}
  firrtl.class @Bad() attributes {linkage = #firrtl.class_linkage<weak>} {}
  firrtl.module @Top() {}
}
