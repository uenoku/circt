// RUN: circt-opt %s --om-link-modules --split-input-file --verify-diagnostics

// Mixing linkage contracts is an error, with or without external declarations.
module {
  module {
    // expected-note @+1 {{previous definition is here}}
    om.class @Mixed() attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields }
  }
  module {
    // expected-error @+1 {{cannot link class "Mixed" with different linkage}}
    om.class @Mixed() { om.class.fields }
  }
}

// -----

module {
  module {
    om.class.extern @Mixed() {}
  }
  module {
    // expected-note @+1 {{previous definition is here}}
    om.class @Mixed() { om.class.fields }
  }
  module {
    // expected-error @+1 {{cannot link class "Mixed" with different linkage}}
    om.class @Mixed() attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields }
  }
}

// -----

// Coalescing does not disable declaration field checking.
module {
  module {
    // expected-error @+1 {{failed to link class "Fields" since declaration doesn't match the definition: declaration has a field "value" but types don't match, 'i1' vs 'i2'}}
    om.class.extern @Fields(%x: i1) -> (value: i2) {}
  }
  module {
    // expected-note @+1 {{definition is here}}
    om.class @Fields(%x: i1) -> (value: i1) attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields %x : i1 }
  }
  module {
    om.class @Fields(%x: i1) -> (value: i1) attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields %x : i1 }
  }
}

// -----

// Use generic syntax to exercise MLIR's standard inherent-property validation.
// expected-error @+1 {{Invalid attribute `linkage` in property conversion}}
"om.class"() <{sym_name = "Bad", formalParamNames = [], fieldNames = [], fieldTypes = {}, linkage = "linkonce_odr"}> ({
  om.class.fields
}) : () -> ()

// -----

// expected-error @+2 {{expected ::circt::om::Linkage to be one of: external, linkonce_odr}}
// expected-error @+1 {{failed to parse LinkageAttr parameter 'value'}}
om.class @Bad() attributes {linkage = #om.linkage<weak>} { om.class.fields }
