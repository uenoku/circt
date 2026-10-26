// RUN: circt-opt %s --verify-diagnostics --om-link-modules --split-input-file

// Sharing definitions must not bypass declaration compatibility checks.
module {
  module {
    // expected-error @+1 {{failed to link class "Schema" since declaration doesn't match the definition: 0-th argument type is not equal, 'i1' vs 'i2'}}
    om.class.extern @Schema(%x: i2) -> (x: i2) {}
  }
  module {
    // expected-note @+1 {{definition is here}}
    om.class @Schema(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
  }
  module {
    om.class @Schema(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
  }
}

// -----

// Incompatible field-only definitions remain an error when used externally.
module {
  module {
    // expected-error @+1 {{class "Schema" is declared as an external class but there are multiple definitions}}
    om.class.extern @Schema(%x: i1) -> (x: i1) {}
  }
  module {
    // expected-note @+1 {{class "Schema" is defined here}}
    om.class @Schema(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
  }
  module {
    // expected-note @+1 {{class "Schema" is defined here}}
    om.class @Schema(%x: i2) -> (x: i2) { om.class.fields %x : i2 }
  }
}
