// RUN: circt-opt %s --om-link-modules --split-input-file | FileCheck %s --implicit-check-not=om.class.extern

// Coalescing is an ODR contract, not a comparison of bodies. These equivalent
// implementations have different operations and SSA names. The first is kept.
// CHECK-LABEL: om.class @Compute
// CHECK-SAME: attributes {linkage = #om.linkage<linkonce_odr>}
// CHECK-NEXT: om.class.fields %x : !om.integer
// CHECK-NOT: om.class @Compute
// CHECK-NOT: om.integer.add
module {
  module {
    om.class @Compute(%x: !om.integer) -> (value: !om.integer) attributes {linkage = #om.linkage<linkonce_odr>} {
      om.class.fields %x : !om.integer
    }
  }
  module {
    om.class @Compute(%x: !om.integer) -> (value: !om.integer) attributes {linkage = #om.linkage<linkonce_odr>} {
      %zero = om.constant #om.integer<0>
      %result = om.integer.add %x, %zero : !om.integer
      om.class.fields %result : !om.integer
    }
  }
  module {
    om.class.extern @Compute(%x: !om.integer) -> (value: !om.integer) {}
  }
  module {
    om.class.extern @Compute(%x: !om.integer) -> (value: !om.integer) {}
  }
}

// -----

// Linkage and symbol visibility are independent. All three visibilities can
// coalesce, while even identical private definitions with default linkage are
// renamed. This case has no external declarations.
// CHECK-LABEL: om.class @Public()
// CHECK-SAME: attributes {linkage = #om.linkage<linkonce_odr>}
// CHECK-LABEL: om.class private @Private()
// CHECK-SAME: attributes {linkage = #om.linkage<linkonce_odr>}
// CHECK-LABEL: om.class nested @Nested()
// CHECK-SAME: attributes {linkage = #om.linkage<linkonce_odr>}
// CHECK-LABEL: om.class private @Local_A()
// CHECK-LABEL: om.class private @Local_B()
// CHECK-NOT: om.class{{[ ]}}
module {
  module attributes {om.namespace = "A"} {
    om.class @Public() attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields }
    om.class private @Private() attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields }
    om.class nested @Nested() attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields }
    om.class private @Local() { om.class.fields }
  }
  module attributes {om.namespace = "B"} {
    om.class @Public() attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields }
    om.class private @Private() attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields }
    om.class nested @Nested() attributes {linkage = #om.linkage<linkonce_odr>} { om.class.fields }
    om.class private @Local() { om.class.fields }
  }
}

// -----

// A single linkonce_odr definition resolves external declarations normally,
// including private declarations. Its linkage is preserved for further links.
// CHECK-LABEL: om.class private @Single
// CHECK-SAME: attributes {linkage = #om.linkage<linkonce_odr>}
// CHECK-NOT: om.class{{[ ]}}
module {
  module {
    om.class.extern private @Single(%x: i1) -> (value: i1) {}
  }
  module {
    om.class private @Single(%x: i1) -> (value: i1) attributes {linkage = #om.linkage<linkonce_odr>} {
      om.class.fields %x : i1
    }
  }
}
