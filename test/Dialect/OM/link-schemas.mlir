// RUN: circt-opt %s --verify-diagnostics --om-link-modules --split-input-file | FileCheck %s --implicit-check-not=om.class.extern
// RUN: circt-opt %s --verify-diagnostics --om-link-modules --split-input-file --mlir-disable-threading | FileCheck %s --implicit-check-not=om.class.extern

// Domain artifacts referenced by an external class must share their types. The
// base path need not be a field, and constructor argument names may differ.
// CHECK-LABEL: om.class @Testbench
// CHECK-SAME: !om.class.type<@ClockDomain>
// CHECK-SAME: -> (d_out: !om.class.type<@ClockDomain_out>)
// CHECK: om.object @Dut(%{{.+}}, %{{.+}}) : (!om.frozenbasepath, !om.class.type<@ClockDomain>) -> !om.class.type<@Dut>
// CHECK-LABEL: om.class @ClockDomain_out
// CHECK-SAME: !om.class.type<@ClockDomain>
// CHECK-LABEL: om.class @ClockDomain(
// CHECK-LABEL: om.class @Dut
// CHECK-SAME: !om.class.type<@ClockDomain>
// CHECK: om.elaborated_object @ClockDomain_out
// CHECK-NOT: om.class @ClockDomain
// CHECK-NOT: om.class.extern
module {
  module {
    om.class.extern private @Dut(%base: !om.frozenbasepath, %d: !om.class.type<@ClockDomain>) -> (d_out: !om.class.type<@ClockDomain_out>) {}
    om.class @Testbench(%base: !om.frozenbasepath, %d: !om.class.type<@ClockDomain>) -> (d_out: !om.class.type<@ClockDomain_out>) {
      %dut = om.object @Dut(%base, %d) : (!om.frozenbasepath, !om.class.type<@ClockDomain>) -> !om.class.type<@Dut>
      %out = om.object.field %dut["d_out"] : (!om.class.type<@Dut>) -> !om.class.type<@ClockDomain_out>
      om.class.fields %out : !om.class.type<@ClockDomain_out>
    }
    om.class @ClockDomain_out(%base: !om.frozenbasepath, %d: !om.class.type<@ClockDomain>, %paths: !om.list<!om.frozenpath>) -> (domainInfo_out: !om.class.type<@ClockDomain>, associations_out: !om.list<!om.frozenpath>) {
      om.class.fields %d, %paths : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
    }
    om.class @ClockDomain(%base: !om.frozenbasepath, %name: !om.string) -> (name_out: !om.string) {
      om.class.fields %name : !om.string
    }
  }
  module {
    om.class @Dut(%base: !om.frozenbasepath, %d: !om.class.type<@ClockDomain>) -> (d_out: !om.class.type<@ClockDomain_out>) {
      %paths = om.list_create : !om.frozenpath
      %out = om.elaborated_object @ClockDomain_out(%d, %paths) : (!om.class.type<@ClockDomain>, !om.list<!om.frozenpath>) -> !om.class.type<@ClockDomain_out>
      om.class.fields %out : !om.class.type<@ClockDomain_out>
    }
    om.class @ClockDomain(%basepath: !om.frozenbasepath, %name_in: !om.string) -> (name_out: !om.string) {
      om.class.fields %name_in : !om.string
    }
    om.class @ClockDomain_out(%basepath: !om.frozenbasepath, %domainInfo_in: !om.class.type<@ClockDomain>, %associations_in: !om.list<!om.frozenpath>) -> (domainInfo_out: !om.class.type<@ClockDomain>, associations_out: !om.list<!om.frozenpath>) {
      om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
    }
  }
}

// -----

// Empty schemas can have multiple definitions even when declared externally.
// CHECK-LABEL: om.class @Empty()
// CHECK-NOT: om.class @
module {
  module {
    om.class.extern @Empty() {}
  }
  module {
    om.class @Empty() { om.class.fields }
  }
  module {
    om.class @Empty() { om.class.fields }
  }
}

// -----

// Private schemas are not shared, nor are public schemas which depend on them.
// Dependencies nested in containers and unused arguments must also be checked.
// CHECK-LABEL: om.class private @Leaf_A
// CHECK-LABEL: om.class @List_A
// CHECK-SAME: !om.list<!om.class.type<@Leaf_A>>
// CHECK-SAME: -> (items: !om.list<!om.class.type<@Leaf_A>>)
// CHECK-LABEL: om.class @Wrapper_A
// CHECK-SAME: !om.class.type<@List_A>
// CHECK-LABEL: om.class private @Leaf_B
// CHECK-LABEL: om.class @List_B
// CHECK-SAME: !om.list<!om.class.type<@Leaf_B>>
// CHECK-SAME: -> (items: !om.list<!om.class.type<@Leaf_B>>)
// CHECK-LABEL: om.class @Wrapper_B
// CHECK-SAME: !om.class.type<@List_B>
module {
  module attributes {om.namespace = "A"} {
    om.class private @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @List(%items: !om.list<!om.class.type<@Leaf>>) -> (items: !om.list<!om.class.type<@Leaf>>) {
      om.class.fields %items : !om.list<!om.class.type<@Leaf>>
    }
    om.class @Wrapper(%unused: !om.class.type<@List>) { om.class.fields }
  }
  module attributes {om.namespace = "B"} {
    om.class private @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @List(%items: !om.list<!om.class.type<@Leaf>>) -> (items: !om.list<!om.class.type<@Leaf>>) {
      om.class.fields %items : !om.list<!om.class.type<@Leaf>>
    }
    om.class @Wrapper(%unused: !om.class.type<@List>) { om.class.fields }
  }
}

// -----

// The constructor-to-field mapping and field order must match. Merely having
// the same field names and types does not imply the same constructor semantics.
// CHECK-LABEL: om.class @Mapping_A
// CHECK-LABEL: om.class @Order_A
// CHECK-LABEL: om.class @Mapping_B
// CHECK-LABEL: om.class @Order_B
module {
  module attributes {om.namespace = "A"} {
    om.class @Mapping(%x: i1, %y: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @Order(%x: i1, %y: i1) -> (x: i1, y: i1) { om.class.fields %x, %y : i1, i1 }
  }
  module attributes {om.namespace = "B"} {
    om.class @Mapping(%x: i1, %y: i1) -> (x: i1) { om.class.fields %y : i1 }
    om.class @Order(%x: i1, %y: i1) -> (y: i1, x: i1) { om.class.fields %y, %x : i1, i1 }
  }
}

// -----

// Primitive type mismatches invalidate all dependent schemas. A computational
// body also prevents sharing, even when the computations are identical.
// CHECK-LABEL: om.class @Leaf_A
// CHECK-LABEL: om.class @Parent_A
// CHECK-SAME: !om.class.type<@Leaf_A>
// CHECK-LABEL: om.class @Computed_A
// CHECK-LABEL: om.class @Leaf_B
// CHECK-LABEL: om.class @Parent_B
// CHECK-SAME: !om.class.type<@Leaf_B>
// CHECK-LABEL: om.class @Computed_B
module {
  module attributes {om.namespace = "A"} {
    om.class @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @Parent(%x: !om.class.type<@Leaf>) -> (x: !om.class.type<@Leaf>) {
      om.class.fields %x : !om.class.type<@Leaf>
    }
    om.class @Computed() -> (x: i1) {
      %x = om.constant false
      om.class.fields %x : i1
    }
  }
  module attributes {om.namespace = "B"} {
    om.class @Leaf(%x: i2) -> (x: i2) { om.class.fields %x : i2 }
    om.class @Parent(%x: !om.class.type<@Leaf>) -> (x: !om.class.type<@Leaf>) {
      om.class.fields %x : !om.class.type<@Leaf>
    }
    om.class @Computed() -> (x: i1) {
      %x = om.constant false
      om.class.fields %x : i1
    }
  }
}

// -----

// Recursive schema references terminate and retain their nominal types.
// CHECK-LABEL: om.class @A
// CHECK-SAME: !om.class.type<@B>
// CHECK-LABEL: om.class @B
// CHECK-SAME: !om.class.type<@A>
// CHECK-NOT: om.class @
module {
  module {
    om.class @A(%b: !om.class.type<@B>) -> (b: !om.class.type<@B>) { om.class.fields %b : !om.class.type<@B> }
    om.class @B(%a: !om.class.type<@A>) -> (a: !om.class.type<@A>) { om.class.fields %a : !om.class.type<@A> }
  }
  module {
    om.class @A(%b: !om.class.type<@B>) -> (b: !om.class.type<@B>) { om.class.fields %b : !om.class.type<@B> }
    om.class @B(%a: !om.class.type<@A>) -> (a: !om.class.type<@A>) { om.class.fields %a : !om.class.type<@A> }
  }
}

// -----

// Sharing extends through container types and chains of schema references.
// CHECK-LABEL: om.class @Leaf(
// CHECK-LABEL: om.class @List(
// CHECK-SAME: !om.list<!om.class.type<@Leaf>>
// CHECK-LABEL: om.class @Wrapper(
// CHECK-SAME: !om.class.type<@List>
// CHECK-NOT: om.class @
module {
  module {
    om.class @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @List(%items: !om.list<!om.class.type<@Leaf>>) -> (items: !om.list<!om.class.type<@Leaf>>) {
      om.class.fields %items : !om.list<!om.class.type<@Leaf>>
    }
    om.class @Wrapper(%x: !om.class.type<@List>) -> (x: !om.class.type<@List>) {
      om.class.fields %x : !om.class.type<@List>
    }
  }
  module {
    om.class @Wrapper(%x: !om.class.type<@List>) -> (x: !om.class.type<@List>) {
      om.class.fields %x : !om.class.type<@List>
    }
    om.class @List(%items: !om.list<!om.class.type<@Leaf>>) -> (items: !om.list<!om.class.type<@Leaf>>) {
      om.class.fields %items : !om.list<!om.class.type<@Leaf>>
    }
    om.class @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
  }
}

// -----

// Unused arguments, field names, and additional class attributes are part of
// schema compatibility, even if the field types and values match.
// CHECK-LABEL: om.class @Arguments_A
// CHECK-LABEL: om.class @Names_A
// CHECK-LABEL: om.class @Attributes_A
// CHECK-LABEL: om.class @Arguments_B
// CHECK-LABEL: om.class @Names_B
// CHECK-LABEL: om.class @Attributes_B
module {
  module attributes {om.namespace = "A"} {
    om.class @Arguments(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @Names(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @Attributes(%x: i1) -> (x: i1) attributes {schema.tag = "a"} { om.class.fields %x : i1 }
  }
  module attributes {om.namespace = "B"} {
    om.class @Arguments(%x: i1, %unused: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @Names(%x: i1) -> (y: i1) { om.class.fields %x : i1 }
    om.class @Attributes(%x: i1) -> (x: i1) attributes {schema.tag = "b"} { om.class.fields %x : i1 }
  }
}
