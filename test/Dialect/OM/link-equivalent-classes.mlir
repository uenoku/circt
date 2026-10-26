// RUN: circt-opt %s --om-link-modules --split-input-file | FileCheck %s --implicit-check-not=om.class.extern
// RUN: circt-opt %s --om-link-modules --split-input-file --mlir-disable-threading | FileCheck %s --implicit-check-not=om.class.extern

// Equivalent field-only definitions share a name, including class types inside
// containers. Operation locations do not affect equivalence.
// CHECK-LABEL: om.class @Record(
// CHECK-LABEL: om.class @List(
// CHECK-SAME: !om.list<!om.class.type<@Record>>
// CHECK-NOT: om.class @
module {
  module {
    om.class @Record(%x: i1) -> (x: i1) { om.class.fields %x : i1 } loc("first")
    om.class @List(%items: !om.list<!om.class.type<@Record>>) -> (items: !om.list<!om.class.type<@Record>>) {
      om.class.fields %items : !om.list<!om.class.type<@Record>>
    }
  }
  module {
    om.class @Record(%x: i1) -> (x: i1) { om.class.fields %x : i1 } loc("second")
    om.class @List(%items: !om.list<!om.class.type<@Record>>) -> (items: !om.list<!om.class.type<@Record>>) {
      om.class.fields %items : !om.list<!om.class.type<@Record>>
    }
  }
  module {
    om.class.extern @Record(%x: i1) -> (x: i1) {}
  }
}

// -----

// Parameter names, visibility, and field-to-argument mappings must match.
// Computational and private definitions are not coalesced, nor are definitions
// whose class types refer to such colliding definitions.
// CHECK-LABEL: om.class @Names_A
// CHECK-LABEL: om.class public @Visibility_A
// CHECK-LABEL: om.class @Mapping_A
// CHECK-LABEL: om.class @Computed_A
// CHECK-LABEL: om.class private @Leaf_A
// CHECK-LABEL: om.class @List_A
// CHECK-SAME: !om.list<!om.class.type<@Leaf_A>>
// CHECK-LABEL: om.class @Names_B
// CHECK-LABEL: om.class @Visibility_B
// CHECK-LABEL: om.class @Mapping_B
// CHECK-LABEL: om.class @Computed_B
// CHECK-LABEL: om.class private @Leaf_B
// CHECK-LABEL: om.class @List_B
// CHECK-SAME: !om.list<!om.class.type<@Leaf_B>>
module {
  module attributes {om.namespace = "A"} {
    om.class @Names(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class public @Visibility() { om.class.fields }
    om.class @Mapping(%x: i1, %y: i1) -> (value: i1) { om.class.fields %x : i1 }
    om.class @Computed() -> (x: i1) { %x = om.constant false om.class.fields %x : i1 }
    om.class private @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @List(%items: !om.list<!om.class.type<@Leaf>>) {
      om.class.fields
    }
  }
  module attributes {om.namespace = "B"} {
    om.class @Names(%y: i1) -> (x: i1) { om.class.fields %y : i1 }
    om.class @Visibility() { om.class.fields }
    om.class @Mapping(%x: i1, %y: i1) -> (value: i1) { om.class.fields %y : i1 }
    om.class @Computed() -> (x: i1) { %x = om.constant false om.class.fields %x : i1 }
    om.class private @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @List(%items: !om.list<!om.class.type<@Leaf>>) {
      om.class.fields
    }
  }
}

// -----

// A unique definition always resolves to one entity, even if its constructor
// computes values. Field-only classes referring to it can still be coalesced.
// CHECK-LABEL: om.class @Holder(
// CHECK-SAME: !om.class.type<@Helper>
// CHECK-LABEL: om.class private @Helper()
// CHECK-NOT: om.class @Holder
module {
  module {
    om.class @Holder(%obj: !om.class.type<@Helper>) -> (obj: !om.class.type<@Helper>) {
      om.class.fields %obj : !om.class.type<@Helper>
    }
    om.class private @Helper() -> (x: i1) { %x = om.constant false om.class.fields %x : i1 }
  }
  module {
    om.class.extern private @Helper() -> (x: i1) {}
    om.class @Holder(%obj: !om.class.type<@Helper>) -> (obj: !om.class.type<@Helper>) {
      om.class.fields %obj : !om.class.type<@Helper>
    }
  }
}

// -----

// Recursive references terminate and retain their nominal types.
// CHECK-LABEL: om.class @A(
// CHECK-SAME: !om.class.type<@B>
// CHECK-LABEL: om.class @B(
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
