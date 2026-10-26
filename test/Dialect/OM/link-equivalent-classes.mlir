// RUN: circt-opt %s --om-link-modules --split-input-file | FileCheck %s --implicit-check-not=om.class.extern
// RUN: circt-opt %s --om-link-modules --split-input-file --mlir-disable-threading | FileCheck %s --implicit-check-not=om.class.extern

// Coalesce equivalent definitions, including recursive container types, while
// ignoring operation locations. External declarations still resolve normally.
// CHECK-LABEL: om.class @A(
// CHECK-SAME: !om.list<!om.class.type<@B>>
// CHECK-LABEL: om.class @B(
// CHECK-SAME: !om.class.type<@A>
// CHECK-NOT: om.class @
module {
  module {
    om.class @A(%b: !om.list<!om.class.type<@B>>) -> (b: !om.list<!om.class.type<@B>>) {
      om.class.fields %b : !om.list<!om.class.type<@B>>
    } loc("first")
    om.class @B(%a: !om.class.type<@A>) -> (a: !om.class.type<@A>) { om.class.fields %a : !om.class.type<@A> }
  }
  module {
    om.class @A(%b: !om.list<!om.class.type<@B>>) -> (b: !om.list<!om.class.type<@B>>) {
      om.class.fields %b : !om.list<!om.class.type<@B>>
    } loc("second")
    om.class @B(%a: !om.class.type<@A>) -> (a: !om.class.type<@A>) { om.class.fields %a : !om.class.type<@A> }
  }
  module {
    om.class.extern @A(%b: !om.list<!om.class.type<@B>>) -> (b: !om.list<!om.class.type<@B>>) {}
  }
}

// -----

// Different parameter names, visibility, or field mappings prevent coalescing.
// Private collisions also prevent coalescing recursive dependent class types.
// CHECK-LABEL: om.class @Names_A
// CHECK-LABEL: om.class public @Visibility_A
// CHECK-LABEL: om.class @Mapping_A
// CHECK-LABEL: om.class private @Leaf_A
// CHECK-LABEL: om.class @List_A
// CHECK-SAME: !om.list<!om.class.type<@Leaf_A>>
// CHECK-LABEL: om.class @Cycle_A
// CHECK-SAME: !om.class.type<@List_A>
// CHECK-LABEL: om.class @Names_B
// CHECK-LABEL: om.class @Visibility_B
// CHECK-LABEL: om.class @Mapping_B
// CHECK-LABEL: om.class private @Leaf_B
// CHECK-LABEL: om.class @List_B
// CHECK-SAME: !om.list<!om.class.type<@Leaf_B>>
// CHECK-LABEL: om.class @Cycle_B
// CHECK-SAME: !om.class.type<@List_B>
module {
  module attributes {om.namespace = "A"} {
    om.class @Names(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class public @Visibility() { om.class.fields }
    om.class @Mapping(%x: i1, %y: i1) -> (value: i1) { om.class.fields %x : i1 }
    om.class private @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @List(%cycle: !om.class.type<@Cycle>, %items: !om.list<!om.class.type<@Leaf>>) { om.class.fields }
    om.class @Cycle(%list: !om.class.type<@List>) { om.class.fields }
  }
  module attributes {om.namespace = "B"} {
    om.class @Names(%y: i1) -> (x: i1) { om.class.fields %y : i1 }
    om.class @Visibility() { om.class.fields }
    om.class @Mapping(%x: i1, %y: i1) -> (value: i1) { om.class.fields %y : i1 }
    om.class private @Leaf(%x: i1) -> (x: i1) { om.class.fields %x : i1 }
    om.class @List(%cycle: !om.class.type<@Cycle>, %items: !om.list<!om.class.type<@Leaf>>) { om.class.fields }
    om.class @Cycle(%list: !om.class.type<@List>) { om.class.fields }
  }
}

// -----

// A unique computational definition does not prevent coalescing its users.
// CHECK-LABEL: om.class @Holder(
// CHECK-SAME: !om.class.type<@Helper>
// CHECK-LABEL: om.class private @Helper()
// CHECK-NOT: om.class @Holder
module {
  module {
    om.class @Holder(%obj: !om.class.type<@Helper>) { om.class.fields }
    om.class private @Helper() -> (x: i1) { %x = om.constant false om.class.fields %x : i1 }
  }
  module {
    om.class.extern private @Helper() -> (x: i1) {}
    om.class @Holder(%obj: !om.class.type<@Helper>) { om.class.fields }
  }
}
