// RUN: circt-opt -om-elaborate-object %s | FileCheck %s

om.class @Leaf() -> (x: !om.integer) {
  %c0 = om.constant #om.integer<42 : i6> : !om.integer
  om.class.fields %c0 : !om.integer
}

om.class @Inner() -> (leaf: !om.class.type<@Leaf>, x: !om.integer) {
  %leaf = om.object @Leaf() : () -> !om.class.type<@Leaf>
  %x = om.object.field %leaf[@x] : (!om.class.type<@Leaf>) -> !om.integer
  om.class.fields %leaf, %x : !om.class.type<@Leaf>, !om.integer
}

// CHECK: om.class @__om_elaborated_Leaf() -> (root: !om.class.type<@Leaf>) {
// CHECK:   %[[C0:.+]] = om.constant #om.integer<42 : i6> : !om.integer
// CHECK:   %[[ROOT0:.+]] = om.elaborated_object @Leaf(%[[C0]]) : (!om.integer) -> !om.class.type<@Leaf>
// CHECK:   om.class.fields %[[ROOT0]] : !om.class.type<@Leaf>
// CHECK: }

// CHECK: om.class @__om_elaborated_Inner() -> (root: !om.class.type<@Inner>) {
// CHECK:   %[[C1:.+]] = om.constant #om.integer<42 : i6> : !om.integer
// CHECK:   %[[LEAF:.+]] = om.elaborated_object @Leaf(%[[C1]]) : (!om.integer) -> !om.class.type<@Leaf>
// CHECK:   %[[ROOT1:.+]] = om.elaborated_object @Inner(%[[LEAF]], %[[C1]]) : (!om.class.type<@Leaf>, !om.integer) -> !om.class.type<@Inner>
// CHECK:   om.class.fields %[[ROOT1]] : !om.class.type<@Inner>
// CHECK: }
