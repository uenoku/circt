// RUN: circt-opt -om-elaborate-object='target-class=Top' %s | FileCheck %s

!list = !om.class.type<@LinkedList>

om.class @InputBox(%input: !om.integer) -> (value: !om.integer) {
  om.class.fields %input : !om.integer
}

om.class @LinkedList(%input: !om.integer, %next: !list) -> (value: !om.integer, next: !list) {
  %box = om.object @InputBox(%input) : (!om.integer) -> !om.class.type<@InputBox>
  %value = om.object.field %box[@value] : (!om.class.type<@InputBox>) -> !om.integer
  om.class.fields %value, %next : !om.integer, !list
}

om.class @Top() -> (list: !list, head: !om.integer, tail: !list) {
  %one = om.constant #om.integer<1 : i6> : !om.integer
  %two = om.constant #om.integer<2 : i6> : !om.integer
  %tail = om.object @LinkedList(%two, %list) : (!om.integer, !list) -> !list
  %list = om.object @LinkedList(%one, %tail) : (!om.integer, !list) -> !list
  %head = om.object.field %list[@value] : (!list) -> !om.integer
  om.class.fields %list, %head, %tail : !list, !om.integer, !list
}

om.class @Other() -> (unused: !om.integer) {
  %c0 = om.constant #om.integer<0 : i1> : !om.integer
  om.class.fields %c0 : !om.integer
}

// CHECK: om.class @__om_elaborated_Top() -> (root: !om.class.type<@Top>) {
// CHECK-DAG:   %[[TWO:.+]] = om.constant #om.integer<2 : i6> : !om.integer
// CHECK-DAG:   %[[ONE:.+]] = om.constant #om.integer<1 : i6> : !om.integer
// CHECK:   %{{.+}} = om.elaborated_object @Top(%{{.+}}, %[[ONE]], %{{.+}}) : (!om.class.type<@LinkedList>, !om.integer, !om.class.type<@LinkedList>) -> !om.class.type<@Top>
// CHECK:   %{{.+}} = om.elaborated_object @LinkedList(%[[TWO]], %{{.+}}) : (!om.integer, !om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
// CHECK:   %{{.+}} = om.elaborated_object @LinkedList(%[[ONE]], %{{.+}}) : (!om.integer, !om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
// CHECK:   om.class.fields %{{.+}} : !om.class.type<@Top>
// CHECK: }
// CHECK-NOT: om.class @__om_elaborated_Other()
