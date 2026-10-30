// RUN: circt-opt %s --om-freeze-paths | FileCheck %s

// Unknown membership and nested path types must be frozen as well.
// CHECK-LABEL: om.class @Unknowns()
// CHECK-SAME: -> (items: !om.collection<!om.frozenpath>, nested: !om.collection<!om.list<!om.collection<!om.frozenbasepath>>>, path: !om.frozenpath)
om.class @Unknowns() -> (items: !om.collection<!om.path>, nested: !om.collection<!om.list<!om.collection<!om.basepath>>>, path: !om.path) {
  // CHECK: %[[ITEMS:.*]] = om.unknown {test = "keep"} : !om.collection<!om.frozenpath>
  %items = om.unknown {test = "keep"} : !om.collection<!om.path>
  // CHECK: %[[NESTED:.*]] = om.unknown : !om.collection<!om.list<!om.collection<!om.frozenbasepath>>>
  %nested = om.unknown : !om.collection<!om.list<!om.collection<!om.basepath>>>
  // CHECK: %[[PATH:.*]] = om.unknown : !om.frozenpath
  %path = om.unknown : !om.path
  // CHECK: om.class.fields %[[ITEMS]], %[[NESTED]], %[[PATH]]
  om.class.fields %items, %nested, %path : !om.collection<!om.path>, !om.collection<!om.list<!om.collection<!om.basepath>>>, !om.path
}

// Result types can be updated in place without discarding field-access attributes.
// CHECK-LABEL: om.class @Field(%object: !om.class.type<@Unknowns>)
// CHECK-SAME: -> (items: !om.collection<!om.frozenpath>)
om.class @Field(%object: !om.class.type<@Unknowns>) -> (items: !om.collection<!om.path>) {
  // CHECK: %[[FIELD:.*]] = om.object.field %object["items"] {test = "keep"} : (!om.class.type<@Unknowns>) -> !om.collection<!om.frozenpath>
  %items = om.object.field %object["items"] {test = "keep"} : (!om.class.type<@Unknowns>) -> !om.collection<!om.path>
  // CHECK: om.class.fields %[[FIELD]]
  om.class.fields %items : !om.collection<!om.path>
}

// CHECK-LABEL: om.class @Paths(%input: !om.collection<!om.frozenpath>)
// CHECK-SAME: -> (items: !om.collection<!om.frozenpath>, nested: !om.collection<!om.list<!om.collection<!om.frozenpath>>>)
om.class @Paths(%input: !om.collection<!om.path>) -> (items: !om.collection<!om.path>, nested: !om.collection<!om.list<!om.collection<!om.path>>>) {
  // CHECK: %[[ITEMS:.*]] = om.collection {test = "keep"} : !om.collection<!om.frozenpath>
  %items = om.collection {test = "keep"} : !om.collection<!om.path>
  // CHECK: %[[PATH:.*]] = om.frozenpath_empty
  %path = om.path_empty
  // CHECK: om.collection_push %input, %[[PATH]] : !om.collection<!om.frozenpath>, !om.frozenpath
  om.collection_push %input, %path : !om.collection<!om.path>, !om.path
  // CHECK: om.collection_push %[[ITEMS]], %[[PATH]] : !om.collection<!om.frozenpath>, !om.frozenpath
  om.collection_push %items, %path : !om.collection<!om.path>, !om.path
  // CHECK: om.collection %[[PATH]] : !om.collection<!om.frozenpath>
  %initial = om.collection %path : !om.collection<!om.path>
  // CHECK: %[[NESTED:.*]] = om.collection : !om.collection<!om.list<!om.collection<!om.frozenpath>>>
  %nested = om.collection : !om.collection<!om.list<!om.collection<!om.path>>>
  // CHECK: %[[LIST:.*]] = om.list_create %[[ITEMS]] : !om.collection<!om.frozenpath>
  %list = om.list_create %items : !om.collection<!om.path>
  // CHECK: om.collection_push %[[NESTED]], %[[LIST]] : !om.collection<!om.list<!om.collection<!om.frozenpath>>>, !om.list<!om.collection<!om.frozenpath>>
  om.collection_push %nested, %list : !om.collection<!om.list<!om.collection<!om.path>>>, !om.list<!om.collection<!om.path>>
  om.class.fields %items, %nested : !om.collection<!om.path>, !om.collection<!om.list<!om.collection<!om.path>>>
}
