// RUN: circt-opt %s --canonicalize --cse | FileCheck %s

firrtl.circuit "Collections" {
  // Remove unused allocations, but do not merge equal allocations or pushes.
  // CHECK-LABEL: firrtl.module @Collections
  // CHECK: %[[ONE:.*]] = firrtl.integer 1
  // CHECK-NEXT: %[[FIRST:.*]] = firrtl.prop.collection : !firrtl.mutable_collection<integer>
  // CHECK-NEXT: %[[SECOND:.*]] = firrtl.prop.collection : !firrtl.mutable_collection<integer>
  // CHECK-NOT: firrtl.prop.collection
  // CHECK: firrtl.prop.push %[[FIRST]], %[[ONE]]
  // CHECK-NEXT: firrtl.prop.push %[[FIRST]], %[[ONE]]
  // CHECK: firrtl.propassign %a, %[[FIRST]]
  // CHECK: firrtl.propassign %b, %[[SECOND]]
  firrtl.module @Collections(out %a: !firrtl.mutable_collection<integer>, out %b: !firrtl.mutable_collection<integer>) {
    %first = firrtl.prop.collection : !firrtl.mutable_collection<integer>
    %second = firrtl.prop.collection : !firrtl.mutable_collection<integer>
    %unused = firrtl.prop.collection : !firrtl.mutable_collection<integer>
    %one = firrtl.integer 1
    firrtl.prop.push %first, %one : !firrtl.mutable_collection<integer>, !firrtl.integer
    firrtl.prop.push %first, %one : !firrtl.mutable_collection<integer>, !firrtl.integer
    firrtl.propassign %a, %first : !firrtl.mutable_collection<integer>
    firrtl.propassign %b, %second : !firrtl.mutable_collection<integer>
  }
}
