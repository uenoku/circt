// RUN: circt-opt %s --firrtl-lower-domains --firrtl-lower-classes | FileCheck %s

// Domain information and association classes get the ODR contract. Ordinary
// FIRRTL classes and classes extracted from hardware modules retain default
// linkage unless explicitly marked.
// CHECK-DAG: om.class @ResetDomain({{.*}} -> (isCold_out: i1) attributes {linkage = #om.linkage<linkonce_odr>}
// CHECK-DAG: om.class @ResetDomain_out({{.*}} attributes {linkage = #om.linkage<linkonce_odr>}
// CHECK-DAG: om.class @Ordinary({{.*}} -> (value: i1) {
// CHECK-DAG: om.class @Explicit({{.*}} -> (value: i1) attributes {linkage = #om.linkage<linkonce_odr>}
// CHECK-DAG: om.class @Top_Class({{.*}} -> (value: i1) {
firrtl.circuit "Top" {
  firrtl.domain @ResetDomain [#firrtl.domain.field<"isCold", !firrtl.bool>]

  firrtl.class @Ordinary(in %value_in: !firrtl.bool, out %value: !firrtl.bool) {
    firrtl.propassign %value, %value_in : !firrtl.bool
  }

  firrtl.class @Explicit(in %value_in: !firrtl.bool, out %value: !firrtl.bool) attributes {linkage = #firrtl.class_linkage<linkonce_odr>} {
    firrtl.propassign %value, %value_in : !firrtl.bool
  }

  firrtl.module @Top(out %value: !firrtl.bool) {
    %true = firrtl.bool true
    firrtl.propassign %value, %true : !firrtl.bool
  }
}
