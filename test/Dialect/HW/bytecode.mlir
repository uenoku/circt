// RUN: circt-opt %s -emit-bytecode | circt-dis | FileCheck %s

sv.macro.decl @FOO

// CHECK-LABEL: sv.ifdef @FOO {
sv.ifdef @FOO {
}

// CHECK-LABEL: hw.module @Child(in %in : !hw.struct<a: i1>, inout %bus : i1, out out : !hw.struct<a: i1>) {
hw.module @Child(in %in: !hw.struct<a: i1>, inout %bus: i1, out out: !hw.struct<a: i1>) {
  // CHECK: hw.wire %in sym [<@wire,1,private>] : !hw.struct<a: i1>
  %wire = hw.wire %in sym [<@wire, 1, private>] : !hw.struct<a: i1>
  // CHECK: sv.wire : !hw.inout<i1>
  %direct_inout = sv.wire : !hw.inout<i1>
  hw.output %wire : !hw.struct<a: i1>
}

// CHECK-LABEL: hw.module @Top
hw.module @Top() {
  // CHECK: sv.verbatim "ref {{.*}}" {symbols = [#hw.innerNameRef<@Child::@wire>]}
  sv.verbatim "ref {{0}}" {symbols = [#hw.innerNameRef<@Child::@wire>]}
  hw.output
}
