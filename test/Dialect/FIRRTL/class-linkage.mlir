// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --mlir-print-op-generic | circt-opt | FileCheck %s
// RUN: circt-opt %s --verify-roundtrip > /dev/null
// RUN: circt-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=PROPERTIES

firrtl.circuit "Top" {
  // CHECK-LABEL: firrtl.class @Default()
  // CHECK-SAME: {
  // CHECK-NOT: linkage
  firrtl.class @Default() {}

  // CHECK-LABEL: firrtl.class @External()
  // CHECK-SAME: {
  // CHECK-NOT: linkage
  firrtl.class @External() attributes {linkage = #firrtl.class_linkage<external>} {}

  // PROPERTIES: "firrtl.class"() <{domainInfo = [], linkage = #firrtl.class_linkage<linkonce_odr>,{{.*}}sym_name = "LinkOnce"
  // CHECK-LABEL: firrtl.class private @LinkOnce()
  // CHECK-SAME: attributes {linkage = #firrtl.class_linkage<linkonce_odr>}
  firrtl.class private @LinkOnce() attributes {linkage = #firrtl.class_linkage<linkonce_odr>} {}

  // CHECK-LABEL: firrtl.module @Top()
  firrtl.module @Top() {}
}
