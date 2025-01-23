// RUN: circt-opt %s -pass-pipeline=builtin.module(hw.design(export-verilog)) --split-input-file | FileCheck %s

hw.design {
// CHECK-LABEL: Generated
// CHECK-NEXT: module Foo(
hw.module @Foo(in %a: i1 loc("")) {
  hw.output
}
}

// -----

hw.design attributes {circt.loweringOptions = "omitVersionComment"} {
// CHECK-NOT: Generated
// CHECK-LABEL: module Bar(
hw.module @Bar() {
  hw.output
}
}
