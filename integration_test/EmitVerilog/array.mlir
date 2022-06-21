// REQUIRES: verilator
// REQUIRES: yosys
// RUN: circt-opt -export-verilog %s -o /dev/null > %t.sv && cat %t.sv | FileCheck %s
// RUN: verilator --lint-only %t.sv

// CHECK-LABEL: StrurctExtractInline
hw.module @StrurctExtractInline(%a: !hw.struct<v: i1>) -> (b: i1, c: i1) {
  %0 = hw.struct_extract %a["v"] : !hw.struct<v: i1>
  // CHECK:      assign b = a.v;
  // CHECK-NEXT: assign c = a.v;
  hw.output %0, %0 : i1, i1
}