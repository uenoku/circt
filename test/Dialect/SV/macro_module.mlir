// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --export-verilog | FileCheck %s --check-prefix=VERILOG

sv.macro.decl @WHICH_MODULE

// CHECK-LABEL: sv.macro.module @MyModule macro @WHICH_MODULE(in %a : i1, out c : i1)
sv.macro.module @MyModule macro @WHICH_MODULE(in %a: i1, out c: i1)

// CHECK-LABEL: sv.macro.module @MyModule2 macro @WHICH_MODULE(in %x : i4, in %y : i4, out z : i8)
sv.macro.module @MyModule2 macro @WHICH_MODULE(in %x: i4, in %y: i4, out z: i8)

// Test that it can be instantiated like a regular module
hw.module @Top(in %a: i1, out c: i1) {
  // CHECK: %inst.c = hw.instance "inst" @MyModule(a: %a: i1) -> (c: i1)
  %0 = hw.instance "inst" @MyModule(a: %a: i1) -> (c: i1)
  hw.output %0 : i1
}

// VERILOG-LABEL: module Top
// VERILOG: `WHICH_MODULE inst (
// VERILOG-NEXT: .a (a),
// VERILOG-NEXT: .c (c)
// VERILOG-NEXT: );

