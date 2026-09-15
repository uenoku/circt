// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefix=CHECK
// RUN: circt-opt --test-apply-lowering-options='options=disallowReplicate' --export-verilog %s | FileCheck %s --check-prefix=DISALLOW

// CHECK-LABEL: module replicateOptOut
// DISALLOW-LABEL: module replicateOptOut
hw.module @replicateOptOut(in %arg0: i7, in %arg1: i1, out r1: i21, out r2: i4, out r3: i16) {
  // CHECK: assign r1 = {3{arg0}};
  // DISALLOW: assign r1 = {arg0, arg0, arg0};
  %r1 = comb.replicate %arg0 : (i7) -> i21

  // CHECK: assign r2 = {4{arg1}};
  // DISALLOW: assign r2 = {arg1, arg1, arg1, arg1};
  %r2 = comb.replicate %arg1 : (i1) -> i4

  // CHECK: assign r3 = {2{arg0, arg1}};
  // DISALLOW: assign r3 = {arg0, arg1, arg0, arg1};
  %c = comb.concat %arg0, %arg1 : i7, i1
  %r3 = comb.replicate %c : (i8) -> i16

  hw.output %r1, %r2, %r3 : i21, i4, i16
}
