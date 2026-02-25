// RUN: circt-lec --native-sca -c1 eq_spec -c2 eq_impl %s | FileCheck %s

hw.module @eq_spec(in %a: i4, in %b: i4, out out: i4) {
  %0 = comb.xor %a, %b : i4
  %1 = comb.and %0, %a : i4
  hw.output %1 : i4
}

hw.module @eq_impl(in %a: i4, in %b: i4, out out: i4) {
  %0 = comb.xor %a, %b : i4
  %1 = comb.and %0, %a : i4
  hw.output %1 : i4
}

// CHECK: c1 == c2
