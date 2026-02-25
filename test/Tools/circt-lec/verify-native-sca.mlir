// RUN: circt-opt --verify-native-sca="first-module=eq_spec second-module=eq_impl" %s | FileCheck %s

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

hw.module @neq_spec(in %a: i4, in %b: i4, out out: i4) {
  %0 = comb.and %a, %b : i4
  hw.output %0 : i4
}

hw.module @neq_impl(in %a: i4, in %b: i4, out out: i4) {
  %0 = comb.or %a, %b : i4
  hw.output %0 : i4
}

// CHECK: c1 == c2
