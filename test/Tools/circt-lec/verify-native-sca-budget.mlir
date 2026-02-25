// RUN: circt-opt --verify-native-sca="first-module=eq_spec second-module=eq_impl max-steps=0" --verify-diagnostics %s

hw.module @eq_spec(in %a: i4, in %b: i4, out out: i4) {
  %0 = comb.xor %a, %b : i4
  %1 = comb.and %0, %a : i4
  hw.output %1 : i4
}

// expected-error @below {{inconclusive: native SCA rewrite budget exceeded}}
hw.module @eq_impl(in %a: i4, in %b: i4, out out: i4) {
  %0 = comb.xor %a, %b : i4
  %1 = comb.and %0, %a : i4
  hw.output %1 : i4
}
