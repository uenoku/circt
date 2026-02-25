// RUN: circt-opt --verify-native-sca="first-module=neq_spec second-module=neq_impl" --verify-diagnostics %s

hw.module @neq_spec(in %a: i4, in %b: i4, out out: i4) {
  %0 = comb.and %a, %b : i4
  hw.output %0 : i4
}

// expected-error @below {{native SCA check failed: non-zero residual}}
hw.module @neq_impl(in %a: i4, in %b: i4, out out: i4) {
  %0 = comb.or %a, %b : i4
  hw.output %0 : i4
}
