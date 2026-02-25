// RUN: circt-opt --verify-native-sca="first-module=dp_neq_spec second-module=dp_neq_impl" --verify-diagnostics %s

hw.module @dp_neq_spec(in %a: i4, in %b: i4, in %c: i4, out out: i4) {
  %0:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  %1 = comb.add %0#0, %0#1 : i4
  hw.output %1 : i4
}

// expected-error @below {{native SCA check failed: non-zero residual}}
hw.module @dp_neq_impl(in %a: i4, in %b: i4, in %c: i4, out out: i4) {
  %z = hw.constant 0 : i4
  %0:2 = datapath.compress %a, %b, %z : i4 [3 -> 2]
  %1 = comb.add %0#0, %0#1 : i4
  hw.output %1 : i4
}
