// RUN: circt-synth %s -analysis-output=- -top counter --enable-sop-balancing | FileCheck %s --check-prefixes COMMON,AIG
// RUN: circt-synth %s -analysis-output=- -top counter -lower-to-k-lut 6 | FileCheck %s --check-prefixes COMMON,LUT6
// RUN: circt-synth %s -analysis-output=- -top mul_const_mix --enable-sop-balancing | FileCheck %s --check-prefixes MIX,AIG-MIX
// RUN: circt-synth %s -analysis-output=- -top mul_const_mix -lower-to-k-lut 6 | FileCheck %s --check-prefixes MIX,LUT6-MIX
// RUN: circt-synth %s -analysis-output=- -top test -analysis-output-format=json | FileCheck %s --check-prefix JSON

// COMMON-LABEL: # Longest Path Analysis result for "counter"
// COMMON-NEXT: Found 168 paths
// COMMON-NEXT: Found 32 unique end points
// AIG-NEXT: Maximum path delay: 27
// LUT6-NEXT: Maximum path delay: 6
// MIX-LABEL: # Longest Path Analysis result for "mul_const_mix"
// MIX-NEXT: Found 101 paths
// MIX-NEXT: Found 13 unique end points
// AIG-MIX-NEXT: Maximum path delay: 22
// LUT6-MIX-NEXT: Maximum path delay: 5
// AIG-MIX: synth.aig.and_inv: {{ *}}675
// Don't test detailed reports as they are not stable.

hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %add, %clk : i16
    %add = comb.mul %reg, %a : i16
    hw.output %reg : i16
}

hw.module @mul_const_mix(in %a: i4, in %b: i5, in %c: i4, in %d: i5, out result: i16) {
    %c15_i4 = hw.constant 15 : i4
    %c7_i3 = hw.constant 7 : i3
    %c0_i3 = hw.constant 0 : i3
    %lhs0 = comb.concat %a, %c15_i4 : i4, i4
    %lhs1 = comb.concat %lhs0, %c : i8, i4
    %lhs = comb.concat %lhs1, %c15_i4 : i12, i4
    %rhs0 = comb.concat %b, %c7_i3 : i5, i3
    %rhs1 = comb.concat %rhs0, %d : i8, i5
    %rhs = comb.concat %rhs1, %c0_i3 : i13, i3
    %mul = comb.mul %lhs, %rhs : i16
    hw.output %mul : i16
}

// Make sure json is emitted.
// JSON: {"module_name":"test","timing_levels":[
// COMMON-NOT: "test"
hw.module @test(in %a: i16, out result: i16) {
    hw.output %a : i16
}
