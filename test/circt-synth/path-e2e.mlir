// RUN: circt-synth %s -analysis-output=- -top counter --enable-sop-balancing | FileCheck %s --check-prefixes COMMON,AIG
// RUN: circt-synth %s -analysis-output=- -top counter --target-ir mig | FileCheck %s --check-prefixes COMMON,MIG
// RUN: circt-synth %s -analysis-output=- -top counter -lower-to-k-lut 6 | FileCheck %s --check-prefixes COMMON,LUT6
// RUN: circt-synth %s -analysis-output=- -top test -analysis-output-format=json | FileCheck %s --check-prefix JSON

// COMMON-LABEL: # Longest Path Analysis result for "counter"
// AIG: Maximum path delay: 27
// MIG: Maximum path delay: 30
// LUT6: Maximum path delay: 6
// Don't test detailed reports as they are not stable.


hw.module @create_reg(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %a, %clk : i16
    hw.output %reg : i16
}

hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = hw.instance "reg" @create_reg(a: %a: i16, clk: %clk: !seq.clock) -> (result: i16)
    %add = comb.mul %reg, %a : i16
    hw.output %add : i16
}

// Make sure json is emitted.
// JSON: {"module_name":"test","timing_levels":[
// COMMON-NOT: "test"
hw.module @test(in %a: i16, out result: i16) {
    hw.output %a : i16
}
