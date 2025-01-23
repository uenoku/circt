// RUN: circt-opt --split-input-file --pass-pipeline='builtin.module(hw.design(export-verilog))'  %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt --split-input-file --pass-pipeline='builtin.module(hw.design(test-apply-lowering-options{options=""}, export-verilog))'  %s | FileCheck %s --check-prefix=CLEAR
// RUN: circt-opt --split-input-file --pass-pipeline='builtin.module(hw.design(test-apply-lowering-options{options="noAlwaysComb"}, export-verilog))'  %s | FileCheck %s --check-prefix=NOALWAYSCOMB

hw.design {
hw.module @test() {
  sv.alwayscomb {
  }
}
}

// DEFAULT: always_comb begin
// DEFAULT: end // always_comb

// CLEAR: always_comb begin
// CLEAR: end // always_comb

// NOALWAYSCOMB: always @(*) begin
// NOALWAYSCOMB: end // always @(*)

// -----

hw.design attributes {circt.loweringOptions = "noAlwaysComb"} {
hw.module @test() {
  sv.alwayscomb {
  }
}
}

// DEFAULT: always @(*) begin
// DEFAULT: end // always @(*)

// CLEAR: always_comb begin
// CLEAR: end // always_comb

// NOALWAYSCOMB: always @(*) begin
// NOALWAYSCOMB: end // always @(*)
