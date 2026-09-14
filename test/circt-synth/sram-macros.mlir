// RUN: circt-synth --liberty %S/Inputs/sample.lib --sram-macros %S/Inputs/sram-macros.mlir --top small_mem %s | FileCheck %s

// A 1R1W memory with a matching SRAM macro maps to a macro instance
// instead of exploding to flops.

// CHECK-LABEL: hw.module @small_mem(
// CHECK: hw.instance {{.*}} @sram_16x8_1r1w(
// CHECK-NOT: seq.firmem
hw.module @small_mem(in %addr : i4, in %en : i1, in %clk : i1, in %wdata : i8, out rdata : i8) {
  %c = seq.to_clock %clk
  %mem = seq.firmem 0, 1, undefined, undefined : <16 x 8>
  %d = seq.firmem.read_port %mem[%addr], clock %c : <16 x 8>
  seq.firmem.write_port %mem[%addr] = %wdata, clock %c enable %en : <16 x 8>
  hw.output %d : i8
}
