// RUN: circt-opt --pass-pipeline='builtin.module(synth-seq-mem-mapper)' %s | FileCheck %s

// 1R1W synchronous SRAM, read latency 1.

// CHECK-LABEL: hw.module.extern @sram_256x8
hw.module.extern @sram_256x8(in %clk : i1, in %re : i1, in %raddr : i8, out rdout : i8, in %we : i1, in %waddr : i8, in %wdin : i8) attributes {synth.sram.macro = #synth.sram_macro<8 x 256, [#synth.sram_port<read, "clk", "re", "raddr", 8, "rdout", 8, "", 0, 1>, #synth.sram_port<write, "clk", "we", "waddr", 8, "wdin", 8, "", 0, 1>]>}

// CHECK-LABEL: hw.module @top_basic(
// CHECK: %[[DOUT:.*]] = hw.instance "mem" @sram_256x8(clk: %clk: i1, re: %{{.*}}: i1, raddr: %raddr: i8, we: %{{.*}}: i1, waddr: %waddr: i8, wdin: %wdata: i8) -> (rdout: i8)
// CHECK: hw.output %[[DOUT]] : i8
hw.module @top_basic(in %clk : i1, in %raddr : i8, in %waddr : i8, in %wdata : i8, out dout : i8) {
  %c = seq.to_clock %clk
  %mem = seq.firmem 1, 1, undefined, undefined : <256 x 8>
  %d = seq.firmem.read_port %mem[%raddr], clock %c : <256 x 8>
  seq.firmem.write_port %mem[%waddr] = %wdata, clock %c : <256 x 8>
  hw.output %d : i8
}

// Masked macro: per-bit write masks.

// CHECK-LABEL: hw.module.extern @sram_64x32_mask
hw.module.extern @sram_64x32_mask(in %clk : i1, in %re : i1, in %raddr : i6, out rdout : i32, in %we : i1, in %waddr : i6, in %wdin : i32, in %wmask : i4) attributes {synth.sram.macro = #synth.sram_macro<32 x 64, [#synth.sram_port<read, "clk", "re", "raddr", 6, "rdout", 32, "", 0, 1>, #synth.sram_port<write, "clk", "we", "waddr", 6, "wdin", 32, "wmask", 4, 1>]>}

// Masked write maps with the mask wired through; unmasked ports tie enable.
// CHECK-LABEL: hw.module @top_masked(
// CHECK: hw.instance "mem" @sram_64x32_mask({{[^)]*}}wmask: %mask: i4)
// CHECK: hw.output %{{.*}} : i32
hw.module @top_masked(in %clk : i1, in %raddr : i6, in %waddr : i6, in %wdata : i32, in %mask : i4, in %en : i1, out dout : i32) {
  %c = seq.to_clock %clk
  %mem = seq.firmem 1, 1, undefined, undefined : <64 x 32, mask 4>
  %d = seq.firmem.read_port %mem[%raddr], clock %c enable %en : <64 x 32, mask 4>
  seq.firmem.write_port %mem[%waddr] = %wdata, clock %c mask %mask : <64 x 32, mask 4>, i4
  hw.output %d : i32
}

// Unmasked write into a masked macro ties the mask to all-ones.
// CHECK-LABEL: hw.module @top_masktie(
// CHECK: %[[ONES:.*]] = hw.constant -1 : i4
// CHECK: hw.instance "mem" @sram_64x32_mask({{[^)]*}}wmask: %[[ONES]]: i4)
hw.module @top_masktie(in %clk : i1, in %raddr : i6, in %waddr : i6, in %wdata : i32, out dout : i32) {
  %c = seq.to_clock %clk
  %mem = seq.firmem 1, 1, undefined, undefined : <64 x 32, mask 4>
  %d = seq.firmem.read_port %mem[%raddr], clock %c : <64 x 32, mask 4>
  seq.firmem.write_port %mem[%waddr] = %wdata, clock %c : <64 x 32, mask 4>
  hw.output %d : i32
}

// Skips: geometry mismatch, read-write ports, init, collision behavior.
// All firmems below must survive (m0 also has read latency 1, which the
// flop fallback rejects too).
// CHECK-LABEL: hw.module @top_skips(
// CHECK: seq.firmem {{.*}} : <128 x 8>
// CHECK: seq.firmem {{.*}} : <64 x 32, mask 4>
// CHECK: seq.firmem {{.*}} : <64 x 32, mask 4>
// CHECK: seq.firmem {{.*}} : <64 x 32, mask 4>
hw.module @top_skips(in %clk : i1, in %addr : i8, in %addr6 : i6, in %addr7 : i7, in %data : i32, in %data8 : i8, in %en : i1, in %mode : i1, out o0 : i8, out o1 : i32, out o2 : i32, out o3 : i32) {
  %c = seq.to_clock %clk
  // Wrong depth.
  %m0 = seq.firmem 1, 1, undefined, undefined : <128 x 8>
  %r0 = seq.firmem.read_port %m0[%addr7], clock %c : <128 x 8>
  // Read-write port.
  %m1 = seq.firmem 1, 1, undefined, undefined : <64 x 32, mask 4>
  %r1 = seq.firmem.read_write_port %m1[%addr6] = %data if %mode, clock %c : <64 x 32, mask 4>
  // Power-on init.
  %m2 = seq.firmem 1, 1, undefined, undefined {init = #seq.firmem.init<"mem.txt", false, false>} : <64 x 32, mask 4>
  %r2 = seq.firmem.read_port %m2[%addr6], clock %c : <64 x 32, mask 4>
  // Defined collision behavior.
  %m3 = seq.firmem 1, 1, old, undefined : <64 x 32, mask 4>
  %r3 = seq.firmem.read_port %m3[%addr6], clock %c : <64 x 32, mask 4>
  hw.output %r0, %r1, %r2, %r3 : i8, i32, i32, i32
}

// No macro matches: unmasked asynchronous-read memory explodes to flops
// (standard memory-mapper default).
// CHECK-LABEL: hw.module @top_explode(
// CHECK-NOT: seq.firmem
// CHECK-COUNT-16: seq.firreg {{.*}} : i1
// CHECK: hw.output
// CHECK-NOT: seq.firmem
hw.module @top_explode(in %clk : i1, in %raddr : i2, in %waddr : i2, in %wdata : i8, in %wen : i1, out dout : i8) {
  %c = seq.to_clock %clk
  %mem = seq.firmem 0, 1, undefined, undefined : <4 x 8>
  %d = seq.firmem.read_port %mem[%raddr], clock %c : <4 x 8>
  seq.firmem.write_port %mem[%waddr] = %wdata, clock %c enable %wen : <4 x 8>
  hw.output %d : i8
}

// Masked writes explode with per-lane write enables: bit 0 follows mask
// bit 0, bit 4 follows mask bit 1 (lanes of 4).
// CHECK-LABEL: hw.module @top_explode_masked(
// CHECK-NOT: seq.firmem
// CHECK-COUNT-32: seq.firreg {{.*}} : i1
// CHECK: comb.extract {{.*}} from 0{{.*}} -> i1
// CHECK: comb.extract {{.*}} from 1{{.*}} -> i1
hw.module @top_explode_masked(in %clk : i1, in %raddr : i2, in %waddr : i2, in %wdata : i8, in %wen : i1, in %mask : i2, out dout : i8) {
  %c = seq.to_clock %clk
  %mem = seq.firmem 0, 1, undefined, undefined : <4 x 8, mask 2>
  %d = seq.firmem.read_port %mem[%raddr], clock %c : <4 x 8, mask 2>
  seq.firmem.write_port %mem[%waddr] = %wdata, clock %c enable %wen mask %mask : <4 x 8, mask 2>, i2
  hw.output %d : i8
}
