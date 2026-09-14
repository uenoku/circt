// RUN: circt-opt --pass-pipeline='builtin.module(synth-seq-mem-mapper{max-fallback-bits=1024})' %s | FileCheck %s

// A 32x32 (1024-bit) memory sits exactly at max-fallback-bits and still
// explodes to flops (the limit only rejects strictly larger memories).
// CHECK-LABEL: hw.module @top_at_limit(
// CHECK-NOT: seq.firmem
// CHECK-COUNT-1024: seq.firreg {{.*}} : i1
hw.module @top_at_limit(in %clk : i1, in %raddr : i5, in %waddr : i5, in %wdata : i32, in %wen : i1, out dout : i32) {
  %c = seq.to_clock %clk
  %mem = seq.firmem 0, 1, undefined, undefined : <32 x 32>
  %d = seq.firmem.read_port %mem[%raddr], clock %c : <32 x 32>
  seq.firmem.write_port %mem[%waddr] = %wdata, clock %c enable %wen : <32 x 32>
  hw.output %d : i32
}
