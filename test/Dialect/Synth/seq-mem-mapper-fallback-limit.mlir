// RUN: circt-opt --pass-pipeline='builtin.module(synth-seq-mem-mapper{max-fallback-bits=1024})' --verify-diagnostics %s
// RUN: circt-opt --pass-pipeline='builtin.module(synth-seq-mem-mapper)' --verify-diagnostics %s

// A 1024x128 (131072-bit) memory with no macro match exceeds both the
// lowered and the default max-fallback-bits: hard error, no silent flop
// explosion. The error fires before any flops are built, so this stays fast.
hw.module @top_over_limit(in %clk : i1, in %raddr : i10, in %waddr : i10, in %wdata : i128, in %wen : i1, out dout : i128) {
  %c = seq.to_clock %clk
  %mem = seq.firmem 0, 1, undefined, undefined : <1024 x 128> // expected-error {{has no matching SRAM macro and exceeds max-fallback-bits}}
  %d = seq.firmem.read_port %mem[%raddr], clock %c : <1024 x 128>
  seq.firmem.write_port %mem[%waddr] = %wdata, clock %c enable %wen : <1024 x 128>
  hw.output %d : i128
}
