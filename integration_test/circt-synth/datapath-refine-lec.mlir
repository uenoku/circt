// REQUIRES: z3

// Equivalence of probe-guided selective refinement: each selectively
// refined variant must equal the source function. Probes are stripped on
// both sides first (they are transparent pass-throughs; circt-lec has no
// lowering for them).
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-strip-timing-probes)' -o %t.ref.mlir

// Selective reduce-delay (only the probe-fed chain folds) equals source.
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(datapath-reduce-delay{pressure-margin-levels=0}), synth-strip-timing-probes)' -o %t.sel.mlir
// RUN: circt-lec.sh %t.sel.mlir %t.ref.mlir -c1=two_chains -c2=two_chains
hw.module @two_chains(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %e : i4, in %f : i4, out x : i4, out y : i4) {
  %t0 = comb.add %a, %b : i4
  %s0 = comb.add %t0, %c : i4
  %p = synth.timing_probe %s0 : i4 {required_time = 10000 : i64}
  %t1 = comb.add %d, %e : i4
  %s1 = comb.add %t1, %f : i4
  hw.output %p, %s1 : i4, i4
}

// Per-operation Booth (recorded policy on one multiplier region only)
// equals source.
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-policy=booth}, hw.module(convert-datapath-to-comb), synth-strip-timing-probes)' -o %t.mix.mlir
// RUN: circt-lec.sh %t.mix.mlir %t.ref.mlir -c1=two_muls -c2=two_muls
hw.module @two_muls(in %a : i8, in %b : i8, in %c : i8, in %d : i8, out x : i8, out y : i8) {
  %m0 = comb.mul %a, %b : i8
  %p = synth.timing_probe %m0 : i8 {required_time = 10000 : i64}
  %m1 = comb.mul %c, %d : i8
  hw.output %p, %m1 : i8, i8
}

// Per-operation compress-to-add (recorded policy on one compressor region
// only) equals source.
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-policy=compress-to-add}, hw.module(convert-datapath-to-comb), synth-strip-timing-probes)' -o %t.cta.mlir
// RUN: circt-lec.sh %t.cta.mlir %t.ref.mlir -c1=two_compress -c2=two_compress
hw.module @two_compress(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %e : i4, in %f : i4, out x : i4, out y : i4) {
  %add0 = comb.add %a, %b, %c : i4
  %p = synth.timing_probe %add0 : i4 {required_time = 10000 : i64}
  %add1 = comb.add %d, %e, %f : i4
  hw.output %p, %add1 : i4, i4
}
