// RUN: mkdir -p %t/bin && cp %S/Inputs/refine-fake-sta-violate.sh %t/bin/sta && chmod +x %t/bin/sta && env PATH=%t/bin:$PATH circt-synth --liberty %S/Inputs/refine.lib --strip-lib-to-extern --top top --probe-clocks=clk --datapath-retry=4 --datapath-refinement --retry-timeout=120 %s -o %t/out.mlir 2>&1 | FileCheck %s
// RUN: circt-synth --liberty %S/Inputs/refine.lib --strip-lib-to-extern --top top --probe-clocks=clk --datapath-refinement --datapath-refinement-policy=baseline --datapath-pressure-margin-levels=-1 %s -o %t/base.mlir
// RUN: diff %t/out.mlir %t/base.mlir

// No-improvement refinement: every candidate violates like the baseline,
// so the baseline wins all ties and is emitted exactly (byte-identical to
// the equivalent single run).
// CHECK: refinement results
// CHECK: [BEST] baseline
hw.module @sub(in %clk : i1, in %a : i4, in %b : i4, in %c : i4, out x : i4, out y : i4) {
  %ck = seq.to_clock %clk
  %m = comb.mul %a, %b : i4
  %r0 = seq.firreg %m clock %ck : i4
  %s = comb.add %r0, %r0, %c : i4
  %r1 = seq.firreg %s clock %ck : i4
  hw.output %r0, %r1 : i4, i4
}
hw.module @top(in %clk : i1, in %a : i4, in %b : i4, in %c : i4, out x : i4, out y : i4) {
  %x, %y = hw.instance "u_sub" @sub(clk: %clk: i1, a: %a: i4, b: %b: i4, c: %c: i4) -> (x: i4, y: i4)
  hw.output %x, %y : i4, i4
}
