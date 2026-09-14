// RUN: circt-opt --pass-pipeline='builtin.module(synth-seq-tech-mapper)' %s | FileCheck %s

// Library cells (synth.ff bodies, as produced by the Liberty importer).

// CHECK-LABEL: hw.module @dff(
hw.module @dff(in %CLK : i1, in %D : i1, out Q : i1) attributes {synth.liberty.cell = {area = 20.0 : f64}} {
  %q, %qn = synth.ff %D clock %CLK edge rise : i1, i1
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @dffR(
hw.module @dffR(in %CLK : i1, in %D : i1, in %RESET_B : i1, out Q : i1) attributes {synth.liberty.cell = {area = 30.0 : f64}} {
  %true = hw.constant true
  %rst = comb.xor %RESET_B, %true : i1
  %q, %qn = synth.ff %D clock %CLK edge rise clear %rst : i1, i1
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @dffE(
hw.module @dffE(in %CLK : i1, in %D : i1, in %DE : i1, out Q : i1) attributes {synth.liberty.cell = {area = 40.0 : f64}} {
  %q, %qn = synth.ff %D clock %CLK edge rise enable %DE : i1, i1
  hw.output %q : i1
}

// Scan flop: picked only when nothing cheaper fits; scan tied off.
// CHECK-LABEL: hw.module @sdf(
hw.module @sdf(in %CLK : i1, in %D : i1, in %SCD : i1, in %SCE : i1, out Q : i1) attributes {synth.liberty.cell = {area = 10.0 : f64}} {
  %next = comb.mux %SCE, %SCD, %D : i1
  %q, %qn = synth.ff %next clock %CLK edge rise : i1, i1
  hw.output %q : i1
}

// Falling-edge flop: never matches seq's positive-edge flops, but must not
// break the pass.
// CHECK-LABEL: hw.module @dffN(
hw.module @dffN(in %CLK : i1, in %D : i1, out Q : i1) {
  %q, %qn = synth.ff %D clock %CLK edge fall : i1, i1
  hw.output %q : i1
}

// Plain flop gets the cheapest compatible cell: the scan flop with its scan
// inputs tied off (area 10 < dff area 20).
// CHECK-LABEL: hw.module @top_plain(
// CHECK: %[[Q:.*]] = hw.instance "r" @sdf(CLK: %clk: i1, D: %d: i1, SCD: %[[C0:.*]]: i1, SCE: %[[C1:.*]]: i1) -> (Q: i1)
// CHECK: hw.output %[[Q]] : i1
hw.module @top_plain(in %clk : i1, in %d : i1, out q : i1) {
  %c = seq.to_clock %clk
  %r = seq.firreg %d clock %c : i1
  hw.output %r : i1
}

// Async reset to zero matches the clear cell; active-high design reset is
// inverted for the active-low port.
// CHECK-LABEL: hw.module @top_async(
// CHECK: %[[RST:.*]] = comb.xor %rst, %{{.*}} : i1
// CHECK: hw.instance "r" @dffR(CLK: %clk: i1, D: %d: i1, RESET_B: %[[RST]]: i1) -> (Q: i1)
hw.module @top_async(in %clk : i1, in %d : i1, in %rst : i1, out q : i1) {
  %c = seq.to_clock %clk
  %false = hw.constant false
  %r = seq.firreg %d clock %c reset async %rst, %false : i1
  hw.output %r : i1
}

// Synchronous reset is emulated with a mux into a reset-less cell.
// CHECK-LABEL: hw.module @top_sync(
// CHECK: %[[MUX:.*]] = comb.mux %rst, %{{.*}}, %d : i1
// CHECK: hw.instance "r" @sdf(CLK: %clk: i1, D: %[[MUX]]: i1, SCD: %{{.*}}: i1, SCE: %{{.*}}: i1) -> (Q: i1)
hw.module @top_sync(in %clk : i1, in %d : i1, in %rst : i1, out q : i1) {
  %c = seq.to_clock %clk
  %false = hw.constant false
  %r = seq.firreg %d clock %c reset sync %rst, %false : i1
  hw.output %r : i1
}

// Clock enable maps to the enable cell directly.
// CHECK-LABEL: hw.module @top_ce(
// CHECK: hw.instance "r" @dffE(CLK: %clk: i1, D: %d: i1, DE: %en: i1) -> (Q: i1)
hw.module @top_ce(in %clk : i1, in %d : i1, in %en : i1, out q : i1) {
  %c = seq.to_clock %clk
  %r = seq.compreg.ce %d, %c, %en : i1
  hw.output %r : i1
}

// Clock enable without an enable cell in the library would use feedback;
// here dffE exists so the direct mapping wins. Async reset to one with no
// preset cell stays unmapped (async cannot be emulated).
// CHECK-LABEL: hw.module @top_async1(
// CHECK: seq.firreg {{.*}} reset async
hw.module @top_async1(in %clk : i1, in %d : i1, in %rst : i1, out q : i1) {
  %c = seq.to_clock %clk
  %true = hw.constant true
  %r = seq.firreg %d clock %c reset async %rst, %true : i1
  hw.output %r : i1
}

// Wide flops explode bit by bit (like Yosys techmap), each bit mapping
// to the cheapest cell.
// CHECK-LABEL: hw.module @top_wide(
// CHECK-COUNT-8: hw.instance {{.*}} @sdf(
// CHECK-NOT: seq.firreg
hw.module @top_wide(in %clk : i1, in %d : i8, out q : i8) {
  %c = seq.to_clock %clk
  %r = seq.firreg %d clock %c : i8
  hw.output %r : i8
}

// Array flops explode through lossless bitcasts, bit by bit (the
// surrounding bitcasts fold away in later canonicalization).
// CHECK-LABEL: hw.module @top_array(
// CHECK-COUNT-32: hw.instance {{.*}} @sdf(
// CHECK-NOT: seq.firreg
hw.module @top_array(in %clk : i1, in %d : i32, out q : i32) {
  %c = seq.to_clock %clk
  %a = hw.bitcast %d : (i32) -> !hw.array<4xi8>
  %r = seq.firreg %a clock %c : !hw.array<4xi8>
  %o = hw.bitcast %r : (!hw.array<4xi8>) -> i32
  hw.output %o : i32
}
