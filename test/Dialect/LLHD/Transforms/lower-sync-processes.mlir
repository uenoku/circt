// RUN: circt-opt --llhd-wrap-procedural-ops --llhd-inline-calls --llhd-mem2reg --llhd-hoist-signals --llhd-lower-sync-processes %s | FileCheck %s

// Lowering synchronous processes with conditional drives (the shape Deseq
// leaves behind: single observed clock, rising-edge resume gate, one
// conditional drive per state signal) to plain seq.firreg with next-state
// muxes -- the same shape Yosys builds, since Liberty flops have no
// sync-reset/enable pins.

// CHECK-LABEL: hw.module @enableFlop
// CHECK-NOT: llhd.process
// CHECK: seq.firreg {{%.+}} clock {{%.+}} : i8
module {
hw.module @enableFlop(in %clock : i1, in %en : i1, in %d : i8, out q : i8) {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %1 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %c0_i8 = hw.constant 0 : i8
    %false = hw.constant false
    %clock_0 = llhd.sig name "clock" %false : i1
    %2 = llhd.prb %clock_0 : i1
    %en_1 = llhd.sig name "en" %false : i1
    %d_2 = llhd.sig name "d" %c0_i8 : i8
    %q = llhd.sig %c0_i8 : i8
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
      %4 = llhd.prb %clock_0 : i1
      llhd.wait (%2 : i1), ^bb2
    ^bb2:  // pred: ^bb1
      %5 = llhd.prb %clock_0 : i1
      %6 = comb.xor bin %4, %true : i1
      %7 = comb.and bin %6, %5 : i1
      cf.cond_br %7, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %8 = llhd.prb %en_1 : i1
      cf.cond_br %8, ^bb4, ^bb1
    ^bb4:  // pred: ^bb3
      %9 = llhd.prb %d_2 : i8
      llhd.drv %q, %9 after %1 : i8
      cf.br ^bb1
    }
    llhd.drv %clock_0, %clock after %0 : i1
    llhd.drv %en_1, %en after %0 : i1
    llhd.drv %d_2, %d after %0 : i8
    %3 = llhd.prb %q : i8
    hw.output %3 : i8
  }
hw.module @syncReset(in %clock : i1, in %rst : i1, in %d : i8, out q : i8, out r : i8) {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %1 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %c0_i8 = hw.constant 0 : i8
    %false = hw.constant false
    %clock_0 = llhd.sig name "clock" %false : i1
    %2 = llhd.prb %clock_0 : i1
    %rst_1 = llhd.sig name "rst" %false : i1
    %d_2 = llhd.sig name "d" %c0_i8 : i8
    %q = llhd.sig %c0_i8 : i8
    %r = llhd.sig %c0_i8 : i8
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %5 = llhd.prb %clock_0 : i1
      llhd.wait (%2 : i1), ^bb2
    ^bb2:  // pred: ^bb1
      %6 = llhd.prb %clock_0 : i1
      %7 = comb.xor bin %5, %true : i1
      %8 = comb.and bin %7, %6 : i1
      cf.cond_br %8, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %9 = llhd.prb %d_2 : i8
      llhd.drv %q, %9 after %1 : i8
      %10 = llhd.prb %rst_1 : i1
      cf.cond_br %10, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %r, %c0_i8 after %1 : i8
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %11 = llhd.prb %q : i8
      llhd.drv %r, %11 after %1 : i8
      cf.br ^bb1
    }
    llhd.drv %clock_0, %clock after %0 : i1
    llhd.drv %rst_1, %rst after %0 : i1
    llhd.drv %d_2, %d after %0 : i8
    %3 = llhd.prb %q : i8
    %4 = llhd.prb %r : i8
    hw.output %3, %4 : i8, i8
  }
}

// CHECK-LABEL: hw.module @syncReset
// CHECK-NOT: llhd.process
// CHECK-COUNT-2: seq.firreg
