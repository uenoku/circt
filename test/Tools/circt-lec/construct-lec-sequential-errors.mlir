// RUN: circt-opt --construct-lec="first-module=modA0 second-module=modB0 sequential-mode=arc-state insert-mode=none" --split-input-file --verify-diagnostics %s

builtin.module {
  hw.module @modA0(in %clk0: !seq.clock, in %clk1: !seq.clock, in %a: i8, out out: i8) {
    %s = arc.state @id_i8(%a) clock %clk0 latency 1 {names = ["s"]} : (i8) -> i8 // expected-error {{state 's' has mismatched clock inputs}}
    hw.output %s : i8
  }

  hw.module @modB0(in %clk0: !seq.clock, in %clk1: !seq.clock, in %a: i8, out out: i8) {
    %s = arc.state @id_i8(%a) clock %clk1 latency 1 {names = ["s"]} : (i8) -> i8
    hw.output %s : i8
  }

  arc.define @id_i8(%arg0: i8) -> i8 {
    arc.output %arg0 : i8
  }
}

// -----

builtin.module {
  hw.module @modA0(in %clk: !seq.clock, in %a: i8, out out: i8) {
    %s = arc.state @id_i8(%a) clock %clk latency 1 : (i8) -> i8 // expected-error {{sequential arc-state mode requires every state result to have a stable name}}
    hw.output %s : i8
  }

  hw.module @modB0(in %clk: !seq.clock, in %a: i8, out out: i8) {
    %s = arc.state @id_i8(%a) clock %clk latency 1 {names = ["s"]} : (i8) -> i8
    hw.output %s : i8
  }

  arc.define @id_i8(%arg0: i8) -> i8 {
    arc.output %arg0 : i8
  }
}

// -----

builtin.module {
  hw.module @modA0(in %clk: !seq.clock, in %addr: i2, out out: i8) {
    %mem = arc.memory <4 x i8, i2> // expected-error {{sequential arc-state mode does not yet support memories}}
    %data = arc.memory_read_port %mem[%addr] : <4 x i8, i2>
    hw.output %data : i8
  }

  hw.module @modB0(in %clk: !seq.clock, in %addr: i2, out out: i8) {
    %mem = arc.memory <4 x i8, i2>
    %data = arc.memory_read_port %mem[%addr] : <4 x i8, i2>
    hw.output %data : i8
  }
}
