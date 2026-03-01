// RUN: circt-synth %s --timing-report-dir=- --top dut -o /dev/null | FileCheck %s

// CHECK: === Timing Report ===
// CHECK: Module: dut
// CHECK: Delay Model: nldm
// CHECK: Arrival Iterations:
// CHECK: Slew Converged:
// CHECK: Path 1: delay = 25
// CHECK: Startpoint:
// CHECK: Endpoint:

module attributes {
  synth.liberty.library = {name = "dummy", time_unit = "1ns"},
  synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
} {
  hw.module private @BUF(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.2 : f64}},
      out Y : i1 {synth.liberty.pin = {
        direction = "output",
        synth.nldm.arcs = [
          #synth.nldm_arc<"A", "Y", "positive_unate", [], [], [0.025 : f64], [], [], [], [], [], [], [], [], []>
        ]
      }}) {
    hw.output %A : i1
  }

  hw.module @dut(in %a : i1, out y : i1) {
    %i0 = hw.instance "u_buf" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
    hw.output %i0 : i1
  }
}
