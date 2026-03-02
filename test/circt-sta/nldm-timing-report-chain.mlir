// RUN: circt-synth %s --top dut -o - | circt-sta --timing-report-dir=- --top dut -o /dev/null | FileCheck %s

// CHECK: === Timing Report ===
// CHECK: Module: dut
// CHECK: Delay Model: nldm
// CHECK: Arrival Iterations:
// CHECK: Slew Converged:
// CHECK: Path 1: delay = 50
// CHECK:   Startpoint:
// CHECK:   Endpoint:
// CHECK:   Path:
// CHECK:     Point{{.*}}ArcDelay{{.*}}Arrival{{.*}}Slew{{.*}}Location
// CHECK:     a[0]{{.*}}-{{.*}}0{{.*}}loc(
// CHECK:     u0/Y[0]{{.*}}25{{.*}}25{{.*}}loc(
// CHECK:     u1/Y[0]{{.*}}25{{.*}}50{{.*}}loc(
// CHECK:     y[0]{{.*}}50{{.*}}loc(

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
    %i0 = hw.instance "u0" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
    %i1 = hw.instance "u1" @BUF(A: %i0: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
    hw.output %i1 : i1
  }
}
