// RUN: circt-sta %s --timing-report-dir=- --top dut -o /dev/null | FileCheck %s

// Inverter with negative_unate timing sense and different rise/fall delays.
// cell_rise = 0.020 ns (20 ps), cell_fall = 0.030 ns (30 ps)
// Through INV -> INV chain:
//   Rise input -> Fall output (cell_fall=30ps) -> Rise output (cell_rise=20ps) = 50ps
//   Fall input -> Rise output (cell_rise=20ps) -> Fall output (cell_fall=30ps) = 50ps
// Both paths give 50ps total, so delay = 50.

// CHECK: === Timing Report ===
// CHECK: Module: dut
// CHECK: Delay Model: nldm
// CHECK: Path 1: delay = 50

module attributes {
  synth.liberty.library = {name = "dummy", time_unit = "1ns"},
  synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
} {
  hw.module private @INV(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.1 : f64}},
      out Y : i1 {synth.liberty.pin = {
        direction = "output",
        synth.nldm.arcs = [
          #synth.nldm_arc<"A", "Y", "negative_unate",
            [], [], [0.020 : f64],
            [], [], [0.030 : f64],
            [], [], [],
            [], [], []>
        ]
      }}) {
    %0 = comb.xor %A, %A : i1
    hw.output %0 : i1
  }

  hw.module @dut(in %a : i1, out y : i1) {
    %i0 = hw.instance "u0" @INV(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "INV"}
    %i1 = hw.instance "u1" @INV(A: %i0: i1) -> (Y: i1) {synth.liberty.cell = "INV"}
    hw.output %i1 : i1
  }
}
