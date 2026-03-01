// RUN: circt-synth %s --top dut -o - | circt-sta --timing-report-dir=- --top dut -o /dev/null | FileCheck %s

// End-to-end test: Liberty-annotated cells → synth-annotate-techlib → tech-mapper → STA.
// The annotate-techlib pass derives hw.techlib.info from synth.liberty.area and
// synth.nldm.arcs, enabling TechMapper to use Liberty-imported cells.

// CHECK: === Timing Report ===
// CHECK: Module: dut
// CHECK: Delay Model: nldm
// CHECK: Path 1: delay = 45
// CHECK: Startpoint: a[0] (input port)
// CHECK: Endpoint:   y[0] (output port)
// CHECK: Path 2: delay = 43
// CHECK: Startpoint: b[0] (input port)
// CHECK: Endpoint:   y[0] (output port)

module attributes {
  synth.liberty.library = {
    name = "testlib",
    time_unit = "1ns"
  },
  synth.nldm.default_input_slew = 0.1 : f64,
  synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
} {
  // AND-inverter cell: Y = ~(A & B)
  // area = 2.0, worst-case cell_rise = 0.030, cell_fall = 0.025
  // With time_unit = 1000 ps/unit: delay = 30 ps per input
  hw.module private @NAND2(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.1 : f64}},
      in %B : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.1 : f64}},
      out Y : i1 {synth.liberty.pin = {
        direction = "output",
        synth.nldm.arcs = [
          #synth.nldm_arc<"A", "Y", "negative_unate",
            [1.000000e-02, 3.000000e-02], [0.000000e+00, 2.000000e-01],
            [1.200000e-02, 1.800000e-02, 2.000000e-02, 3.000000e-02],
            [], [], [2.500000e-02],
            [], [], [], [], [], []>,
          #synth.nldm_arc<"B", "Y", "negative_unate",
            [1.000000e-02, 3.000000e-02], [0.000000e+00, 2.000000e-01],
            [1.000000e-02, 1.500000e-02, 1.800000e-02, 2.800000e-02],
            [], [], [2.000000e-02],
            [], [], [], [], [], []>
        ]
      }}) attributes {synth.liberty.area = 2.000000e+00 : f64} {
    %0 = comb.and %A, %B : i1
    %1 = comb.xor %0, %true : i1
    %true = hw.constant true
    hw.output %1 : i1
  }

  // Buffer cell: Y = A
  // area = 1.0, worst-case delay = 0.020
  hw.module private @BUF(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.05 : f64}},
      out Y : i1 {synth.liberty.pin = {
        direction = "output",
        synth.nldm.arcs = [
          #synth.nldm_arc<"A", "Y", "positive_unate",
            [], [], [2.000000e-02],
            [], [], [1.500000e-02],
            [], [], [], [], [], []>
        ]
      }}) attributes {synth.liberty.area = 1.000000e+00 : f64} {
    hw.output %A : i1
  }

  // Design under test: simple NAND tree
  hw.module @dut(in %a : i1, in %b : i1, out y : i1) {
    %nand = hw.instance "u0" @NAND2(A: %a: i1, B: %b: i1) -> (Y: i1) {synth.liberty.cell = "NAND2"}
    %buf = hw.instance "u1" @BUF(A: %nand: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
    hw.output %buf : i1
  }
}
