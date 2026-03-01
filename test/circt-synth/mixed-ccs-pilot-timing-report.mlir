// RUN: circt-synth %s --top dut -o - | circt-sta --timing-report-dir=- --top dut -o /dev/null | FileCheck %s
// RUN: circt-sta %s --timing-report-dir=- --top dut -o /dev/null | FileCheck %s --check-prefix=STA

// CHECK: Delay Model: mixed-ccs-pilot
// CHECK: Path 1: delay = 225

// STA: Delay Model: mixed-ccs-pilot
// STA: Path 1: delay = 225

module attributes {
  synth.timing.model = "mixed-ccs-pilot",
  synth.ccs.pilot.waveform_delay = true,
  synth.ccs.pilot.cells = ["BUF_CCS"],
  synth.liberty.library = {
    name = "dummy",
    time_unit = "1ns",
    default_input_transition = "0.5"
  },
  synth.nldm.default_input_slew = 0.5 : f64,
  synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
} {
  hw.module private @BUF_CCS(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
      out Y : i1 {synth.liberty.pin = {
        direction = "output",
        synth.nldm.arcs = [
          #synth.nldm_arc<"A", "Y", "positive_unate", [], [], [0.025 : f64], [], [], [], [], [], [], [], [], []>
        ],
        synth.ccs.pilot.arcs = [
          #synth.ccs_pilot_arc<"A", "Y", [0.0 : f64, 0.4 : f64], [0.0 : f64, 1.0 : f64], [0.0 : f64, 0.4 : f64], [1.0 : f64, 0.0 : f64]>
        ]
      }}) {
    hw.output %A : i1
  }

  hw.module private @BUF_NLDM(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
      out Y : i1 {synth.liberty.pin = {
        direction = "output",
        synth.nldm.arcs = [
          #synth.nldm_arc<"A", "Y", "positive_unate", [], [], [0.025 : f64], [], [], [], [], [], [], [], [], []>
        ]
      }}) {
    hw.output %A : i1
  }

  hw.module @dut(in %a : i1, out y : i1) {
    %i0 = hw.instance "u0" @BUF_CCS(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF_CCS"}
    %i1 = hw.instance "u1" @BUF_NLDM(A: %i0: i1) -> (Y: i1) {synth.liberty.cell = "BUF_NLDM"}
    hw.output %i1 : i1
  }
}
