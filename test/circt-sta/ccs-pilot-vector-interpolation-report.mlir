// RUN: circt-sta %s --timing-report-dir=- --top dut -o /dev/null | FileCheck %s
// RUN: circt-sta %s --timing-report-dir=- --top dut --show-waveform-details -o /dev/null | FileCheck %s --check-prefix=WAVE

// CHECK: === Timing Report ===
// CHECK: Delay Model: ccs-pilot
// CHECK: Path 1: delay = 350

// WAVE: Waveform Details:
// WAVE: t50=700
// WAVE: slew10-90=400

module attributes {
  synth.timing.model = "ccs-pilot",
  synth.ccs.pilot.waveform_delay = true,
  synth.liberty.library = {
    name = "dummy",
    time_unit = "1ns",
    default_input_transition = "0.5"
  },
  synth.nldm.default_input_slew = 0.5 : f64,
  synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
} {
  hw.module private @BUF(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
      out Y : i1 {synth.liberty.pin = {
        direction = "output",
        synth.nldm.arcs = [
          #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64], [0.0 : f64], [0.025 : f64], [], [], [], [], [], [], [], [], []>
        ],
        synth.ccs.pilot.arcs = [
          #synth.ccs_pilot_arc<"A", "Y", [], [], [], [], [0.0 : f64, 1.0 : f64], [0.5 : f64, 0.5 : f64], [0.0 : f64, 0.2 : f64], [[0.0 : f64, 0.2 : f64], [0.0 : f64, 0.8 : f64]], [[0.0 : f64, 1.0 : f64], [0.0 : f64, 1.0 : f64]], [], [], [], [], []>
        ]
      }}) {
    hw.output %A : i1
  }

  hw.module @dut(in %a : i1, out y : i1) {
    %i0 = hw.instance "u0" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
    hw.output %i0 : i1
  }
}
