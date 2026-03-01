// RUN: circt-synth %s --top dut -o - | circt-sta --timing-report-dir=- --top dut -o /dev/null | FileCheck %s
// RUN: circt-synth %s --top dut -o - | circt-sta --timing-report-dir=- --top dut --show-convergence-table -o /dev/null | FileCheck %s --check-prefix=TABLE
// RUN: circt-sta %s --timing-report-dir=- --top dut -o /dev/null | FileCheck %s --check-prefix=STA
// RUN: circt-sta %s --timing-report-dir=- --top dut --show-convergence-table -o /dev/null | FileCheck %s --check-prefix=STA-TABLE
// RUN: circt-synth %s --top dut -o - | circt-sta --timing-report-dir=- --top dut --max-slew-iterations=4 --slew-epsilon=0.001 --slew-relative-epsilon=0.2 --slew-hint-damping=0.5 --adaptive-slew-damping-mode=conservative -o /dev/null | FileCheck %s --check-prefix=TUNE
// RUN: circt-sta %s --timing-report-dir=- --top dut --max-slew-iterations=4 --slew-epsilon=0.001 --slew-relative-epsilon=0.2 --slew-hint-damping=0.5 --adaptive-slew-damping-mode=conservative -o /dev/null | FileCheck %s --check-prefix=STA-TUNE

// CHECK: === Timing Report ===
// CHECK: Module: dut
// CHECK: Delay Model: nldm
// CHECK: Arrival Iterations:
// CHECK: Slew Converged:
// CHECK: Slew Trend Class:
// CHECK: Slew Reduction Ratio:
// CHECK: Slew Advice:
// CHECK: Path 1: delay = 25
// CHECK: Startpoint:
// CHECK: Endpoint:

// TABLE: === Timing Report ===
// TABLE: --- Slew Convergence ---
// TABLE: Iter | Max Slew Delta | Applied Damping
// TABLE: 1 | 0 | 1
// TABLE: Path 1: delay = 25

// STA: === Timing Report ===
// STA: Delay Model: nldm
// STA: Path 1: delay = 25

// STA-TABLE: === Timing Report ===
// STA-TABLE: --- Slew Convergence ---
// STA-TABLE: Iter | Max Slew Delta | Applied Damping
// STA-TABLE: 1 | 0 | 1

// TUNE: Slew Hint Damping: 5.000000e-01
// TUNE: Adaptive Slew Damping Mode: conservative
// TUNE: Relative Slew Epsilon: 2.000000e-01

// STA-TUNE: Slew Hint Damping: 5.000000e-01
// STA-TUNE: Adaptive Slew Damping Mode: conservative
// STA-TUNE: Relative Slew Epsilon: 2.000000e-01

module attributes {
  synth.liberty.library = {
    name = "dummy",
    time_unit = "1ns",
    default_input_transition = "0.5"
  },
  synth.nldm.default_input_slew = 0.5 : f64,
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
