// RUN: circt-opt --synth-annotate-techlib %s | FileCheck %s
// RUN: circt-opt --synth-annotate-techlib="delay-extraction=first" %s | FileCheck %s --check-prefix=FIRST
// RUN: circt-opt --synth-annotate-techlib="delay-extraction=center" %s | FileCheck %s --check-prefix=CENTER

// Test module with NLDM arcs and area annotation.
// cell_rise values: [0.012, 0.018, 0.020, 0.030], cell_fall values: [0.020]
// time_unit = 1000 ps/unit, so delays in ps.
// worst-case: max(0.030, 0.020) = 0.030 * 1000 = 30
// first: max(0.012, 0.020) = 0.020 * 1000 = 20
// center: cell_rise[2]=0.020, cell_fall[0]=0.020 -> max(0.020, 0.020) = 0.020 * 1000 = 20

// CHECK: hw.module @BUF
// CHECK-SAME: attributes {hw.techlib.info = {area = 1.000000e+00 : f64, delay = {{\[}}{{\[}}30]{{\]}}}, synth.liberty.area

// FIRST: hw.module @BUF
// FIRST-SAME: attributes {hw.techlib.info = {area = 1.000000e+00 : f64, delay = {{\[}}{{\[}}20]{{\]}}}, synth.liberty.area

// CENTER: hw.module @BUF
// CENTER-SAME: attributes {hw.techlib.info = {area = 1.000000e+00 : f64, delay = {{\[}}{{\[}}20]{{\]}}}, synth.liberty.area

module attributes {synth.nldm.time_unit = #synth.nldm_time_unit<1.000000e+03 : f64>} {
  hw.module @BUF(in %A : i1 {synth.liberty.pin = {capacitance = 2.000000e-03 : f64}}, out Y : i1 {synth.liberty.pin = {function = "A", synth.nldm.arcs = [#synth.nldm_arc<"A", "Y", "positive_unate", [1.000000e-02, 3.000000e-02], [0.000000e+00, 2.000000e-01], [1.200000e-02, 1.800000e-02, 2.000000e-02, 3.000000e-02], [], [], [2.000000e-02], [1.000000e-02, 3.000000e-02], [0.000000e+00, 2.000000e-01], [8.000000e-03, 1.100000e-02, 1.300000e-02, 1.600000e-02], [], [], [1.250000e-01]>]}}) attributes {synth.liberty.area = 1.000000e+00 : f64} {
    hw.output %A : i1
  }

  // Module without NLDM arcs should not be annotated.
  // CHECK: hw.module @NoArcs
  // CHECK-NOT: hw.techlib.info
  hw.module @NoArcs(in %a : i1, out b : i1) {
    hw.output %a : i1
  }
}
