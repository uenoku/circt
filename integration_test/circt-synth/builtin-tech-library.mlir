// RUN: circt-synth %s --tech-library=asap7 --until-after=mapping | FileCheck %s --check-prefix=ASAP7
// RUN: circt-synth %s --tech-library=sky130 --until-after=mapping | FileCheck %s --check-prefix=SKY130

// ASAP7-LABEL: hw.module @simple_and(
// ASAP7: hw.instance {{.*}} @AND2x2_ASAP7_75t_R
// SKY130-LABEL: hw.module @simple_and(
// SKY130: hw.instance {{.*}} @sky130_fd_sc_hd__and2_2
hw.module @simple_and(in %a : i1, in %b : i1, out result : i1) {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}
