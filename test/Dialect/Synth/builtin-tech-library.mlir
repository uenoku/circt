// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{builtin-library=asap7 strategy=area test=true max-cuts-per-root=8})' %s | FileCheck %s --check-prefix=ASAP7
// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{builtin-library=sky130 strategy=area test=true max-cuts-per-root=8})' %s | FileCheck %s --check-prefix=SKY130
// RUN: not circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{builtin-library=missing})' %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN

// ASAP7-LABEL: hw.module @simple_and(
// ASAP7: hw.instance {{.*}} @AND2x2_ASAP7_75t_R
// SKY130-LABEL: hw.module @simple_and(
// SKY130: hw.instance {{.*}} @sky130_fd_sc_hd__and2_2
// UNKNOWN: unknown built-in tech library 'missing'; expected one of: asap7, sky130
hw.module @simple_and(in %a : i1, in %b : i1, out result : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
