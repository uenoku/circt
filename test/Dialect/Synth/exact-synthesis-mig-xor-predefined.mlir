// REQUIRES: z3-integration
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-gen-predefined{kind=npn max-inputs=2},synth-exact-synthesis{kind=mig-xor sat-solver=z3})' > /dev/null 2>&1

module {
}
