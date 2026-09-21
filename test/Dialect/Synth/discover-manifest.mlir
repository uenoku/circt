// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{policy-file=%S/Inputs/refine-policy.json refinement-report=-})' 2>&1 | FileCheck %s
// RUN: not circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{policy-file=%S/Inputs/refine-policy.json policy-file-strict})' 2>&1 | FileCheck %s --check-prefix=STRICT

// Explicit manifest overrides structural selection: @arith:0 gets booth
// even though the structural knob would choose reduce-delay. The unknown
// region ID warns (permissive) and the bogus policy is diagnosed and
// skipped, leaving @arith:3 on the structural policy.
// CHECK: warning: datapath policy manifest: unknown policy 'bogus'
// CHECK: warning: datapath policy manifest: unknown region '@arith:NOPE'
// CHECK: {"id": "@arith:0", "module": "arith", "policy": "booth", "reason": "policy manifest"
// CHECK: {"id": "@arith:3", "module": "arith", "policy": "reduce-delay", "reason": "structural pressure"
// STRICT: error: invalid policy file '
hw.module @arith(in %a : i8, in %b : i8, in %c : i8, in %d : i8, in %e : i8, out x : i8, out y : i8) {
  %add = comb.add %a, %b, %c : i8
  %p1 = synth.timing_probe %add : i8 {required_time = 10000 : i64}
  %mul = comb.mul %d, %e : i8
  %p2 = synth.timing_probe %mul : i8 {required_time = 10000 : i64}
  hw.output %p1, %p2 : i8, i8
}
