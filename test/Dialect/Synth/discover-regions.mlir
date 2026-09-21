// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-report=-})' | FileCheck %s --check-prefix=REPORT
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-policy=baseline refinement-report=-})' | FileCheck %s --check-prefix=BASELINE

// Two datapath regions (add cone, multiplier cone), each feeding a probe.
// Both cones are structurally critical, so the default policy selects both.
// REPORT: {"schema": "circt.synth.datapath-refinement-report.v1", "regions": [{"id": "@arith:0", "module": "arith", "policy": "reduce-delay", "reason": "structural pressure", "violated": false, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@arith:2"], "endpoints": []}, {"id": "@arith:3", "module": "arith", "policy": "reduce-delay", "reason": "structural pressure", "violated": false, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@arith:6"], "endpoints": []}]
// BASELINE: {"schema": "circt.synth.datapath-refinement-report.v1", "regions": [{"id": "@arith:0", "module": "arith", "policy": "baseline", "reason": "relaxed (no pressure)", "violated": false, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@arith:2"], "endpoints": []}, {"id": "@arith:3", "module": "arith", "policy": "baseline", "reason": "relaxed (no pressure)", "violated": false, "confidence": "StructuralEstimate", "worst_slack_levels": 0, "probes": ["@arith:6"], "endpoints": []}]
hw.module @arith(in %a : i8, in %b : i8, in %c : i8, in %d : i8, in %e : i8, out x : i8, out y : i8) {
  %add = comb.add %a, %b, %c : i8
  %p1 = synth.timing_probe %add : i8 {required_time = 10000 : i64}
  %mul = comb.mul %d, %e : i8
  %p2 = synth.timing_probe %mul : i8 {required_time = 10000 : i64}
  hw.output %p1, %p2 : i8, i8
}
