// RUN: circt-opt --synth-print-resource-usage-analysis='top-module-name=top output-file="-"' %s | FileCheck %s
// RUN: circt-opt --synth-print-resource-usage-analysis='top-module-name=top output-file="-" emit-json=true' %s | FileCheck %s --check-prefix=JSON
// CHECK:      Resource Usage Analysis for module: top
// CHECK-NEXT: ========================================
// CHECK-NEXT: Total:
// CHECK-NEXT:   comb.and: 2
// CHECK-NEXT:   comb.or: 3
// CHECK-NEXT:   comb.xor: 2
// CHECK-NEXT:   synth.aig.and_inv: 2
// JSON: [{"instances":[{
// JSON:    "instanceName":"inst1",
// JSON:    "moduleName":"basic",
// JSON:    "usage":{
// JSON:      "instances":[],
// JSON:      "local":{
// JSON:        "comb.and":1,
// JSON:        "comb.or":1,
// JSON:        "comb.xor":1,
// JSON:        "synth.aig.and_inv":1
// JSON:      }
// JSON:    }
// JSON:    "instanceName":"inst2",
// JSON:    "moduleName":"basic",

hw.module private @basic(in %a : i1, in %b : i1, out x : i1) {
  %p = synth.aig.and_inv not %a, %b : i1
  %q = comb.and %p, %a : i1
  %r = comb.or %q, %a : i1
  %s = comb.xor %r, %a : i1
  hw.output %s : i1
}
hw.module private @top(in %a : i1, in %b : i1, out x : i1) {
  %0 = hw.instance "inst1" @basic(a: %a: i1, b: %b: i1) -> (x: i1)
  %1 = hw.instance "inst2" @basic(a: %a: i1, b: %b: i1) -> (x: i1)
  %s = comb.or %0, %1 : i1
  hw.output %s : i1
}

hw.module private @unrelated(in %a : i1, in %b : i1, out x : i1) {
  %p = synth.aig.and_inv not %a, %b : i1
  hw.output %p : i1
}