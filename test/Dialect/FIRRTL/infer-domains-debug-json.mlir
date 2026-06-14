// RUN: not circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all debug-domains-json=%t}))' %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: FileCheck %s --input-file=%t/IllegalDomainCrossing.domain.json --check-prefix=JSON

// DIAG: error: illegal domain crossing
// DIAG: note: conflict source: this connection drives b from a
// DIAG: note: domain inference debug JSON written to

// JSON-DAG: "module": "IllegalDomainCrossing"
// JSON-DAG: "focusModule": "IllegalDomainCrossing"
// JSON-DAG: "failureKind": "illegal-domain-crossing"
// JSON-DAG: $root/bad:IllegalDomainCrossing
// JSON-DAG: "hierarchy"
// JSON-DAG: "aliases"
// JSON-DAG: "annotations"
// JSON-DAG: "suggestions"
// JSON-DAG: IllegalDomainCrossing.b
// JSON-DAG: IllegalDomainCrossing.a
// JSON-DAG: ClockDomain

firrtl.circuit "Top" {
  firrtl.domain @ClockDomain
  firrtl.module @IllegalDomainCrossing(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
  firrtl.module @Top() {
    %bad_A, %bad_B, %bad_a, %bad_b = firrtl.instance bad @IllegalDomainCrossing(in A: !firrtl.domain<@ClockDomain()>, in B: !firrtl.domain<@ClockDomain()>, in a: !firrtl.uint<1> domains [A], out b: !firrtl.uint<1> domains [B])
  }
}
