// RUN: not circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all debug-domains-html=%t}))' %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: FileCheck %s --input-file=%t/IllegalDomainCrossing.domain.html --check-prefix=HTML

// DIAG: error: illegal domain crossing
// DIAG: note: domain inference debugger written to

// HTML-DAG: FIRRTL Domain Inference Debugger
// HTML-DAG: "module": "IllegalDomainCrossing"
// HTML-DAG: "focusModule": "IllegalDomainCrossing"
// HTML-DAG: "failureKind": "illegal-domain-crossing"
// HTML-DAG: $root/bad:IllegalDomainCrossing
// HTML-DAG: renderLifetimeAnnotations
// HTML-DAG: Why This Failed
// HTML-DAG: Lifetime Annotations
// HTML-DAG: Constraint Graph
// HTML-DAG: Conflicting Clusters

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
