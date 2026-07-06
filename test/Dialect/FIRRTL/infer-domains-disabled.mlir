// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=check disable-domain=PowerDomain}))' %s --split-input-file --verify-diagnostics

// The "disable-domain" option strips the named domains from the circuit in a
// prepass, so they are removed before checking occurs.

// CHECK-LABEL: firrtl.circuit "DisabledChecks"
firrtl.circuit "DisabledChecks" {
  firrtl.domain @PowerDomain
  firrtl.domain @ClockDomain

  // expected-note @below {{in module "DisabledCrossing"}}
  firrtl.module @DisabledCrossing(
    in %A: !firrtl.domain<@PowerDomain()>,
    in %B: !firrtl.domain<@PowerDomain()>,
    // expected-error @below {{missing "ClockDomain" association for port "a"}}
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // PowerDomain is stripped, so the connection has no domain constraints.
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }

  // expected-note @below {{in module "DisabledChecks"}}
  firrtl.module @DisabledChecks(
    in %A: !firrtl.domain<@PowerDomain()>,
    // PowerDomain is stripped, so no association required.
    // expected-error @below {{missing "ClockDomain" association for port "x"}}
    in %x: !firrtl.uint<1>
  ) {}

  // Errors on other domains still fire
  firrtl.module @MixedClockErrors(
    // expected-note @below {{input module port CK1 declared here}}
    in %CK1: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{input module port CK2 declared here}}
    in %CK2: !firrtl.domain<@ClockDomain()>,
    in %PW: !firrtl.domain<@PowerDomain()>,
    // expected-note @below {{a has domains}}
    in %a: !firrtl.uint<1> domains [%CK1, %PW],
    // expected-note @below {{b has domains}}
    out %b: !firrtl.uint<1> domains [%CK2, %PW]
  ) {
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
