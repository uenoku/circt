// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=check disable-domain=PowerDomain}))' %s --split-input-file --verify-diagnostics | FileCheck %s

// The "disable-domain" option excludes the named domains from checking, so
// their errors do not interfere with development of other domains.

// A domain crossing purely within a disabled domain is suppressed.
//
// CHECK-LABEL: firrtl.circuit "DisabledCrossing"
firrtl.circuit "DisabledCrossing" {
  firrtl.domain @PowerDomain
  firrtl.module @DisabledCrossing(
    in %A: !firrtl.domain<@PowerDomain()>,
    in %B: !firrtl.domain<@PowerDomain()>,
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // No error: the PowerDomain crossing is suppressed.
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// -----

// Missing, duplicate, and undriven-port errors are also suppressed for a
// disabled domain.
//
// CHECK-LABEL: firrtl.circuit "DisabledChecks"
firrtl.circuit "DisabledChecks" {
  firrtl.domain @PowerDomain
  firrtl.module @DisabledChecks(
    in %A: !firrtl.domain<@PowerDomain()>,
    // No "missing PowerDomain association" error for %x.
    in %x: !firrtl.uint<1>,
    // No "duplicate PowerDomain association" error for %a.
    in %a: !firrtl.uint<1> domains [%A, %A],
    // No "undriven domain port" error for %c.
    out %c: !firrtl.domain<@PowerDomain()>
  ) {}
}

// -----

// Disabling a domain is name-specific: errors on other domains still fire, and
// a crossing that also involves a non-disabled domain is still reported. Note
// that the disabled PowerDomain does not contribute source notes.
firrtl.circuit "MixedClockErrors" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain
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
