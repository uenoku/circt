; Example: multi-domain, multi-depth Registry clockGates
;
; Hierarchy:
;   Foo
;     gateA0          -- local gate on domain A (depth 0)
;     midA : MidA
;       gateA1        -- domain A (depth 1)
;       leafA : LeafA
;         gateA2      -- domain A (depth 2)
;     midB : MidB
;       gateB1        -- domain B (depth 1)
;       leafB : LeafB
;         gateB2a     -- domain B (depth 2)
;         gateB2b     -- domain B (depth 2, second gate)
;
; Run:
;   ./build/bin/firtool examples/registry-clock-gates.fir \
;       -domain-mode=check -output-final-mlir /tmp/reg.mlir -o /dev/null
;
;   ./build/bin/domaintool --module Foo \
;       --domain ClockDomain,A,,synchronous \
;       --domain ClockDomain,B,A,synchronous \
;       --assign 0 --assign 1 \
;       /tmp/reg.mlir

FIRRTL version 7.0.0
circuit Foo:
  ; Fields required by domaintool's Clock Spec JSON handler, plus a registry.
  domain ClockDomain:
    name: String
    source: String
    relationship: String
    clockGates: Registry<Path>

  extmodule ClockGate:

  ; Deepest leaf on domain A: one gate.
  public module LeafA:
    input A: Domain of ClockDomain
    input d: UInt<1> domains [A]

    inst gateA2 of ClockGate
    register A[clockGates], ref_instance(gateA2)

  ; Mid level on domain A: local gate + deeper leaf.
  public module MidA:
    input A: Domain of ClockDomain
    input d: UInt<1> domains [A]

    inst gateA1 of ClockGate
    register A[clockGates], ref_instance(gateA1)

    inst leafA of LeafA
    domain_define leafA.A = A
    connect leafA.d, d

  ; Deepest leaf on domain B: two gates at the same level.
  public module LeafB:
    input B: Domain of ClockDomain
    input d: UInt<1> domains [B]

    inst gateB2a of ClockGate
    inst gateB2b of ClockGate
    register B[clockGates], ref_instance(gateB2a)
    register B[clockGates], ref_instance(gateB2b)

  ; Mid level on domain B: local gate + deeper leaf.
  public module MidB:
    input B: Domain of ClockDomain
    input d: UInt<1> domains [B]

    inst gateB1 of ClockGate
    register B[clockGates], ref_instance(gateB1)

    inst leafB of LeafB
    domain_define leafB.B = B
    connect leafB.d, d

  ; Top: two domains, mixed depths, plus a local A gate.
  public module Foo:
    input A: Domain of ClockDomain
    input B: Domain of ClockDomain
    input a: UInt<1> domains [A]
    input b: UInt<1> domains [B]

    ; Domain A local at the top (depth 0).
    inst gateA0 of ClockGate
    register A[clockGates], ref_instance(gateA0)

    ; Domain A subtree: depth 1 and 2.
    inst midA of MidA
    domain_define midA.A = A
    connect midA.d, a

    ; Domain B subtree: depth 1 and 2 (two gates at depth 2).
    inst midB of MidB
    domain_define midB.B = B
    connect midB.d, b
