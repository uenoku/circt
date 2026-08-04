; Example: Registry<Path> clockGates through firtool + domaintool
;
;   ./build/bin/firtool examples/registry-clock-gates.fir \
;       -domain-mode=check -output-final-mlir /tmp/reg.mlir -o /dev/null
;
;   ./build/bin/domaintool --module Foo \
;       --domain ClockDomain,A,,synchronous --assign 0 \
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

  ; Child accumulates one local clock gate into domain A's registry.
  public module Bar:
    input A: Domain of ClockDomain
    input d: UInt<1> domains [A]

    inst clockGate of ClockGate
    register A[clockGates], path("OMReferenceTarget:~Foo|Bar>clockGate")

  ; Parent fans the same domain into two children; registries concatenate.
  public module Foo:
    input A: Domain of ClockDomain
    input a: UInt<1> domains [A]
    input b: UInt<1> domains [A]

    inst bar1 of Bar
    inst bar2 of Bar

    domain_define bar1.A = A
    domain_define bar2.A = A
    connect bar1.d, a
    connect bar2.d, b
