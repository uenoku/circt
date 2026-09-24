; RUN: firtool %s --ir-hw -o %t.hw.mlir
; RUN: circt-opt --seq-check-domain='module-name=Top' %t.hw.mlir

FIRRTL version 4.0.0
circuit Top:
  public module Top:
    input clock: Clock
    input in: UInt<8>
    output out: UInt<8>
    reg r: UInt<8>, clock
    connect r, in
    node value = add(r, UInt<8>(1))
    connect out, value
    intrinsic(circt_check_clock_domain, value, clock)
