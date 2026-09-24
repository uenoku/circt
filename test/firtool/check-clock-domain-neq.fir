; RUN: firtool %s --ir-hw -o %t.hw.mlir
; RUN: not circt-opt --seq-check-domain='module-name=Top' %t.hw.mlir 2>&1 | FileCheck %s

FIRRTL version 4.0.0
circuit Top:
  public module Top:
    input expected: Clock
    input in: UInt<8>
    reg r: UInt<8>, expected
    connect r, in
    intrinsic(circt_check_clock_domain_neq, r, expected)

; CHECK: error: 'seq.check_clock_domain_neq' op input depends on a sequential element clocked by module input 'expected', but it must not depend on the expected clock module input 'expected'
