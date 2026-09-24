; RUN: firtool %s --ir-hw -o %t.hw.mlir
; RUN: not circt-opt --seq-check-domain='module-name=Top' %t.hw.mlir 2>&1 | FileCheck %s

FIRRTL version 4.0.0
circuit Top:
  public module Top:
    input expected: Clock
    input actual: Clock
    input in: UInt<8>
    reg r: UInt<8>, actual
    connect r, in
    intrinsic(circt_check_clock_domain, r, expected)

; CHECK: error: 'seq.check_clock_domain' op input depends on a sequential element clocked by module input 'actual', not the expected clock module input 'expected'
