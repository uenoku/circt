; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s

FIRRTL version 3.0.0
circuit test :
  module test :
    input x : UInt<8>
    ; expected-error @below {{expected expression in primitive operand}}
    node a = add(x, )

;// -----

FIRRTL version 3.0.0
circuit test :
  module test :
    input x : UInt<8>
    ; expected-error @below {{expected integer literal}}
    node a = bits(x, 4, x)

;// -----

FIRRTL version 3.0.0
circuit test :
  module test :
    input x : UInt<8>
    ; expected-error @below {{expected ')'}}
    node a = shl(x, 1, 2)
