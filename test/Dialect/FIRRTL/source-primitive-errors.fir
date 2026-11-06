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

;// -----

FIRRTL version 3.0.0
circuit test :
  module test :
    input x : UInt<8>
    ; Preserve the existing integer-literal parser's treatment of -1.
    ; expected-error @below {{high must be equal or greater than low, but got high = 4, low = 4294967295}}
    node a = bits(x, 4, -1)

;// -----

FIRRTL version 3.0.0
circuit test :
  module test :
    input x : UInt<8>
    ; expected-error @below {{high must be smaller than the width of input, but got high = 4294967296, width = 8}}
    node a = bits(x, 4294967296, 0)

;// -----

FIRRTL version 3.0.0
circuit test :
  module test :
    input x : UInt<8>
    ; expected-error @below {{value is too big to handle}}
    node a = shl(x, -4294967296)

;// -----

FIRRTL version 3.0.0
circuit test :
  module test :
    input x : UInt<8>
    ; expected-error @below {{value is too big to handle}}
    node a = shr(x, -4294967296)
