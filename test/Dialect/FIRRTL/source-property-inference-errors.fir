; RUN: circt-translate --import-firrtl --verify-diagnostics --split-input-file %s

FIRRTL version 7.0.0
circuit InvalidLeft :
  public module InvalidLeft :
    input value : Double
    ; expected-error @below {{lhs of prop_eq must be String, Bool, or Integer type}}
    propassert prop_eq(value, Integer(1)), "bad lhs"

;// -----

FIRRTL version 7.0.0
circuit InvalidRight :
  public module InvalidRight :
    input value : Double
    ; expected-error @below {{rhs of prop_eq must be String, Bool, or Integer type}}
    propassert prop_eq(Integer(1), value), "bad rhs"

;// -----

FIRRTL version 7.0.0
circuit MismatchedTypes :
  public module MismatchedTypes :
    ; expected-error @below {{prop_eq operands must have the same type}}
    propassert prop_eq(Integer(1), String("other")), "different types"

;// -----

FIRRTL version 7.0.0
circuit MissingOperand :
  public module MissingOperand :
    ; expected-error @below {{expected expression in primitive operand}}
    propassert prop_eq(Integer(1), ), "missing rhs"
