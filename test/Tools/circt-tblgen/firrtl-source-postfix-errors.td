include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

class TestIndex<string format> : FIRRTLOp<"test.index"> {
  let arguments = (ins IntType:$base, I32Attr:$index);
  let results = (outs IntType:$result);
  let firrtlFormat = format;
}

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DWRONG_BASE 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef WRONG_BASE
def BadOp : TestIndex<"$unknown `[` $index `]`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DWRONG_INDEX 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef WRONG_INDEX
def BadOp : TestIndex<"$base `[` $unknown `]`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DMISSING_BRACKET 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef MISSING_BRACKET
def BadOp : TestIndex<"$base `[` $index">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRAILING 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef TRAILING
def BadOp : TestIndex<"$base `[` $index `]` `extra`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DMISSING_FIELD 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef MISSING_FIELD
def BadOp : TestIndex<"$base `.` fieldName($other)">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DFIELD_OPERAND 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef FIELD_OPERAND
def BadOp : TestIndex<"$base `.` fieldName($index)"> {
  let arguments = (ins IntType:$base, IntType:$index);
}
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DEXTRA_ARGUMENT 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef EXTRA_ARGUMENT
def BadOp : TestIndex<"$base `[` $index `]`"> {
  let arguments = (ins IntType:$base, I32Attr:$index, I32Attr:$other);
}
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCUSTOM_PARSER 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef CUSTOM_PARSER
def BadOp : TestIndex<"$base `[` $index `]`"> {
  let firrtlHasCustomParser = 1;
}
#endif
// FORMAT: error: invalid postfix firrtlFormat: expected a fixed input and field name or index in ODS argument order

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DWRONG_ATTR 2>&1 | FileCheck %s --check-prefix=ATTR
#ifdef WRONG_ATTR
def BadOp : TestIndex<"$base `[` $index `]`"> {
  let arguments = (ins IntType:$base, I64Attr:$index);
}
// ATTR: error: primitive attributes must be named, required I32Attr arguments
#endif
