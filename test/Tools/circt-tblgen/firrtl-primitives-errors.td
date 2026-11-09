include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

class TestPrimitive<string name> : FIRRTLOp<name> {
  let firrtlFormat = "`" # name # "` `(` $input `)`";
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
}

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DBAD_NAME 2>&1 | FileCheck %s --check-prefix=NAME
#ifdef BAD_NAME
def BadOp : TestPrimitive<"bad-name">;
// NAME: error: primitive spelling must be a lexer and C++ token identifier
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DDUPLICATE 2>&1 | FileCheck %s --check-prefix=DUP
#ifdef DUPLICATE
def DuplicateOp : TestPrimitive<"add">;
// DUP: error: duplicate FIRRTL primitive spelling 'add'
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DRESULT 2>&1 | FileCheck %s --check-prefix=RESULT
#ifdef RESULT
def BadOp : TestPrimitive<"bad"> { let results = (outs IntType:$a, IntType:$b); }
// RESULT: error: primitive must have exactly one fixed result and no regions or successors
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPTIONAL_RESULT 2>&1 | FileCheck %s --check-prefix=RESULT
#ifdef OPTIONAL_RESULT
def BadOp : TestPrimitive<"bad"> { let results = (outs Optional<IntType>:$result); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DREGION 2>&1 | FileCheck %s --check-prefix=RESULT
#ifdef REGION
def BadOp : TestPrimitive<"bad"> { let regions = (region AnyRegion:$body); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DSUCCESSOR 2>&1 | FileCheck %s --check-prefix=RESULT
#ifdef SUCCESSOR
def BadOp : TestPrimitive<"bad"> { let successors = (successor AnySuccessor:$dest); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DVARIADIC 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef VARIADIC
def BadOp : TestPrimitive<"bad"> {
  let arguments = (ins Variadic<IntType>:$inputs);
  let firrtlFormat = "`bad` `(` $inputs `)`";
}
// OPERANDS: error: primitive operands must be fixed and precede integer attributes
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPTIONAL_OPERAND 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef OPTIONAL_OPERAND
def BadOp : TestPrimitive<"bad"> { let arguments = (ins Optional<IntType>:$input); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DINTERLEAVED 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef INTERLEAVED
def BadOp : TestPrimitive<"bad"> {
  let arguments = (ins I32Attr:$n, IntType:$input);
  let firrtlFormat = "`bad` `(` $n `,` $input `)`";
}
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DWRONG_ATTR 2>&1 | FileCheck %s --check-prefix=ATTR
#ifdef WRONG_ATTR
def BadOp : TestPrimitive<"bad"> {
  let arguments = (ins IntType:$input, I64Attr:$n);
  let firrtlFormat = "`bad` `(` $input `,` $n `)`";
}
// ATTR: error: primitive attributes must be named, required I32Attr arguments
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPTIONAL_ATTR 2>&1 | FileCheck %s --check-prefix=ATTR
#ifdef OPTIONAL_ATTR
def BadOp : TestPrimitive<"bad"> {
  let arguments = (ins IntType:$input, OptionalAttr<I32Attr>:$n);
  let firrtlFormat = "`bad` `(` $input `,` $n `)`";
}
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DDEFAULT_ATTR 2>&1 | FileCheck %s --check-prefix=ATTR
#ifdef DEFAULT_ATTR
def BadOp : TestPrimitive<"bad"> {
  let arguments = (ins IntType:$input, DefaultValuedAttr<I32Attr, "1">:$n);
  let firrtlFormat = "`bad` `(` $input `,` $n `)`";
}
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DBAD_VERSION 2>&1 | FileCheck %s --check-prefix=VERSION
#ifdef BAD_VERSION
def BadOp : TestPrimitive<"bad"> { let firrtlVersion = FIRRTLReleasedVersion<65536, 0, 0>; }
// VERSION: error: FIRRTL version components must fit in uint16_t
#endif
