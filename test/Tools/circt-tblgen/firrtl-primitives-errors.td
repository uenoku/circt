include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

class TestPrimitive<string name> : FIRRTLOp<name>, FIRRTLPrimitive<name> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
}

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DNOT_OP 2>&1 | FileCheck %s --check-prefix=NOT_OP
#ifdef NOT_OP
def NotAnOp : FIRRTLPrimitive<"not_an_op">;
// NOT_OP: error: FIRRTLPrimitive must be an operation
#endif

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
def BadOp : TestPrimitive<"bad"> { let arguments = (ins Variadic<IntType>:$inputs); }
// OPERANDS: error: primitive operands must be fixed and precede integer attributes
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPTIONAL_OPERAND 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef OPTIONAL_OPERAND
def BadOp : TestPrimitive<"bad"> { let arguments = (ins Optional<IntType>:$input); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DINTERLEAVED 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef INTERLEAVED
def BadOp : TestPrimitive<"bad"> { let arguments = (ins I32Attr:$n, IntType:$input); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DWRONG_ATTR 2>&1 | FileCheck %s --check-prefix=ATTR
#ifdef WRONG_ATTR
def BadOp : TestPrimitive<"bad"> { let arguments = (ins IntType:$input, I64Attr:$n); }
// ATTR: error: primitive attributes must be named, required I32Attr arguments
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPTIONAL_ATTR 2>&1 | FileCheck %s --check-prefix=ATTR
#ifdef OPTIONAL_ATTR
def BadOp : TestPrimitive<"bad"> { let arguments = (ins IntType:$input, OptionalAttr<I32Attr>:$n); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DDEFAULT_ATTR 2>&1 | FileCheck %s --check-prefix=ATTR
#ifdef DEFAULT_ATTR
def BadOp : TestPrimitive<"bad"> { let arguments = (ins IntType:$input, DefaultValuedAttr<I32Attr, "1">:$n); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DBAD_VERSION 2>&1 | FileCheck %s --check-prefix=VERSION
#ifdef BAD_VERSION
def BadOp : TestPrimitive<"bad"> { let firrtlVersion = FIRRTLReleasedVersion<65536, 0, 0>; }
// VERSION: error: FIRRTL version components must fit in uint16_t
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DBAD_SPLIT 2>&1 | FileCheck %s --check-prefix=VERSION
#ifdef BAD_SPLIT
def BadOp : TestPrimitive<"bad"> {
  let firrtlVersion = FIRRTLSplitVersion<FIRRTLBaseVersion,
                                         FIRRTLReleasedVersion<65536, 0, 0>>;
}
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DNESTED_SPLIT 2>&1 | FileCheck %s --check-prefix=NESTED
#ifdef NESTED_SPLIT
def BadOp : TestPrimitive<"bad"> {
  let firrtlVersion = FIRRTLSplitVersion<FIRRTLBaseVersion,
      FIRRTLSplitVersion<FIRRTLBaseVersion, FIRRTLUnspecifiedVersion>>;
}
// NESTED: error: expected a released, unspecified, or next FIRRTL source version
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_NOT_OP 2>&1 | FileCheck %s --check-prefix=CALL_NOT_OP
#ifdef CALL_NOT_OP
def NotAnOp : FIRRTLSourceCall<"other", FIRRTLBaseVersion, "other", FIRRTLUnaryCall>;
// CALL_NOT_OP: error: FIRRTLSourceCall must be an operation
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_BOTH 2>&1 | FileCheck %s --check-prefix=CALL_BOTH
#ifdef CALL_BOTH
def BadOp : TestPrimitive<"bad">,
            FIRRTLSourceCall<"other", FIRRTLBaseVersion, "other", FIRRTLUnaryCall>;
// CALL_BOTH: error: operation cannot be both a primitive and a source call
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_ATTR 2>&1 | FileCheck %s --check-prefix=CALL_ATTR
#ifdef CALL_ATTR
def BadOp : FIRRTLOp<"bad">,
            FIRRTLSourceCall<"other", FIRRTLBaseVersion, "other", FIRRTLPrimitiveCall> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs IntType:$result);
}
// CALL_ATTR: error: source call arguments must be operands
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_UNARY 2>&1 | FileCheck %s --check-prefix=CALL_UNARY
#ifdef CALL_UNARY
def BadOp : FIRRTLOp<"bad">,
            FIRRTLSourceCall<"other", FIRRTLBaseVersion, "other", FIRRTLUnaryCall> {
  let arguments = (ins Variadic<IntType>:$inputs);
  let results = (outs IntType:$result);
}
// CALL_UNARY: error: unary source call must have one fixed operand
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_DUP 2>&1 | FileCheck %s --check-prefix=CALL_DUP
#ifdef CALL_DUP
def BadOp : FIRRTLOp<"bad">,
            FIRRTLSourceCall<"add", FIRRTLBaseVersion, "add", FIRRTLUnaryCall> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
}
// CALL_DUP: error: duplicate FIRRTL source spelling 'add'
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_STYLE 2>&1 | FileCheck %s --check-prefix=CALL_STYLE
#ifdef CALL_STYLE
def OtherStyle : FIRRTLSourceCallStyle;
def BadOp : FIRRTLOp<"bad">,
            FIRRTLSourceCall<"other", FIRRTLBaseVersion, "other", OtherStyle> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
}
// CALL_STYLE: error: unknown source call style
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRANSPARENT_NOT_OP 2>&1 | FileCheck %s --check-prefix=TRANSPARENT_NOT_OP
#ifdef TRANSPARENT_NOT_OP
def NotAnOp : FIRRTLSourceTransparent;
// TRANSPARENT_NOT_OP: error: FIRRTLSourceTransparent must be an operation
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRANSPARENT_PRIMITIVE 2>&1 | FileCheck %s --check-prefix=TRANSPARENT_BOTH
#ifdef TRANSPARENT_PRIMITIVE
def BadOp : TestPrimitive<"bad">, FIRRTLSourceTransparent;
// TRANSPARENT_BOTH: error: transparent operation cannot also be a primitive or source call
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRANSPARENT_CALL 2>&1 | FileCheck %s --check-prefix=TRANSPARENT_BOTH
#ifdef TRANSPARENT_CALL
def BadOp : FIRRTLOp<"bad">, FIRRTLSourceTransparent,
            FIRRTLSourceCall<"other", FIRRTLBaseVersion, "other", FIRRTLUnaryCall> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
}
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRANSPARENT_OPTIONAL 2>&1 | FileCheck %s --check-prefix=TRANSPARENT_SHAPE
#ifdef TRANSPARENT_OPTIONAL
def BadOp : FIRRTLOp<"bad">, FIRRTLSourceTransparent {
  let arguments = (ins Optional<IntType>:$input);
  let results = (outs IntType:$result);
}
// TRANSPARENT_SHAPE: error: transparent operation must have one fixed operand and result, and no attributes, properties, regions or successors
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRANSPARENT_ATTR 2>&1 | FileCheck %s --check-prefix=TRANSPARENT_SHAPE
#ifdef TRANSPARENT_ATTR
def BadOp : FIRRTLOp<"bad">, FIRRTLSourceTransparent {
  let arguments = (ins IntType:$input, I32Attr:$attr);
  let results = (outs IntType:$result);
}
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRANSPARENT_RESULT 2>&1 | FileCheck %s --check-prefix=TRANSPARENT_SHAPE
#ifdef TRANSPARENT_RESULT
def BadOp : FIRRTLOp<"bad">, FIRRTLSourceTransparent {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$first, IntType:$second);
}
#endif
