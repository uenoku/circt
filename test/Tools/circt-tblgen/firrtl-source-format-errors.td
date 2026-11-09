include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

class TestFormat<string format> : FIRRTLOp<"test.format"> {
  let arguments = (ins IntType:$input, I32Attr:$hi, I32Attr:$lo);
  let results = (outs IntType:$result);
  let firrtlFormat = format;
}

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DMALFORMED 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef MALFORMED
def BadOp : TestFormat<"`test` `(` $input `,` $hi `,` $lo )">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DUNKNOWN 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef UNKNOWN
def BadOp : TestFormat<"`test` `(` $unknown `,` $hi `,` $lo `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DMISSING 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef MISSING
def BadOp : TestFormat<"`test` `(` $input `,` $hi `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DREPEATED 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef REPEATED
def BadOp : TestFormat<"`test` `(` $input `,` $hi `,` $hi `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DREORDERED 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef REORDERED
def BadOp : TestFormat<"`test` `(` $input `,` $lo `,` $hi `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DDIRECTIVE 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef DIRECTIVE
def BadOp : TestFormat<"`test` `(` $input `,` $hi `,` $lo `)` attr-dict">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DREPEATED_OPERANDS 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef REPEATED_OPERANDS
def BadOp : TestFormat<"`test` `(` operands `,` operands `,` $hi `,` $lo `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DMIXED_OPERANDS 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef MIXED_OPERANDS
def BadOp : TestFormat<"`test` `(` operands `,` $input `,` $hi `,` $lo `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPERANDS_PREFIX 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef OPERANDS_PREFIX
def BadOp : TestFormat<"`test` `(` operands_extra `,` $hi `,` $lo `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DMISSING_ATTRIBUTE 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef MISSING_ATTRIBUTE
def BadOp : TestFormat<"`test` `(` operands `,` $hi `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DREORDERED_ATTRIBUTE 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef REORDERED_ATTRIBUTE
def BadOp : TestFormat<"`test` `(` operands `,` $lo `,` $hi `)`">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPERANDS_WITHOUT_OPERAND 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef OPERANDS_WITHOUT_OPERAND
def BadOp : FIRRTLOp<"bad"> {
  let arguments = (ins I32Attr:$hi);
  let results = (outs IntType:$result);
  let firrtlFormat = "`test` `(` operands `,` $hi `)`";
}
#endif
// FORMAT: error: invalid firrtlFormat: expected a call with all ODS arguments in order

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_ATTRIBUTE 2>&1 | FileCheck %s --check-prefix=CALL_ATTRIBUTE
#ifdef CALL_ATTRIBUTE
def BadOp : TestFormat<"call">;
// CALL_ATTRIBUTE: error: call shorthand requires only operands
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_MNEMONIC 2>&1 | FileCheck %s --check-prefix=CALL_MNEMONIC
#ifdef CALL_MNEMONIC
def BadOp : FIRRTLOp<"bad.mnemonic"> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let firrtlFormat = "call";
}
// CALL_MNEMONIC: error: call shorthand requires a lexer-compatible MLIR mnemonic
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_VARIADIC 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef CALL_VARIADIC
def BadOp : FIRRTLOp<"bad"> {
  let arguments = (ins Variadic<IntType>:$inputs);
  let results = (outs IntType:$result);
  let firrtlFormat = "call";
}
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_DUPLICATE 2>&1 | FileCheck %s --check-prefix=DUP
#ifdef CALL_DUPLICATE
def BadOp : FIRRTLOp<"add"> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let firrtlFormat = "call";
}
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_NO_OPERAND 2>&1 | FileCheck %s --check-prefix=NO_OPERAND
#ifdef CALL_NO_OPERAND
def BadOp : FIRRTLOp<"bad"> {
  let arguments = (ins);
  let results = (outs IntType:$result);
  let firrtlFormat = "call";
}
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DNAME 2>&1 | FileCheck %s --check-prefix=NAME
#ifdef NAME
def BadOp : TestFormat<"`bad-name` `(` $input `,` $hi `,` $lo `)`">;
// NAME: error: primitive spelling must be a lexer and C++ token identifier
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DDUPLICATE 2>&1 | FileCheck %s --check-prefix=DUP
#ifdef DUPLICATE
def BadOp : TestFormat<"`add` `(` $input `,` $hi `,` $lo `)`">;
// DUP: error: duplicate FIRRTL primitive spelling 'add'
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DLEGACY 2>&1 | FileCheck %s --check-prefix=LEGACY
#ifdef LEGACY
def BadOp : TestFormat<"`test` `(` $input `,` $hi `,` $lo `)`">,
            FIRRTLSourceCall<"test", FIRRTLBaseVersion, "test", FIRRTLUnaryCall>;
// LEGACY: error: firrtlFormat cannot be combined with source call or transparent mixins
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DVERSION 2>&1 | FileCheck %s --check-prefix=VERSION
#ifdef VERSION
def BadOp : TestFormat<"`test` `(` $input `,` $hi `,` $lo `)`"> {
  let firrtlVersion = FIRRTLReleasedVersion<65536, 0, 0>;
}
// VERSION: error: FIRRTL version components must fit in uint16_t
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DRESULT 2>&1 | FileCheck %s --check-prefix=RESULT
#ifdef RESULT
def BadOp : TestFormat<"`test` `(` $input `,` $hi `,` $lo `)`"> {
  let results = (outs IntType:$a, IntType:$b);
}
// RESULT: error: primitive must have exactly one fixed result and no regions or successors
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DVARIADIC 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef VARIADIC
def BadOp : FIRRTLOp<"bad"> {
  let arguments = (ins Variadic<IntType>:$input);
  let results = (outs IntType:$result);
  let firrtlFormat = "`test` `(` operands `)`";
}
// OPERANDS: error: primitive operands must be fixed and precede integer attributes
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DINTERLEAVED 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef INTERLEAVED
def BadOp : FIRRTLOp<"bad"> {
  let arguments = (ins IntType:$lhs, I32Attr:$amount, IntType:$rhs);
  let results = (outs IntType:$result);
  let firrtlFormat = "`test` `(` operands `,` $amount `,` $rhs `)`";
}
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DATTRIBUTE 2>&1 | FileCheck %s --check-prefix=ATTRIBUTE
#ifdef ATTRIBUTE
def BadOp : FIRRTLOp<"bad"> {
  let arguments = (ins IntType:$input, OptionalAttr<I32Attr>:$amount);
  let results = (outs IntType:$result);
  let firrtlFormat = "`test` `(` $input `,` $amount `)`";
}
// ATTRIBUTE: error: primitive attributes must be named, required I32Attr arguments
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DPROPERTY 2>&1 | FileCheck %s --check-prefix=PROPERTY
#ifdef PROPERTY
def BadOp : FIRRTLOp<"bad"> {
  let arguments = (ins IntType:$input, I32Prop:$amount);
  let results = (outs IntType:$result);
  let firrtlFormat = "`test` `(` $input `,` $amount `)`";
}
// PROPERTY: error: primitive properties are not supported
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DUNKNOWN_VERSION 2>&1 | FileCheck %s --check-prefix=UNKNOWN_VERSION
#ifdef UNKNOWN_VERSION
def UnknownVersion : FIRRTLSourceVersion;
def BadOp : TestFormat<"`test` `(` $input `,` $hi `,` $lo `)`"> {
  let firrtlVersion = UnknownVersion;
}
// UNKNOWN_VERSION: error: expected a released, unspecified, or next FIRRTL source version
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DNO_OPERAND 2>&1 | FileCheck %s --check-prefix=NO_OPERAND
#ifdef NO_OPERAND
def BadOp : FIRRTLOp<"bad"> {
  let arguments = (ins I32Attr:$hi);
  let results = (outs IntType:$result);
  let firrtlFormat = "`test` `(` $hi `)`";
}
// NO_OPERAND: error: primitive must have an expression operand
#endif
