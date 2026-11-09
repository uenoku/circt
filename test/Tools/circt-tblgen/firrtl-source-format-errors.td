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

// A format must describe the syntax, not just name an implementation category.
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DCALL_DIRECTIVE 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef CALL_DIRECTIVE
def BadOp : TestFormat<"call">;
#endif

// Mnemonics used in an inherited format must still be valid source spellings.
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DINHERITED_NAME 2>&1 | FileCheck %s --check-prefix=NAME
#ifdef INHERITED_NAME
def BadOp : IntBinaryPrimOp<"bad.mnemonic", IntType>;
#endif
// NAME: error: primitive spelling must be a lexer and C++ token identifier
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
