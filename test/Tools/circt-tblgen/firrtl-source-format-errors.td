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
// FORMAT: error: invalid firrtlFormat: expected a call with all ODS arguments in order

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
  let firrtlFormat = "`test` `(` $input `)`";
}
// OPERANDS: error: primitive operands must be fixed and precede integer attributes
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
