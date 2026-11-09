include "circt/Dialect/FIRRTL/FIRRTLStatements.td"

class TestStatement<string format> : FIRRTLOp<"test.statement"> {
  let arguments = (ins ClockType:$clock, UInt1Type:$cond,
                       I32Attr:$number, StrAttr:$label);
  let results = (outs);
  let firrtlFormat = format;
}

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DBAD_NAME 2>&1 | FileCheck %s --check-prefix=NAME
#ifdef BAD_NAME
def BadOp : TestStatement<"`bad-name` `(` operands `,` $number `)` optionalName($label) info">;
// NAME: error: statement spelling must be a lexer and C++ token identifier
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DDUPLICATE 2>&1 | FileCheck %s --check-prefix=DUP
#ifdef DUPLICATE
def BadOp : TestStatement<"`stop` `(` operands `,` $number `)` optionalName($label) info">;
// DUP: error: duplicate FIRRTL source spelling 'stop'
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DBAD_LITERAL 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef BAD_LITERAL
def BadOp : TestStatement<"`test` `(` operands `,` $number ) optionalName($label) info">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DNO_NAME 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef NO_NAME
def BadOp : TestStatement<"`test` `(` operands `,` $number `)` info">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DWRONG_NAME 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef WRONG_NAME
def BadOp : TestStatement<"`test` `(` operands `,` $number `)` optionalName($wrong) info">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DNO_INFO 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef NO_INFO
def BadOp : TestStatement<"`test` `(` operands `,` $number `)` optionalName($label)">;
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DWRONG_ARG 2>&1 | FileCheck %s --check-prefix=FORMAT
#ifdef WRONG_ARG
def BadOp : TestStatement<"`test` `(` operands `,` $wrong `)` optionalName($label) info">;
#endif
// FORMAT: error: invalid statement firrtlFormat

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DKEYWORD_ARGS 2>&1 | FileCheck %s --check-prefix=KEYWORD
#ifdef KEYWORD_ARGS
def BadOp : TestStatement<"`test` info">;
// KEYWORD: error: keyword statement must have no arguments
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DVARIADIC 2>&1 | FileCheck %s --check-prefix=OPERANDS
#ifdef VARIADIC
def BadOp : TestStatement<"`test` `(` operands `,` $number `)` optionalName($label) info"> {
  let arguments = (ins Variadic<ClockType>:$clock, I32Attr:$number, StrAttr:$label);
}
// OPERANDS: error: statement operands must be named, fixed, and precede integer attributes
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DATTRIBUTE 2>&1 | FileCheck %s --check-prefix=ATTRIBUTE
#ifdef ATTRIBUTE
def BadOp : TestStatement<"`test` `(` operands `,` $number `)` optionalName($label) info"> {
  let arguments = (ins ClockType:$clock, I64Attr:$number, StrAttr:$label);
}
// ATTRIBUTE: error: statement attributes must be named, required I32Attr arguments
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DNAME_ATTR 2>&1 | FileCheck %s --check-prefix=NAME_ATTR
#ifdef NAME_ATTR
def BadOp : TestStatement<"`test` `(` operands `,` $number `)` optionalName($label) info"> {
  let arguments = (ins ClockType:$clock, I32Attr:$number, OptionalAttr<StrAttr>:$label);
}
// NAME_ATTR: error: statement name must be a named, required StrAttr argument
#endif
// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DREGION 2>&1 | FileCheck %s --check-prefix=SHAPE
#ifdef REGION
def BadOp : TestStatement<"`test` `(` operands `,` $number `)` optionalName($label) info"> {
  let regions = (region AnyRegion:$body);
}
// SHAPE: error: statement must have no results, regions, or successors
#endif
