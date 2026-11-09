include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

class TestEmitter<string format> : FIRRTLOp<"test.emitter"> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let firrtlFormat = format;
}

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRANSPARENT_PARSER 2>&1 | FileCheck %s --check-prefix=TRANSPARENT_PARSER
#ifdef TRANSPARENT_PARSER
def BadOp : TestEmitter<"$input"> { let firrtlHasCustomParser = 1; }
// TRANSPARENT_PARSER: error: transparent format cannot have a custom parser
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DWRONG_REF 2>&1 | FileCheck %s --check-prefix=REFERENCE
#ifdef WRONG_REF
def BadOp : TestEmitter<"$unknown">;
// REFERENCE: error: transparent format must reference its single fixed operand
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DTRAILING 2>&1 | FileCheck %s --check-prefix=REFERENCE
#ifdef TRAILING
def BadOp : TestEmitter<"$input `extra`">;
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DVARIADIC_CAST 2>&1 | FileCheck %s --check-prefix=REFERENCE
#ifdef VARIADIC_CAST
def BadOp : TestEmitter<"$inputs"> { let arguments = (ins Variadic<IntType>:$inputs); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPTIONAL_CAST 2>&1 | FileCheck %s --check-prefix=REFERENCE
#ifdef OPTIONAL_CAST
def BadOp : TestEmitter<"$input"> { let arguments = (ins Optional<IntType>:$input); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DMULTIPLE 2>&1 | FileCheck %s --check-prefix=SHAPE
#ifdef MULTIPLE
def BadOp : TestEmitter<"$input"> { let arguments = (ins IntType:$input, IntType:$other); }
// SHAPE: error: transparent format must have one fixed operand and no attributes
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DATTRIBUTE 2>&1 | FileCheck %s --check-prefix=SHAPE
#ifdef ATTRIBUTE
def BadOp : TestEmitter<"$input"> { let arguments = (ins IntType:$input, I32Attr:$amount); }
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DRESULT 2>&1 | FileCheck %s --check-prefix=RESULT
#ifdef RESULT
def BadOp : TestEmitter<"$input"> { let results = (outs); }
// RESULT: error: primitive must have exactly one fixed result and no regions or successors
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DVARIADIC_REFERENCE 2>&1 | FileCheck %s --check-prefix=FIXED
#ifdef VARIADIC_REFERENCE
def BadOp : TestEmitter<"`test` `(` $inputs `)`"> {
  let arguments = (ins Variadic<IntType>:$inputs);
  let firrtlHasCustomParser = 1;
}
// FIXED: error: primitive operands must be fixed and precede integer attributes
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DOPTIONAL_OPERAND 2>&1 | FileCheck %s --check-prefix=FIXED
#ifdef OPTIONAL_OPERAND
def BadOp : TestEmitter<"`test` `(` operands `)`"> {
  let arguments = (ins Optional<IntType>:$input);
  let firrtlHasCustomParser = 1;
}
#endif

// RUN: not circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s -DDUPLICATE 2>&1 | FileCheck %s --check-prefix=DUP
#ifdef DUPLICATE
def BadOp : TestEmitter<"`add` `(` operands `)`"> {
  let firrtlHasCustomParser = 1;
}
// DUP: error: duplicate FIRRTL primitive spelling 'add'
#endif
