include "circt/Dialect/FIRRTL/FIRRTLStatements.td"

// RUN: not circt-tblgen --firrtl-source-statements %TABLEGEN_INC% %s -DNOT_OP 2>&1 | FileCheck %s --check-prefix=NOT_OP
#ifdef NOT_OP
def NotAnOp : FIRRTLSourceStatement<"example", "parseExample">;
// NOT_OP: error: FIRRTLSourceStatement must be an operation
#endif

// RUN: not circt-tblgen --firrtl-source-statements %TABLEGEN_INC% %s -DBAD_NAME 2>&1 | FileCheck %s --check-prefix=NAME
#ifdef BAD_NAME
def BadOp : FIRRTLOp<"bad">, FIRRTLSourceStatement<"bad-name", "parseBad">;
// NAME: error: statement spelling and parser must be token identifiers
#endif

// RUN: not circt-tblgen --firrtl-source-statements %TABLEGEN_INC% %s -DBAD_PARSER 2>&1 | FileCheck %s --check-prefix=NAME
#ifdef BAD_PARSER
def BadOp : FIRRTLOp<"bad">, FIRRTLSourceStatement<"bad", "parseBad()">;
#endif

// RUN: not circt-tblgen --firrtl-source-statements %TABLEGEN_INC% %s -DDUPLICATE 2>&1 | FileCheck %s --check-prefix=DUP
#ifdef DUPLICATE
def BadOp : FIRRTLOp<"bad">, FIRRTLSourceStatement<"skip", "parseBad">;
// DUP: error: duplicate FIRRTL statement token 'kw_skip'
#endif
