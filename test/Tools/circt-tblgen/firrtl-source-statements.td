// RUN: circt-tblgen --firrtl-source-statements %TABLEGEN_INC% %s | FileCheck %s

include "circt/Dialect/FIRRTL/FIRRTLStatements.td"

// The metadata selects existing token kinds and handwritten parser methods.
// CHECK: #ifdef GET_FIRRTL_STATEMENT_PARSERS
// CHECK: case FIRToken::kw_attach: return parseAttach();
// CHECK: case FIRToken::lp_fflush: return parseFFlush();
// CHECK: case FIRToken::lp_fprintf: return parseFPrintf();
// CHECK: case FIRToken::lp_printf: return parsePrintf();
// CHECK: case FIRToken::kw_skip: return parseSkip();
// CHECK: case FIRToken::lp_stop: return parseStop();
// CHECK: #ifdef GET_FIRRTL_STATEMENT_DISPATCH
// CHECK: AttachOp,
// CHECK: FFlushOp,
// CHECK: FPrintFOp,
// CHECK: PrintFOp,
// CHECK: SkipOp,
// CHECK: StopOp,
