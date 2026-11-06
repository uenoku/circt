// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --implicit-check-not=CatPrimOp --implicit-check-not=SizeOfIntrinsicOp --implicit-check-not=ElementwiseAndPrimOp
// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --check-prefix=FILTER

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_TOKENS
// CHECK: TOK_LPKEYWORD_PRIM(add, AddPrimOp, 2, 0, FIRVersion(0, 0, 0), "add")
// CHECK: TOK_LPKEYWORD_PRIM(asReset, AsResetPrimOp, 1, 0, missingSpecFIRVersion, "asReset")
// CHECK: TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0), "bits")
// CHECK: TOK_LPKEYWORD_PRIM(bool_and, BoolAndOp, 2, 0, missingSpecFIRVersion, "bool_and")
// CHECK: TOK_LPKEYWORD_PRIM(import_only, ImportOnlyOp, 1, 0, FIRVersion(0, 0, 0), "import_only")
// CHECK: TOK_LPKEYWORD_PRIM(integer_add, IntegerAddOp, 2, 0, FIRVersion(4, 0, 0), "integer_add")
// CHECK: TOK_LPKEYWORD_PRIM(integer_and, IntegerAndOp, 2, 0, missingSpecFIRVersion, "integer_and")
// CHECK: TOK_LPKEYWORD_PRIM(integer_mul, IntegerMulOp, 2, 0, FIRVersion(4, 0, 0), "integer_mul")
// CHECK: TOK_LPKEYWORD_PRIM(integer_shl, IntegerShlOp, 2, 0, FIRVersion(4, 0, 0), "integer_shl")
// CHECK: TOK_LPKEYWORD_PRIM(integer_shr, IntegerShrOp, 2, 0, FIRVersion(4, 0, 0), "integer_shr")
// CHECK: TOK_LPKEYWORD_PRIM(shl, ShlPrimOp, 1, 1, FIRVersion(0, 0, 0), "shl")
// CHECK: TOK_LPKEYWORD_PRIM(shr, ShrPrimOp, 1, 1, FIRVersion(0, 0, 0), "shr")
// CHECK: TOK_LPKEYWORD_PRIM(tagExtract, TagExtractOp, 1, 0, missingSpecFIRVersion, "tagExtract")
// CHECK: TOK_LPKEYWORD_PRIM(example, VersionedOp, 1, 1, missingSpecFIRVersion, "example")
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// CHECK: void emitExpression(AddPrimOp op) {
// CHECK-NEXT: emitPrimExpr("add", op);
// CHECK: void emitExpression(AsResetPrimOp op) {
// CHECK-NEXT: emitPrimExpr("asReset", op);
// CHECK: void emitExpression(BitsPrimOp op) {
// CHECK-NEXT: emitPrimExpr("bits", op, {op.getHi(), op.getLo()});
// CHECK: void emitExpression(BoolAndOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "bool_and"))) return;
// CHECK-NEXT: emitPrimExpr("bool_and", op);
// CHECK: void emitExpression(ExportOnlyOp op) {
// CHECK-NEXT: emitPrimExpr("export_only", op);
// CHECK: void emitExpression(IntegerAddOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_add"))) return;
// CHECK-NEXT: emitPrimExpr("integer_add", op);
// CHECK: void emitExpression(IntegerAndOp op) {
// CHECK-NEXT: if (failed(requireVersion(missingSpecFIRVersion, op, "integer_and"))) return;
// CHECK-NEXT: emitPrimExpr("integer_and", op);
// CHECK: void emitExpression(IntegerMulOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_mul"))) return;
// CHECK-NEXT: emitPrimExpr("integer_mul", op);
// CHECK: void emitExpression(IntegerShlOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_shl"))) return;
// CHECK-NEXT: emitPrimExpr("integer_shl", op);
// CHECK: void emitExpression(IntegerShrOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_shr"))) return;
// CHECK-NEXT: emitPrimExpr("integer_shr", op);
// CHECK: void emitExpression(ShlPrimOp op) {
// CHECK-NEXT: emitPrimExpr("shl", op, op.getAmount());
// CHECK: void emitExpression(VersionedOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "example"))) return;
// CHECK-NEXT: emitPrimExpr("example", op, op.getExampleAmount());
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: AddPrimOp,
// CHECK: BitsPrimOp,
// CHECK: ExportOnlyOp,
// CHECK: IntegerAddOp,
// CHECK: IntegerMulOp,
// CHECK: IntegerShlOp,
// CHECK: IntegerShrOp,
// CHECK: ShlPrimOp,
// CHECK: ShrPrimOp,
// CHECK: VersionedOp,

// FILTER-LABEL: #ifdef GET_FIRRTL_PRIMITIVE_TOKENS
// FILTER-NOT: export_only
// FILTER: #endif
// FILTER-LABEL: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// FILTER-NOT: ImportOnlyOp
// FILTER-NOT: TagExtractOp
// FILTER: #endif
// FILTER-LABEL: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// FILTER-NOT: ImportOnlyOp
// FILTER-NOT: TagExtractOp
// FILTER: #endif

def ExportOnlyOp : FIRRTLOp<"export_only">, FIRRTLPrimitive<"export_only"> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let firrtlHasImport = 0;
}

def ImportOnlyOp : FIRRTLOp<"import_only">, FIRRTLPrimitive<"import_only"> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let firrtlHasExport = 0;
}

def VersionedOp : FIRRTLOp<"example">, FIRRTLPrimitive<"example"> {
  let arguments = (ins IntType:$input, I32Attr:$exampleAmount);
  let results = (outs IntType:$result);
  let firrtlVersion = FIRRTLSplitVersion<FIRRTLUnspecifiedVersion,
                                         FIRRTLReleasedVersion<6, 0, 0>>;
}
