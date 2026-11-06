// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --implicit-check-not=CatPrimOp --implicit-check-not=TagExtractOp --implicit-check-not=SizeOfIntrinsicOp --implicit-check-not=ElementwiseAndPrimOp --implicit-check-not=ElementwiseOrPrimOp --implicit-check-not=ElementwiseXorPrimOp --implicit-check-not=Mux2CellIntrinsicOp --implicit-check-not=Mux4CellIntrinsicOp

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_TOKENS
// CHECK: TOK_LPKEYWORD_PRIM(add, AddPrimOp, 2, 0, FIRVersion(0, 0, 0), "add")
// CHECK: TOK_LPKEYWORD_PRIM(amount_source, AlternateAmountOp, 1, 1, FIRVersion(0, 0, 0), "amount_source")
// CHECK: TOK_LPKEYWORD_PRIM(binary_source, AlternateBinaryOp, 2, 0, FIRVersion(0, 0, 0), "binary_source")
// CHECK: TOK_LPKEYWORD_PRIM(bool_source, AlternateBoolOp, 2, 0, missingSpecFIRVersion, "bool_source")
// CHECK: TOK_LPKEYWORD_PRIM(integer_source, AlternateIntegerOp, 2, 0, FIRVersion(4, 0, 0), "integer_source")
// CHECK: TOK_LPKEYWORD_PRIM(integer_unary_source, AlternateIntegerUnaryOp, 1, 0, FIRVersion(0, 0, 0), "integer_unary_source")
// CHECK: TOK_LPKEYWORD_PRIM(shift_source, AlternateShiftOp, 1, 1, FIRVersion(0, 0, 0), "shift_source")
// CHECK: TOK_LPKEYWORD_PRIM(source_spelling, AlternateSpellingOp, 2, 0, FIRVersion(0, 0, 0), "source_spelling")
// CHECK: TOK_LPKEYWORD_PRIM(unary_source, AlternateUnaryOp, 1, 0, FIRVersion(0, 0, 0), "unary_source")
// CHECK: TOK_LPKEYWORD_PRIM(asReset, AsResetPrimOp, 1, 0, missingSpecFIRVersion, "asReset")
// CHECK: TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0), "bits")
// CHECK: TOK_LPKEYWORD_PRIM(bool_and, BoolAndOp, 2, 0, missingSpecFIRVersion, "bool_and")
// CHECK: TOK_LPKEYWORD_PRIM(integer_add, IntegerAddOp, 2, 0, FIRVersion(4, 0, 0), "integer_add")
// CHECK: TOK_LPKEYWORD_PRIM(integer_and, IntegerAndOp, 2, 0, missingSpecFIRVersion, "integer_and")
// CHECK: TOK_LPKEYWORD_PRIM(integer_mul, IntegerMulOp, 2, 0, FIRVersion(4, 0, 0), "integer_mul")
// CHECK: TOK_LPKEYWORD_PRIM(integer_not, IntegerNotOp, 1, 0, missingSpecFIRVersion, "integer_not")
// CHECK: TOK_LPKEYWORD_PRIM(integer_shl, IntegerShlOp, 2, 0, FIRVersion(4, 0, 0), "integer_shl")
// CHECK: TOK_LPKEYWORD_PRIM(integer_shr, IntegerShrOp, 2, 0, FIRVersion(4, 0, 0), "integer_shr")
// CHECK: TOK_LPKEYWORD_PRIM(shl, ShlPrimOp, 1, 1, FIRVersion(0, 0, 0), "shl")
// CHECK: TOK_LPKEYWORD_PRIM(shr, ShrPrimOp, 1, 1, FIRVersion(0, 0, 0), "shr")
// CHECK: TOK_LPKEYWORD_PRIM(example, VersionedOp, 1, 1, missingSpecFIRVersion, "example")
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// CHECK: void emitExpression(AddPrimOp op) {
// CHECK-NEXT: emitPrimExpr("add", op);
// CHECK: void emitExpression(AlternateAmountOp op) {
// CHECK-NEXT: emitPrimExpr("amount_source", op, op.getAmount());
// CHECK: void emitExpression(AlternateBinaryOp op) {
// CHECK-NEXT: emitPrimExpr("binary_source", op);
// CHECK: void emitExpression(AlternateBoolOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "bool_source"))) return;
// CHECK-NEXT: emitPrimExpr("bool_source", op);
// CHECK: void emitExpression(AlternateIntegerOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_source"))) return;
// CHECK-NEXT: emitPrimExpr("integer_source", op);
// CHECK: void emitExpression(AlternateIntegerUnaryOp op) {
// CHECK-NEXT: emitPrimExpr("integer_unary_source", op);
// CHECK: void emitExpression(AlternateShiftOp op) {
// CHECK-NEXT: emitPrimExpr("shift_source", op, op.getAmount());
// CHECK: void emitExpression(AlternateSpellingOp op) {
// CHECK-NEXT: emitPrimExpr("source_spelling", op);
// CHECK: void emitExpression(AlternateUnaryOp op) {
// CHECK-NEXT: emitPrimExpr("unary_source", op);
// CHECK: void emitExpression(AsResetPrimOp op) {
// CHECK-NEXT: emitPrimExpr("asReset", op);
// CHECK: void emitExpression(BitsPrimOp op) {
// CHECK-NEXT: emitPrimExpr("bits", op, {op.getHi(), op.getLo()});
// CHECK: void emitExpression(BoolAndOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "bool_and"))) return;
// CHECK-NEXT: emitPrimExpr("bool_and", op);
// CHECK: void emitExpression(HeadPrimOp op) {
// CHECK-NEXT: emitPrimExpr("head", op, op.getAmount());
// CHECK: void emitExpression(IntegerAddOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_add"))) return;
// CHECK-NEXT: emitPrimExpr("integer_add", op);
// CHECK: void emitExpression(IntegerAndOp op) {
// CHECK-NEXT: if (failed(requireVersion(missingSpecFIRVersion, op, "integer_and"))) return;
// CHECK-NEXT: emitPrimExpr("integer_and", op);
// CHECK: void emitExpression(IntegerMulOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_mul"))) return;
// CHECK-NEXT: emitPrimExpr("integer_mul", op);
// CHECK: void emitExpression(IntegerNotOp op) {
// CHECK-NEXT: if (failed(requireVersion(missingSpecFIRVersion, op, "integer_not"))) return;
// CHECK-NEXT: emitPrimExpr("integer_not", op);
// CHECK: void emitExpression(IntegerShlOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_shl"))) return;
// CHECK-NEXT: emitPrimExpr("integer_shl", op);
// CHECK: void emitExpression(IntegerShrOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_shr"))) return;
// CHECK-NEXT: emitPrimExpr("integer_shr", op);
// CHECK: void emitExpression(PadPrimOp op) {
// CHECK-NEXT: emitPrimExpr("pad", op, op.getAmount());
// CHECK: void emitExpression(ShlPrimOp op) {
// CHECK-NEXT: emitPrimExpr("shl", op, op.getAmount());
// CHECK: void emitExpression(TailPrimOp op) {
// CHECK-NEXT: emitPrimExpr("tail", op, op.getAmount());
// CHECK: void emitExpression(VersionedOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "example"))) return;
// CHECK-NEXT: emitPrimExpr("example", op, op.getExampleAmount());
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: AddPrimOp,
// CHECK: AlternateAmountOp,
// CHECK: AlternateBinaryOp,
// CHECK: AlternateBoolOp,
// CHECK: AlternateIntegerOp,
// CHECK: AlternateIntegerUnaryOp,
// CHECK: AlternateShiftOp,
// CHECK: AlternateSpellingOp,
// CHECK: AlternateUnaryOp,
// CHECK: BitsPrimOp,
// CHECK: IntegerAddOp,
// CHECK: IntegerMulOp,
// CHECK: IntegerShlOp,
// CHECK: IntegerShrOp,
// CHECK: ShlPrimOp,
// CHECK: ShrPrimOp,
// CHECK: VersionedOp,

// The MLIR mnemonic can differ from the FIRRTL source spelling. Real ops such
// as AddPrimOp exercise the default source spelling instead.
def AlternateAmountOp : AmountPrimOp<"amount.mlir", UIntType, "amount_source">;
def AlternateBinaryOp : BinaryPrimOp<"binary.mlir", IntType, UIntType, IntType,
                                      [], "binary_source">;
def AlternateBoolOp : BoolBinaryPrimOp<"bool.test", [], "bool_source">;
def AlternateIntegerOp : IntegerBinaryPrimOp<"integer.test", [], "integer_source">;
def AlternateIntegerUnaryOp : IntegerUnaryPrimOp<"integer.unary", "integer_unary_source">;
def AlternateShiftOp : ShiftPrimOp<"shift.mlir", "shift_source">;
def AlternateSpellingOp : IntBinaryPrimOp<"mlir_mnemonic", IntType, [],
                                          "source_spelling">;
def AlternateUnaryOp : UnaryPrimOp<"unary.mlir", IntType, UIntType, [],
                                   "unary_source">;

def VersionedOp : FIRRTLOp<"example">, FIRRTLPrimitive<"example"> {
  let arguments = (ins IntType:$input, I32Attr:$exampleAmount);
  let results = (outs IntType:$result);
  let firrtlVersion = FIRRTLSplitVersion<FIRRTLUnspecifiedVersion,
                                         FIRRTLReleasedVersion<6, 0, 0>>;
}
