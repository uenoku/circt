// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

def SpacedFieldOp : FIRRTLOp<"spaced.field"> {
  let arguments = (ins BundleType:$base, I32Attr:$field);
  let results = (outs FIRRTLBaseType:$result);
  let firrtlFormat = "$base `.` fieldName ( $field )";
}

def VersionedIndexOp : FIRRTLOp<"versioned.index"> {
  let arguments = (ins IntType:$base, I32Attr:$offset);
  let results = (outs IntType:$result);
  let firrtlFormat = "$base `[` $offset `]`";
  let firrtlVersion = FIRRTLReleasedVersion<4, 0, 0>;
}

// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_TOKENS
// CHECK-NOT: SubfieldOp
// CHECK-NOT: SubindexOp
// CHECK-NOT: SubaccessOp
// CHECK-NOT: SpacedFieldOp
// CHECK-NOT: VersionedIndexOp
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// CHECK: void emitExpression(OpenSubfieldOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK-NEXT: ps << "." << legalize(firrtl::type_cast<OpenSubfieldOp::InputType>(op.getInput().getType()).getElementNameAttr(op.getFieldIndex()));
// CHECK-NEXT: }
// CHECK: void emitExpression(OpenSubindexOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK-NEXT: ps << "[";
// CHECK-NEXT: ps.addAsString(op.getIndex());
// CHECK-NEXT: ps << "]";
// CHECK-NEXT: }
// CHECK: void emitExpression(SpacedFieldOp op) {
// CHECK-NEXT: emitExpression(op.getBase());
// CHECK-NEXT: ps << "." << legalize(firrtl::type_cast<SpacedFieldOp::InputType>(op.getBase().getType()).getElementNameAttr(op.getField()));
// CHECK-NEXT: }
// CHECK: void emitExpression(SubaccessOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK-NEXT: ps << "[";
// CHECK-NEXT: emitExpression(op.getIndex());
// CHECK-NEXT: ps << "]";
// CHECK-NEXT: }
// CHECK: void emitExpression(SubfieldOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK-NEXT: ps << "." << legalize(firrtl::type_cast<SubfieldOp::InputType>(op.getInput().getType()).getElementNameAttr(op.getFieldIndex()));
// CHECK-NEXT: }
// CHECK: void emitExpression(SubindexOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK-NEXT: ps << "[";
// CHECK-NEXT: ps.addAsString(op.getIndex());
// CHECK-NEXT: ps << "]";
// CHECK-NEXT: }
// CHECK: void emitExpression(VersionedIndexOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "versioned.index"))) return;
// CHECK-NEXT: emitExpression(op.getBase());
// CHECK-NEXT: ps << "[";
// CHECK-NEXT: ps.addAsString(op.getOffset());
// CHECK-NEXT: ps << "]";
// CHECK-NEXT: }
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: OpenSubfieldOp,
// CHECK: OpenSubindexOp,
// CHECK: SpacedFieldOp,
// CHECK: SubaccessOp,
// CHECK: SubfieldOp,
// CHECK: SubindexOp,
// CHECK: VersionedIndexOp,
