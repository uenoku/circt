diff --git a/include/circt/Dialect/Synth/Transforms/CutRewriter.h b/include/circt/Dialect/Synth/Transforms/CutRewriter.h
index cde8e7f13..87f2ecd81 100644
--- a/include/circt/Dialect/Synth/Transforms/CutRewriter.h
+++ b/include/circt/Dialect/Synth/Transforms/CutRewriter.h
@@ -38,7 +38,7 @@ using DelayType = int64_t;
 /// Maximum number of inputs supported for truth table generation.
 /// This limit prevents excessive memory usage as truth table size grows
 /// exponentially with the number of inputs (2^n entries).
-static constexpr unsigned maxTruthTableInputs = 16;
+static constexpr unsigned maxTruthTableInputs = 32;
 
 // This is a helper function to sort operations topologically in a logic
 // network. This is necessary for cut rewriting to ensure that operations are
diff --git a/include/circt/Support/TruthTable.h b/include/circt/Support/TruthTable.h
index 5f0b3dbc7..1d7f914d8 100644
--- a/include/circt/Support/TruthTable.h
+++ b/include/circt/Support/TruthTable.h
@@ -215,9 +215,9 @@ computeCofactors(const llvm::APInt &f, unsigned numVars, unsigned varIndex);
 /// Assumes number of variables is less than 64.
 struct Cube {
   /// Bitmask indicating which variables appear in this cube
-  uint64_t mask = 0;
+  uint32_t mask = 0;
   /// Bitmask indicating which variables are negated
-  uint64_t inverted = 0;
+  uint32_t inverted = 0;
 
   Cube() = default;
 
diff --git a/lib/Dialect/Synth/Transforms/CutRewriter.cpp b/lib/Dialect/Synth/Transforms/CutRewriter.cpp
index 1befa19c6..68c0975f4 100644
--- a/lib/Dialect/Synth/Transforms/CutRewriter.cpp
+++ b/lib/Dialect/Synth/Transforms/CutRewriter.cpp
@@ -852,8 +852,8 @@ void CutEnumerator::dump() const {
         llvm::outs() << getTestVariableName(input, opCounter);
       });
       auto &pattern = cut.getMatchedPattern();
-      llvm::outs() << "}"
-                   << "@t" << cut.getTruthTable().table.getZExtValue() << "d";
+      llvm::outs() << "}" << "@t" << cut.getTruthTable().table.getZExtValue()
+                   << "d";
       if (pattern) {
         llvm::outs() << *std::max_element(pattern->getArrivalTimes().begin(),
                                           pattern->getArrivalTimes().end());
@@ -884,8 +884,7 @@ LogicalResult CutRewriter::run(Operation *topOp) {
 
   // Validate maxCutInputSize doesn't exceed the truth table limit.
   if (options.maxCutInputSize > maxTruthTableInputs) {
-    return mlir::emitError(topOp->getLoc(),
-                           "maxCutInputSize cannot exceed ")
+    return mlir::emitError(topOp->getLoc(), "maxCutInputSize cannot exceed ")
            << maxTruthTableInputs << " (maxTruthTableInputs), but got "
            << options.maxCutInputSize;
   }
diff --git a/lib/Support/TruthTable.cpp b/lib/Support/TruthTable.cpp
index 694749d4c..87141532e 100644
--- a/lib/Support/TruthTable.cpp
+++ b/lib/Support/TruthTable.cpp
@@ -212,7 +212,7 @@ NPNClass NPNClass::computeNPNCanonicalForm(const BinaryTruthTable &tt) {
   NPNClass canonical(tt);
   // Initialize permutation with identity
   canonical.inputPermutation = identityPermutation(tt.numInputs);
-  assert(tt.numInputs <= 8 && "Inputs are too large");
+  assert(tt.numInputs <= 32 && "Inputs are too large");
   // Try all possible tables and pick the lexicographically smallest.
   // FIXME: The time complexity is O(n! * 2^(n + m)) where n is the number
   // of inputs and m is the number of outputs. This is not scalable so
@@ -515,6 +515,7 @@ APInt isopImpl(const APInt &tt, const APInt &dc, unsigned numVars,
 SOPForm circt::extractISOP(const APInt &truthTable, unsigned numVars) {
   assert((1u << numVars) == truthTable.getBitWidth() &&
          "Truth table size must match 2^numVars");
+  assert(numVars < 64);
   SOPForm sop(numVars);
 
   if (numVars == 0 || truthTable.isZero())
