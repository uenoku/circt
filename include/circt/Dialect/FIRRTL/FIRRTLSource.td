//===- FIRRTLSource.td - FIRRTL source syntax metadata -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLSOURCE_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLSOURCE_TD

// These describe .fir source syntax, not MLIR assembly format.
class FIRRTLSourceVersion;
class FIRRTLReleasedVersion<int major, int minor, int patch>
    : FIRRTLSourceVersion {
  int majorVersion = major;
  int minorVersion = minor;
  int patchVersion = patch;
}
def FIRRTLBaseVersion : FIRRTLReleasedVersion<0, 0, 0>;
def FIRRTLUnspecifiedVersion : FIRRTLSourceVersion;
// The next version is distinct from syntax missing from the current spec.
def FIRRTLNextVersion : FIRRTLSourceVersion;

// For existing syntax whose import and export gates differ. New primitives
// should normally use a single released or unspecified version.
class FIRRTLSplitVersion<FIRRTLSourceVersion importGate,
                         FIRRTLSourceVersion exportGate>
    : FIRRTLSourceVersion {
  FIRRTLSourceVersion importVersion = importGate;
  FIRRTLSourceVersion exportVersion = exportGate;
}

// Opt in to ordinary spelling(expression..., integer...) syntax. The op must
// have one fixed result and fixed operands followed by required I32Attr
// arguments. Ops must provide the narrow inferReturnType and builder signatures
// used by the FIRRTL source parser; C++ compilation checks these.
// The spelling is also used as the feature name in version diagnostics.
class FIRRTLPrimitive<string spelling> {
  string firrtlSpelling = spelling;
  FIRRTLSourceVersion firrtlVersion = FIRRTLBaseVersion;
}

// Export-only calls retain their handwritten import parsers. The styles
// preserve the existing pretty-printing behavior of each call.
class FIRRTLSourceCallStyle;
def FIRRTLPrimitiveCall : FIRRTLSourceCallStyle; // Box each operand.
def FIRRTLUnboxedCall : FIRRTLSourceCallStyle;  // Box the call, not operands.
def FIRRTLUnaryCall : FIRRTLSourceCallStyle;    // No additional boxes.

// Calls must have one fixed result, no regions or successors, and only
// operands (which may be variadic). Unary calls require one fixed operand.

class FIRRTLSourceCall<string spelling, FIRRTLSourceVersion version,
                       string feature, FIRRTLSourceCallStyle style> {
  string firrtlCallSpelling = spelling;
  FIRRTLSourceVersion firrtlCallVersion = version;
  string firrtlCallFeature = feature;
  FIRRTLSourceCallStyle firrtlCallStyle = style;
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLSOURCE_TD
