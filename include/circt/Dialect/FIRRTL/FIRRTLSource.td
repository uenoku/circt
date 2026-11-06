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
// arguments. Importable ops must provide the narrow inferReturnType and builder
// signatures used by the FIRRTL source parser; C++ compilation checks these.
// The spelling is also used as the feature name in version diagnostics.
class FIRRTLPrimitive<string spelling> {
  string firrtlSpelling = spelling;
  bit firrtlHasImport = 1;
  bit firrtlHasExport = 1;
  FIRRTLSourceVersion firrtlVersion = FIRRTLBaseVersion;
}

// Register a statement's existing parser and emitter handlers. These handlers
// retain responsibility for source syntax, version checks, and diagnostics.
// `leftParen` selects an existing lp_ token instead of a kw_ token. This does
// not define a lexer keyword: the token must already exist in FIRTokenKinds.def.
class FIRRTLSourceStatement<string spelling, string parser, bit leftParen = 0> {
  string firrtlStatementSpelling = spelling;
  string firrtlStatementParser = parser;
  bit firrtlStatementLeftParen = leftParen;
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLSOURCE_TD
