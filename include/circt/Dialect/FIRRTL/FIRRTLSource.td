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

// Opt in to ordinary spelling(expression..., integer...) syntax. The op must
// have one fixed result and fixed operands followed by required I32Attr
// arguments. It must provide the narrow inferReturnType and builder signatures
// used by the FIRRTL source parser; these are checked by C++ compilation.
// Import and export gates are independent to preserve existing behavior.
class FIRRTLPrimitive<string spelling> {
  string firrtlSpelling = spelling;
  FIRRTLSourceVersion firrtlImportVersion = FIRRTLBaseVersion;
  string firrtlImportFeature = "Base";
  FIRRTLSourceVersion firrtlExportVersion = FIRRTLBaseVersion;
  string firrtlExportFeature = "Base";
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLSOURCE_TD
