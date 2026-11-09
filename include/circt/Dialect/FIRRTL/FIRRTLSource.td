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

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLSOURCE_TD
