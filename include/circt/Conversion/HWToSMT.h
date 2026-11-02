//===- HWToSMT.h - HW to SMT dialect conversion -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWTOSMT_H
#define CIRCT_CONVERSION_HWTOSMT_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

#define GEN_PASS_DECL_CONVERTHWTOSMT
#include "circt/Conversion/Passes.h.inc"

/// Get the HW to SMT conversion patterns.
void populateHWToSMTConversionPatterns(TypeConverter &converter,
                                       RewritePatternSet &patterns,
                                       bool forSMTLIBExport,
                                       bool fourState = false);

/// Get the HW to SMT type conversions.
/// When `fourState` is true, each N-bit integer is encoded as a 2*N-bit
/// bit-vector (low N bits = value, high N bits = poison/X-mask) to model
/// 4-state (0,1,X,Z) logic: (poison,value) = 00->0, 01->1, 10->X, 11->Z.
void populateHWToSMTTypeConverter(TypeConverter &converter,
                                  bool fourState = false);

} // namespace circt

#endif // CIRCT_CONVERSION_HWTOSMT_H
