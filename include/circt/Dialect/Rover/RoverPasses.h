//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for Rover e-graph datapath passes.
// Moved from tamagoyaki-case-studies/rover-mlir; namespace kept as `rover`
// to minimize the diff (follow-up: move to `circt::rover`).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ROVER_ROVERPASSES_H
#define CIRCT_DIALECT_ROVER_ROVERPASSES_H

#include "mlir/Pass/Pass.h"

namespace rover {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Rover/RoverPasses.h.inc"

} // namespace rover

#endif // CIRCT_DIALECT_ROVER_ROVERPASSES_H
