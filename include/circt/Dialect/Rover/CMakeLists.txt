##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## TableGen for Rover passes (moved from tamagoyaki-case-studies/rover-mlir).
## Passes only, no ops: rover-insert-graph / rover-saturate / rover-extract
## over comb/hw/datapath + equivalence/ematch e-graphs.
##
##===----------------------------------------------------------------------===//

set(LLVM_TARGET_DEFINITIONS RoverPasses.td)
mlir_tablegen(RoverPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTRoverIncGen)

add_circt_doc(RoverPasses RoverPasses -gen-pass-doc)
