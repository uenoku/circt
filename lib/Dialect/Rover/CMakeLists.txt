##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## Rover e-graph datapath passes (moved from tamagoyaki-case-studies).
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTRover
  RoverPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Rover

  DEPENDS
  CIRCTRoverIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTDatapath
  CIRCTEquivalence
  CIRCTEmatch
  MLIRIR
  MLIRFuncDialect
  MLIRPDLInterpDialect
  MLIRParser
  MLIRPass
  MLIRTransforms
)
