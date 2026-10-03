##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## Equivalence dialect library (moved from Tamagoyaki).
## NOTE: HiGHS-backed ILP selection is compiled out; EquivalenceSelectIlp.cpp
## reports a diagnostic at runtime without TAMAGOYAKI_HIGHS_ENABLED.
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTEquivalence
  EquivalenceDialect.cpp
  EquivalenceTransforms.cpp
  EquivalencePasses.cpp
  EquivalenceSelectIlp.cpp
  EquivalenceTiming.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Equivalence

  DEPENDS
  CIRCTEquivalenceExtraIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)
