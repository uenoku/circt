##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## Ematch dialect library (moved from Tamagoyaki).
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTEmatch
  EmatchDialect.cpp
  EmatchTransforms.cpp
  EmatchPasses.cpp
  Utils/ClassOpUtils.cpp
  Utils/CongruenceEngine.cpp
  Utils/GraphScope.cpp
  Utils/HashConsPatternRewriter.cpp
  Utils/MutableScopedHashTable.cpp
  Utils/ScopeRepIndex.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Ematch
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Equivalence

  DEPENDS
  CIRCTEmatchExtraIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTEquivalence
  MLIRIR
  MLIRPDLInterpDialect
  MLIRPass
  MLIRTransforms
)
