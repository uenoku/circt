#===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//

# MockturtleIntegration library - only built when integration is enabled
if(CIRCT_MOCKTURTLE_INTEGRATION_ENABLED)

add_circt_library(CIRCTSynthMockturtleIntegration
  MockturtleAdapter.cpp
  Refactor.cpp

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSynthAnalysis
  CIRCTSupport
  MLIRIR
)

# Add mockturtle include directories directly (header-only library)
target_include_directories(CIRCTSynthMockturtleIntegration PRIVATE
  ${mockturtle_SOURCE_DIR}/include
  ${mockturtle_SOURCE_DIR}/lib/kitty
  ${mockturtle_SOURCE_DIR}/lib/parallel_hashmap
  ${mockturtle_SOURCE_DIR}/lib/fmt
  ${mockturtle_SOURCE_DIR}/lib/bill
  ${mockturtle_SOURCE_DIR}/lib/percy
  ${mockturtle_SOURCE_DIR}/lib/rang
  ${mockturtle_SOURCE_DIR}/lib/lorina
  ${mockturtle_SOURCE_DIR}/lib/json
  ${mockturtle_SOURCE_DIR}/lib/abcsat
)

# Define necessary compile definitions for mockturtle and ABC
target_compile_definitions(CIRCTSynthMockturtleIntegration PRIVATE
  FMT_HEADER_ONLY
  DISABLE_NAUTY
  LIN64
  ABC_NAMESPACE=pabc
  ABC_NO_USE_READLINE
)

endif() # CIRCT_MOCKTURTLE_INTEGRATION_ENABLED
