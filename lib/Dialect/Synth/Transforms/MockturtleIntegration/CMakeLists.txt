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
)

endif() # CIRCT_MOCKTURTLE_INTEGRATION_ENABLED
