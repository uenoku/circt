#===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//

# MockturtleIntegration library - only built when integration is enabled
if(CIRCT_MOCKTURTLE_INTEGRATION_ENABLED)

# Create an object library to avoid export issues
add_library(CIRCTSynthMockturtleIntegration OBJECT
  MockturtleAdapter.cpp
)

# Set properties
set_target_properties(CIRCTSynthMockturtleIntegration PROPERTIES
  FOLDER "circt libraries"
)

# Set up include directories
target_include_directories(CIRCTSynthMockturtleIntegration PRIVATE
  ${CIRCT_MAIN_INCLUDE_DIR}
  ${CIRCT_INCLUDE_DIR}
)

# Link against dependencies
target_link_libraries(CIRCTSynthMockturtleIntegration PUBLIC
  CIRCTHW
  CIRCTSynthAnalysis
  CIRCTSupport
  MLIRIR
)

# Link against mockturtle (includes kitty) and kitty interface for clean separation
target_link_libraries(CIRCTSynthMockturtleIntegration PRIVATE 
  mockturtle::mockturtle
  LLVMSupport
)

endif() # CIRCT_MOCKTURTLE_INTEGRATION_ENABLED
