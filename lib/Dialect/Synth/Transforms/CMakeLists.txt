##===----------------------------------------------------------------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTSynthTransforms
  AIGERRunner.cpp
  CutRewriter.cpp
  FunctionalReduction.cpp
  GenericLUTMapper.cpp
  LowerVariadic.cpp
  LowerWordToBits.cpp
  MaximumAndCover.cpp
  MIGAlgebraicRewriting.cpp
  SOPBalancing.cpp
  StructuralHash.cpp
  SynthesisPipeline.cpp
  SynthPreprocess.cpp
  TechMapper.cpp
  TestPriorityCuts.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Synth

  DEPENDS
  CIRCTSynthTransformsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  CIRCTSV
  CIRCTSynthAnalysis
  CIRCTCombToSynth
  CIRCTCombToDatapath
  CIRCTCombTransforms
  CIRCTDatapathToComb
  CIRCTDatapathTransforms
  CIRCTExportAIGER
  CIRCTImportAIGER
  CIRCTHWTransforms
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
)

if(CIRCT_MOCKTURTLE_ENABLED)
add_subdirectory(MockturtleIntegration)
target_link_libraries(CIRCTSynthTransforms PRIVATE
  CIRCTSynthMockturtleIntegration
) 
endif()

if(CIRCT_CADICAL_ENABLED)
  target_link_libraries(CIRCTSynthTransforms PRIVATE CaDiCaL)
endif()
