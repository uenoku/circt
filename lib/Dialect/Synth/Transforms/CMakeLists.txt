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
  ExactSynthesis.cpp
  FunctionalReduction.cpp
  GenericLUTMapper.cpp
  GenSupergates.cpp
  LowerVariadic.cpp
  LowerWordToBits.cpp
  MaximumAndCover.cpp
  SOPBalancing.cpp
  DeriveMappingCost.cpp
  DedupInstances.cpp
  SortTopologically.cpp
  TieCells.cpp
  EmitSDC.cpp
  DeriveProbes.cpp
  BufferHighFanout.cpp
  ExtractBundles.cpp
  MapBundles.cpp
  RestructureMuxTrees.cpp
  SeqMemMapper.cpp
  StripLibToExtern.cpp
  SeqTechMapper.cpp
  StructuralHash.cpp
  SynthesisPipeline.cpp
  TechMapper.cpp
  TestPriorityCuts.cpp
  WidePatternTables.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Synth

  DEPENDS
  CIRCTSynthTransformsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTSV
  CIRCTSeq
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
