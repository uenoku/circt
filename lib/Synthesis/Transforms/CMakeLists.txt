add_circt_library(CIRCTSynthesisTransforms
  GenericLUTMapper.cpp
  SynthesisPipeline.cpp
  TechMapper.cpp

  DEPENDS
  CIRCTSynthesisTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTAIGTransforms
  CIRCTCombToAIG
  CIRCTHWTransforms
  CIRCTSupport
  CIRCTSynthesis

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
