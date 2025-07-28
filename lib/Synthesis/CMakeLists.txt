add_circt_library(CIRCTSynthesis
  SynthesisPipeline.cpp

  LINK_LIBS PUBLIC
  CIRCTAIGTransforms
  CIRCTCombToAIG
  CIRCTHWTransforms
  CIRCTSupport
  CIRCTCombToDatapath
  CIRCTDatapathToComb

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
