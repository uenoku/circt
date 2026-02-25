add_circt_conversion_library(CIRCTDatapathToComb
  DatapathToComb.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTSynthAnalysis
  CIRCTComb
  CIRCTDatapath
  CIRCTHW
  CIRCTVerif
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
