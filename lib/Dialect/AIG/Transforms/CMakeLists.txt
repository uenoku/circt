add_circt_dialect_library(CIRCTAIGTransforms
  GreedyCut.cpp
  LowerCutToLUT.cpp
  LowerVariadic.cpp

  DEPENDS
  CIRCTAIGTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
)
