add_circt_dialect_library(CIRCTAIGAnalysis
  LongestPathAnalysis.cpp
  PrintLongestPathAnalysis.cpp

  DEPENDS
  CIRCTAIGPassesIncGen

  LINK_LIBS PUBLIC
  MLIRSupport
  CIRCTAIG
  CIRCTDatapath
  CIRCTComb
  CIRCTHW
  CIRCTSeq
)
