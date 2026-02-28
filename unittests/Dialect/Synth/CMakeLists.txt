add_circt_unittest(CIRCTSynthTests
  CutRewriterTest.cpp
  LongestPathTest.cpp
)

target_link_libraries(CIRCTSynthTests
  PRIVATE
  CIRCTHW
  CIRCTSeq
  CIRCTSynth
  CIRCTSynthAnalysis
  CIRCTHWTransforms
  CIRCTCombToSynth
  MLIRTransforms
  MLIRIR
  MLIRParser
)
