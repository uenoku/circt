add_circt_unittest(CIRCTSynthTests
  LongestPathTest.cpp
  TimingAnalysisTest.cpp
)

target_link_libraries(CIRCTSynthTests
  PRIVATE
  CIRCTHW
  CIRCTSeq
  CIRCTSynth
  CIRCTSynthAnalysis
  CIRCTTimingAnalysis
  CIRCTHWTransforms
  CIRCTCombToSynth
  MLIRTransforms
  MLIRIR
  MLIRParser
)
