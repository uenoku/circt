add_circt_unittest(CIRCTSynthTests
  LongestPathTest.cpp
  TimingJSONTest.cpp
)

target_link_libraries(CIRCTSynthTests
  PRIVATE
  CIRCTHW
  CIRCTSeq
  CIRCTSynth
  CIRCTSynthAnalysis
  CIRCTSynthTiming
  CIRCTHWTransforms
  CIRCTCombToSynth
  MLIRTransforms
  MLIRIR
  MLIRParser
)
