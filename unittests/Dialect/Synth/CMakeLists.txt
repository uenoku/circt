add_circt_unittest(CIRCTSynthTests
  LongestPathTest.cpp
  TimingV2/FlatTimingTest.cpp
)

target_link_libraries(CIRCTSynthTests
  PRIVATE
  CIRCTDatapath
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
