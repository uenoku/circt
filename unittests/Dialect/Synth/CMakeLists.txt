if(CIRCT_ENABLE_OPENTIMER)
  set(CIRCT_SYNTH_OPTIONAL_TESTS OpenTimerOracleTest.cpp)
endif()
add_circt_unittest(CIRCTSynthTests
  LongestPathTest.cpp
  TimingDesignBuilderTest.cpp
  TimingJSONTest.cpp
  ${CIRCT_SYNTH_OPTIONAL_TESTS}
)
if(CIRCT_ENABLE_OPENTIMER)
  target_link_libraries(CIRCTSynthTests PRIVATE CIRCTSynthOpenTimer)
  target_compile_definitions(CIRCTSynthTests PRIVATE
    CIRCT_TINY_LIB_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../test/Dialect/Synth/Inputs/tiny-timing.lib")
  # The test includes OpenTimer headers through the oracle interface.
  target_compile_options(CIRCTSynthTests PRIVATE -fexceptions -frtti)
endif()

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
