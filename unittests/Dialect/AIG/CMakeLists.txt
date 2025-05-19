add_circt_unittest(CIRCTAIGTests
  LongestPathTest.cpp
)

target_link_libraries(CIRCTAIGTests
  PRIVATE
  CIRCTAIG
  MLIRIR
)
