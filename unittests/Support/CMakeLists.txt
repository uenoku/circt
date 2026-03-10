add_circt_unittest(CIRCTSupportTests
  FVIntTest.cpp
  JSONTest.cpp
  PrettyPrinterTest.cpp
  TruthTableTest.cpp
  SATSolverTest.cpp
)

target_link_libraries(CIRCTSupportTests
  PRIVATE
  CIRCTSupport
)
