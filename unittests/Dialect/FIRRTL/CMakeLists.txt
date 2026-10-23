add_circt_unittest(CIRCTFIRRTLTests
  AttributesTest.cpp
  PassiveWiresTest.cpp
  PortsTest.cpp
  TypesTest.cpp
)

target_link_libraries(CIRCTFIRRTLTests
  PRIVATE
  CIRCTFIRRTL
  CIRCTFIRRTLTransforms
  MLIRParser
  MLIRPass
)
