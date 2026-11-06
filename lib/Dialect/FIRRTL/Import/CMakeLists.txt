add_circt_translation_library(CIRCTImportFIRFile
  FIRAnnotations.cpp
  FIRLexer.cpp
  FIRParser.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTFIRRTLPrimitiveIncGen

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTFIRRTLTransforms
  MLIRTranslateLib
  )
