add_circt_translation_library(CIRCTExportFIRRTL
  FIREmitter.cpp

  DEPENDS
  CIRCTFIRRTLSourceIncGen

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRTranslateLib
)
