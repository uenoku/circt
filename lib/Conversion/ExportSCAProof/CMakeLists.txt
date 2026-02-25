add_circt_translation_library(CIRCTExportSCAProof
  ExportSCAProof.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDatapath
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
