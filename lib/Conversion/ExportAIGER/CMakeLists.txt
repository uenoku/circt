add_circt_translation_library(CIRCTExportAIGER
  ExportAIGER.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTAIG
  CIRCTHW
  CIRCTSupport
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
