add_circt_library(CIRCTImportLiberty
  ImportLiberty.cpp

  DEPENDS
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTranslateLib
)
