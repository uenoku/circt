add_circt_library(CIRCTLECTransforms
  ConstructLEC.cpp

  DEPENDS
  CIRCTLECTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTHW
  CIRCTVerif

  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRTransformUtils
)
