add_circt_library(CIRCTLECTransforms
  ConstructLEC.cpp
  VerifyNativeSCA.cpp

  DEPENDS
  CIRCTLECTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDatapath
  CIRCTHW
  CIRCTVerif

  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRTransformUtils
)
