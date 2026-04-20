add_circt_dialect_library(CIRCTOMTransforms
  ElaborationTransform.cpp
  FreezePaths.cpp
  LinkModules.cpp
  StripOM.cpp
  TestEvaluator.cpp
  VerifyObjectFields.cpp

  DEPENDS
  CIRCTOMTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTOM
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
