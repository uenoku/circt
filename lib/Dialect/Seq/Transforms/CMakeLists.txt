add_circt_dialect_library(CIRCTSeqTransforms
  CheckDomain.cpp
  ExternalizeClockGate.cpp
  HWMemSimImpl.cpp
  RegOfVecToMem.cpp
  LowerSeqCompRegCE.cpp
  LowerSeqHLMem.cpp
  LowerSeqFIFO.cpp
  LowerSeqShiftReg.cpp

  DEPENDS
  CIRCTSeqTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTEmit
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTSV
  CIRCTVerif
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
