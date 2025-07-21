add_circt_public_c_api_library(CIRCTCAPISynthesis
  Synthesis.cpp

  LINK_LIBS PUBLIC
  CIRCTSynthesisTransforms
  MLIRCAPIIR
)
