add_circt_public_c_api_library(CIRCTCAPISynth
  SynthesisPipeline.cpp

  LINK_LIBS PUBLIC
  CIRCTSynth
  CIRCTSynthTransforms
  MLIRCAPIIR
)
