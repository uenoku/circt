set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(circt-synth circt-synth.cpp DatapathRetry.cpp)
target_link_libraries(circt-synth
  PRIVATE
  CIRCTSynth
  CIRCTSynthToComb
  CIRCTSynthAnalysis
  CIRCTSynthTiming
  CIRCTComb
  CIRCTCombToDatapath
  CIRCTCombTransforms
  CIRCTDatapath
  CIRCTDatapathToComb
  CIRCTDebug
  CIRCTEmit
  CIRCTHW
  CIRCTImportAIGER
  CIRCTImportLiberty
  CIRCTLTL
  CIRCTOM
  CIRCTSeq
  CIRCTSim
  CIRCTSV
  CIRCTSynthTransforms
  CIRCTTransforms
  CIRCTVerif
  MLIRBytecodeWriter
  MLIRParser
  MLIRIR
  MLIRTransformDialect
  MLIRTransformDialectTransforms
)
if(CIRCT_ENABLE_OPENTIMER)
  target_link_libraries(circt-synth PRIVATE CIRCTSynthOpenTimer)
  target_compile_definitions(circt-synth PRIVATE CIRCT_OPENTIMER_ENABLED=1)
endif()

llvm_update_compile_flags(circt-synth)
mlir_check_all_link_libraries(circt-synth)
