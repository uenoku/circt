add_circt_tool(circt-sta circt-sta.cpp)
target_link_libraries(circt-sta
  PRIVATE
  CIRCTComb
  CIRCTHW
  CIRCTSynth
  CIRCTSynthAnalysis
  CIRCTSynthTransforms
  MLIRIR
  MLIRParser
  LLVMSupport
)

llvm_update_compile_flags(circt-sta)
mlir_check_all_link_libraries(circt-sta)
