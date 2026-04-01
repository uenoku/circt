add_circt_tool(circt-synth-dbgen circt-synth-dbgen.cpp)
target_link_libraries(circt-synth-dbgen
  PRIVATE
  CIRCTHW
  CIRCTSynth
  CIRCTSynthTransforms
  MLIRBytecodeWriter
  MLIRIR
  LLVMSupport
)

llvm_update_compile_flags(circt-synth-dbgen)
mlir_check_all_link_libraries(circt-synth-dbgen)
