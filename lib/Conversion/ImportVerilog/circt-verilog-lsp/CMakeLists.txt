set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
  # slang uses exceptions; we intercept these in ImportVerilog
  add_compile_options(-fexceptions)
  add_compile_options(-frtti)
  # slang has some classes with virtual funcs but non-virtual destructor.
  add_compile_options(-Wno-non-virtual-dtor)
  # some other warnings we've seen
  add_compile_options(-Wno-c++98-compat-extra-semi)
  add_compile_options(-Wno-ctad-maybe-unsupported)
  add_compile_options(-Wno-cast-qual)
  # visitor switch statements cover all cases but have default
  add_compile_options(-Wno-covered-switch-default)

add_circt_library(CIRCTVerilogLspServerLib
  LSPServer.cpp
  VerilogServer.cpp
  Protocol.cpp
  CirctVerilogLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/circt/Tools/crct-verilog-lsp-server

  DEPENDS
  slang_slang

  LINK_LIBS PUBLIC
  # LLVMSupport
  MLIRLspServerSupportLib
  CIRCTImportVerilog
  CIRCTMooreToCore
  CIRCTMooreTransforms
  CIRCTLLHD
  PRIVATE
  slang_slang
)
