set(LLVM_TARGET_DEFINITIONS SynthPasses.td)

# Set up TableGen preprocessor definitions
set(TABLEGEN_DEFINES "")
if(CIRCT_MOCKTURTLE_ENABLED)
  list(APPEND TABLEGEN_DEFINES "-DCIRCT_MOCKTURTLE_ENABLED")
endif()

mlir_tablegen(SynthPasses.h.inc -gen-pass-decls ${TABLEGEN_DEFINES})
add_public_tablegen_target(CIRCTSynthTransformsIncGen)

# Generate Pass documentation.
add_circt_doc(SynthPasses SynthPasses -gen-pass-doc ${TABLEGEN_DEFINES})
