add_circt_library(circt-tcl SHARED
  SHARED
  TclSupport.cpp
  circt_tcl.cpp
)
get_property(CIRCT_TCL_DIALECT_LIBS GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
target_link_libraries(circt-tcl
  PUBLIC
  CIRCTCAPIComb
  CIRCTCAPIESI
  CIRCTCAPIMSFT
  CIRCTCAPIHW
  CIRCTCAPISeq
  CIRCTCAPISV
  CIRCTCAPIFSM
  ${CIRCT_TCL_DIALECT_LIBS}
)
target_compile_definitions(circt-tcl PUBLIC -DUSE_TCL_STUBS)
target_link_libraries(circt-tcl PUBLIC ${TCL_STUB_LIBRARY})
target_include_directories(circt-tcl PUBLIC ${TCL_INCLUDE_PATH})

file(GENERATE
  OUTPUT "$<TARGET_FILE_DIR:circt-tcl>/pkgIndex.tcl"
  CONTENT
"package ifneeded Circt 1.0 [list load [file join $dir \"$<TARGET_FILE_NAME:circt-tcl>\"] Circt]
")
install(FILES "$<TARGET_FILE_DIR:circt-tcl>/pkgIndex.tcl"
  DESTINATION "lib${LLVM_LIBDIR_SUFFIX}"
  COMPONENT circt-tcl)
