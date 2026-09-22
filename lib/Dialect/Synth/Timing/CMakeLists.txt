# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_circt_dialect_library(CIRCTSynthTiming
  TimingDesignBuilder.cpp
  TimingJSON.cpp
  TimingResult.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Synth

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  MLIRIR
)

if(CIRCT_ENABLE_OPENTIMER)
  add_subdirectory(OpenTimer)
endif()
