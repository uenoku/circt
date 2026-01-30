##===----------------------------------------------------------------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTTimingAnalysis
  ArrivalAnalysis.cpp
  DelayModel.cpp
  ObjectCollection.cpp
  PathEnumerator.cpp
  RequiredTimeAnalysis.cpp
  TimingAnalysis.cpp
  TimingGraph.cpp
  TimingReport.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Synth/Analysis/Timing

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  CIRCTSynth
  CIRCTSupport
  MLIRIR
)

