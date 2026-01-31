#===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//
#
# Mockturtle integration for CIRCT synthesis passes.
# This directory is RTTI/Exception-free (-fno-exceptions -fno-rtti).
# Direct mockturtle calls requiring RTTI/exceptions are isolated in Bindings/.
#
#===----------------------------------------------------------------------===//

add_subdirectory(Bindings)
add_compile_options(-Wno-c++11-narrowing)

add_circt_library(CIRCTSynthMockturtleIntegration
  Emap.cpp
  MockturtlePasses.cpp
  NetworkConversion.cpp

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSynthAnalysis
  CIRCTSupport
  MLIRIR
  mockturtle_interface
  CIRCTSynthMockturtleIntegrationBindings
)
