#===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//
#
# This directory contains mockturtle library bindings that require RTTI and
# exceptions. Code here directly interfaces with mockturtle's C++ API.
#
#===----------------------------------------------------------------------===//

add_compile_options(-fexceptions)
add_compile_options(-frtti)
add_compile_options(-Wno-c++11-narrowing)

add_circt_library(CIRCTSynthMockturtleIntegrationBindings
  Bind.cpp

  LINK_LIBS
  mockturtle_interface
)
