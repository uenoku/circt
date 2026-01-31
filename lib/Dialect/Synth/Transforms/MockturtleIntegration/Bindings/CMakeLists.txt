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

# Enable RTTI and exceptions for source files in this directory (mockturtle requires them).
# We need to override CMAKE_CXX_FLAGS which contains -fno-rtti -fno-exceptions.
# Setting COMPILE_FLAGS on source files ensures these flags come after the global flags
# (last flag wins in GCC/Clang).
set_source_files_properties(Bind.cpp PROPERTIES
  COMPILE_FLAGS "-frtti -fexceptions -Wno-c++11-narrowing -Wno-suggest-override -Wno-non-virtual-dtor -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare -Wno-unused-lambda-capture -Wno-deprecated-literal-operator -Wno-deprecated-builtins -Wno-implicit-fallthrough -Wno-cast-function-type-mismatch -Wno-ctad-maybe-unsupported -Wno-cast-qual"
)

add_circt_library(CIRCTSynthMockturtleIntegrationBindings
  Bind.cpp

  LINK_LIBS
  mockturtle_interface
)
