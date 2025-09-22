//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mockturtle integration functions for CIRCT.
//
//===----------------------------------------------------------------------===//

#include "MockturtleAdapter.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mockturtle-integration"

using namespace circt;
using namespace circt::synth;
