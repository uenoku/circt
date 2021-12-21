//===- circt/Support/Parallel.h - Parallel algorithms ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PARALLEL_H
#define CIRCT_SUPPORT_PARALLEL_H

#include "mlir/IR/Threading.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <numeric>

namespace circt {

/// Invoke the given function on the elements in the provided range
/// asynchronously in descending order of the estimated execution time.
template <typename RandomAccessRangeT, typename FuncT, typename EstimateF>
void parallelForEach(mlir::MLIRContext *context, RandomAccessRangeT &&range,
                     FuncT &&func, EstimateF &&estimate) {
  llvm::SmallVector<size_t> indexes(llvm::size(range));
  // Set indexes to {0, 1, ..., size - 1}.
  std::iota(indexes.begin(), indexes.end(), 0);

  llvm::SmallVector<int64_t> estimatedTime;
  estimatedTime.reserve(indexes.size());
  for (auto i : indexes)
    estimatedTime.push_back(estimate(range[i]));

  llvm::sort(std::begin(indexes), std::end(indexes),
             [&](size_t lhs, size_t rhs) {
               return estimatedTime[lhs] > estimatedTime[rhs];
             });

  mlir::parallelForEach(context, indexes, [&](size_t i) { func(range[i]); });
}
} // namespace circt

#endif