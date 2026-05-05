//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal helpers shared by cut-rewriter-based passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_LIB_DIALECT_SYNTH_TRANSFORMS_CUTREWRITERINTERNAL_H
#define CIRCT_LIB_DIALECT_SYNTH_TRANSFORMS_CUTREWRITERINTERNAL_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>
#include <utility>

namespace circt {
namespace synth {

FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
parseModuleFile(llvm::StringRef path, mlir::MLIRContext *context);

/// Compute the NPN canonical form of a single-output i1 `hw.module`.
FailureOr<NPNClass> getNPNClassFromModule(hw::HWModuleOp module);

class NPNCutRewritePattern : public CutRewritePattern {
public:
  NPNCutRewritePattern(MLIRContext *context, std::string patternName,
                       double area, SmallVector<DelayType> delay,
                       NPNClass npnClass)
      : CutRewritePattern(context), patternName(std::move(patternName)),
        area(area), delay(std::move(delay)), npnClass(std::move(npnClass)) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    if (!cut.getNPNClass(enumerator.getOptions().npnTable)
             .equivalentOtherThanPermutation(npnClass))
      return std::nullopt;
    return MatchResult(area, delay);
  }

  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(npnClass);
    return true;
  }

  unsigned getNumOutputs() const override { return 1; }
  StringRef getPatternName() const override { return patternName; }

protected:
  const NPNClass &getNPNClass() const { return npnClass; }

private:
  std::string patternName;
  double area;
  SmallVector<DelayType> delay;
  NPNClass npnClass;
};

} // namespace synth
} // namespace circt

#endif // CIRCT_LIB_DIALECT_SYNTH_TRANSFORMS_CUTREWRITERINTERNAL_H
