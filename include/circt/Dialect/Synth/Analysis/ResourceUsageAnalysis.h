//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the resource usage analysis for the Synth dialect.
// The analysis computes resource utilization including and-inverter gates,
// DFF bits, and LUTs across module hierarchies.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_RESOURCEUSAGEANALYSIS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_RESOURCEUSAGEANALYSIS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <mlir/IR/Attributes.h>
#include <variant>

namespace mlir {
class AnalysisManager;
} // namespace mlir

namespace circt {
namespace synth {
class ResourceUsageAnalysis {
public:
  ResourceUsageAnalysis(mlir::Operation *moduleOp, mlir::AnalysisManager &am);

  struct ResourceUsage {
    ResourceUsage(llvm::StringMap<uint64_t> counts)
        : counts(std::move(counts)) {}
    ResourceUsage() = default;
    ResourceUsage &operator+=(const ResourceUsage &other) {
      for (const auto &count : other.counts)
        counts[count.getKey()] += count.second;
      return *this;
    }
    const auto &getCounts() const { return counts; }

  private:
    llvm::StringMap<uint64_t> counts;
  };

  struct ModuleResourceUsage {
    ModuleResourceUsage(StringAttr moduleName, ResourceUsage local,
                        ResourceUsage total)
        : moduleName(moduleName), local(std::move(local)),
          total(std::move(total)) {}
    StringAttr moduleName;
    ResourceUsage local, total;
    struct InstanceResource {
      StringAttr moduleName, instanceName;
      ModuleResourceUsage *usage;
      InstanceResource(StringAttr moduleName, StringAttr instanceName,
                       ModuleResourceUsage *usage)
          : moduleName(moduleName), instanceName(instanceName), usage(usage) {}
    };
    SmallVector<InstanceResource> instances;
    const ResourceUsage& getTotal() const { return total; }
    void emitJSON(raw_ostream &os) const;
  };

  ModuleResourceUsage *getResourceUsage(hw::HWModuleOp top);
  ModuleResourceUsage *getResourceUsage(StringAttr moduleName);

  // A map from the top-level module to the resource usage of the design.
  DenseMap<StringAttr, std::unique_ptr<ModuleResourceUsage>> designUsageCache;
  igraph::InstanceGraph *instanceGraph;
};
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_RESOURCEUSAGEANALYSIS_H