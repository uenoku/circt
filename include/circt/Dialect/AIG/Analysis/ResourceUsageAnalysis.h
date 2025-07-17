//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the longest path analysis for the AIG dialect.
// The analysis computes the maximum delay through combinational paths in a
// circuit where each AIG and-inverter operation is considered to have a unit
// delay. It handles module hierarchies and provides detailed path information
// including sources, sinks, delays, and debug points.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_AIG_RESOURCEUSAGEANALYSIS_H
#define CIRCT_ANALYSIS_AIG_RESOURCEUSAGEANALYSIS_H

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
namespace aig {
class ResourceUsageAnalysis {
public:
  ResourceUsageAnalysis(mlir::Operation *moduleOp, mlir::AnalysisManager &am);

  struct ResourceUsage {
    ResourceUsage(uint64_t numAndInverterGates, uint64_t numDFFBits,  uint64_t numLUTs)
        : numAndInverterGates(numAndInverterGates), numDFFBits(numDFFBits), numLUTs(numLUTs) {}
    ResourceUsage() = default;
    ResourceUsage &operator+=(const ResourceUsage &other) {
      numAndInverterGates += other.numAndInverterGates;
      numDFFBits += other.numDFFBits;
      numLUTs += other.numLUTs;
      return *this;
    }

    uint64_t getNumAndInverterGates() const { return numAndInverterGates; }
    uint64_t getNumDFFBits() const { return numDFFBits; }
    uint64_t getNumLUTs() const { return numLUTs; }

  private:
    uint64_t numAndInverterGates = 0;
    uint64_t numDFFBits = 0;
    uint64_t numLUTs = 0;
  };

  struct ModuleResourceUsage {
    ModuleResourceUsage(StringAttr moduleName, ResourceUsage local,
                        ResourceUsage total)
        : moduleName(moduleName), local(local), total(total) {}
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
    ResourceUsage getTotal() const { return total; }
    void emitJSON(raw_ostream &os) const;
  };

  ModuleResourceUsage *getResourceUsage(hw::HWModuleOp top);
  ModuleResourceUsage *getResourceUsage(StringAttr moduleName);

  // A map from the top-level module to the resource usage of the design.
  DenseMap<StringAttr, std::unique_ptr<ModuleResourceUsage>> designUsageCache;
  igraph::InstanceGraph *instanceGraph;
};
} // namespace aig
} // namespace circt

#endif