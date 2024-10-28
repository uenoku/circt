//===- OpCountAnalysis.h - operation count analyses -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving the frequency of different kinds of operations found in a
// builtin.module.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_AIG_ANALYSIS_H
#define CIRCT_ANALYSIS_AIG_ANALYSIS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace mlir {
class AnalysisManager;
} // namespace mlir
namespace circt {
namespace igraph {
class InstanceGraph;
};
namespace aig {

/*
class StaticTimingAnalysis {
public:
  StaticTimingAnalysis(Operation *moduleOp, mlir::AnalysisManager &am);

private:
  struct ElaboratedRegister {
    circt::igraph::InstancePath path;
    Value value; // This must be a register.
    size_t bitPos;
    StringAttr name;
  };

  struct PortNode {
    StringAttr name;
    size_t bitPos;
  };
    // static timing analysis <=> find longest path between two objects
    // where cost is the number of AndInverter gates in the path.

  using LocalGraph =
      DenseMap<BitRef, SmallVector<std::pair<OpOperand, size_t>>>;

  DenseMap<OperationName, size_t> opCounts;
  DenseMap<OperationName, DenseMap<size_t, size_t>> operandCounts;
};*/

class ResourceUsageAnalysis {
public:
  ResourceUsageAnalysis(Operation *moduleOp, mlir::AnalysisManager &am);

  struct ResourceUsage {
    ResourceUsage(size_t numAndInverterGates, size_t numDFFBits)
        : numAndInverterGates(numAndInverterGates), numDFFBits(numDFFBits) {}
    ResourceUsage() = default;
    ResourceUsage &operator+=(const ResourceUsage &other) {
      numAndInverterGates += other.numAndInverterGates;
      numDFFBits += other.numDFFBits;
      return *this;
    }

    size_t getNumAndInverterGates() const { return numAndInverterGates; }
    size_t getNumDFFBits() const { return numDFFBits; }

  private:
    size_t numAndInverterGates = 0;
    size_t numDFFBits = 0;
  };

  struct ModuleResourceUsage {
    ModuleResourceUsage(ResourceUsage local, ResourceUsage total)
        : local(local), total(total) {}
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

  // A map from the top-level module to the resource usage of the design.
  DenseMap<StringAttr, std::shared_ptr<ModuleResourceUsage>> designUsageCache;
  igraph::InstanceGraph *instanceGraph;
};

} // namespace aig
} // namespace circt

#endif // CIRCT_ANALYSIS_AIG_ANALYSIS_H
