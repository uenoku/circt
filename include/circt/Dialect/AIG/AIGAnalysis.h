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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableList.h"
#include <memory>
#include <mlir/IR/BuiltinOps.h>

namespace mlir {
class AnalysisManager;
} // namespace mlir
namespace circt {
namespace igraph {
class InstanceGraph;
}
namespace aig {

struct Object {
  circt::igraph::InstancePath instancePath;
  Value value;
  size_t bitPos;
  Object(circt::igraph::InstancePath path, Value value, size_t bitPos)
      : instancePath(path), value(value), bitPos(bitPos) {}
  Object() = default;
  void print(llvm::raw_ostream &os) const;

  bool operator==(const Object &other) const {
    return instancePath == other.instancePath && value == other.value &&
           bitPos == other.bitPos;
  }
};

// A debug point represents a point in the dataflow graph.
struct DebugPoint {
  DebugPoint(circt::igraph::InstancePath path, Value value, size_t bitPos,
             int64_t delay = 0, StringRef comment = "")
      : object(path, value, bitPos), delay(delay), comment(comment) {}

  // Trait for list of debug points.
  void Profile(llvm::FoldingSetNodeID &ID) const {
    for (auto &inst : object.instancePath) {
      ID.AddPointer(inst.getAsOpaquePointer());
    }
    ID.AddPointer(object.value.getAsOpaquePointer());
    ID.AddInteger(object.bitPos);
    ID.AddInteger(delay);
  }

  void print(llvm::raw_ostream &os) const;

  Object object;
  int64_t delay;
  StringRef comment;
};

// A class represents a path in the dataflow graph.
// The destination is: `instancePath.value[bitPos]` at time `delay`
// going through `history`.
struct DataflowPath {
  Object fanIn;
  int64_t delay = -1;
  llvm::ImmutableList<DebugPoint> history;
  DataflowPath(circt::igraph::InstancePath path, Value value, size_t bitPos,
               int64_t delay = 0, llvm::ImmutableList<DebugPoint> history = {})
      : fanIn(path, value, bitPos), delay(delay), history(history) {
    assert(value);
  }

  DataflowPath() = default;
  void print(llvm::raw_ostream &os) const;
};

struct PathResult {
  Object fanOut;
  DataflowPath fanIn;
  hw::HWModuleOp root;
  PathResult(Object fanOut, DataflowPath fanIn, hw::HWModuleOp root)
      : fanOut(fanOut), fanIn(fanIn), root(root) {}
  PathResult() = default;

  void print(llvm::raw_ostream &os);
};

// This analysis finds the longest paths in the dataflow graph across modules.
// Also be aware of the lifetime of the analysis, the results would be
// invalid if the IR is modified. Currently there is no way to efficiently
// update the analysis results, so it's recommended to only use this analysis
// once on a design, and store the results in a separate data structure which
// users can manage the lifetime.
class LongestPathAnalysis {
public:
  // Entry points for analysis.
  LongestPathAnalysis(Operation *moduleOp, mlir::AnalysisManager &am);
  ~LongestPathAnalysis();

  // Return all longest paths to each Fanin for the given value and bit
  // position. Populates the 'results' vector with PathResult objects, each
  // containing:
  // - fanout: The destination object (value/bit position in instance path)
  // - fanin: The source dataflow path with delay information
  // - root: The HWModuleOp where the path was found
  // Returns failure if the value is not in a HWModuleOp or analysis fails.
  LogicalResult getResults(Value value, size_t bitPos,
                           SmallVectorImpl<PathResult> &results) const;

  // Return the average of the maximum delays across all bits of the given
  // value, which is useful approximation for the delay of the value. For each
  // bit position, finds all paths and takes the maximum delay. Then averages
  // these maximum delays across all bits of the value.
  int64_t getAverageMaxDelay(Value value) const;

  // Paths to FFs are precomputed efficiently, return results.
  void getResultsForFF(SmallVectorImpl<PathResult> &results) const;

  // Erase the cache for the given value and bit position.
  // If bitPos is -1, erases all bit positions.
  void erase(Value value, int64_t bitPos = -1);

  // Return true if the analysis is available for the given module.
  bool isAnalysisAvaiable(hw::HWModuleOp module) const;

  // Return the top nodes that were used for the analysis.
  llvm::ArrayRef<hw::HWModuleOp> getTopModules() const;

  // This is the name of the attribute that can be attached to the module
  // to specify the top module for the analysis. This is optional, if not
  // specified, the analysis will infer the top module from the instance graph.
  // However it's recommended to specify it, as the entire module tends to
  // contain testbench or verification modules, which may have expensive paths
  // that are not of interest.
  static StringRef getTopModuleNameAttrName() {
    return "aig.longest-path-analysis-top";
  }

private:
  struct Impl;
  Impl *impl;
};

} // namespace aig
} // namespace circt

#endif // CIRCT_ANALYSIS_AIG_ANALYSIS_H
