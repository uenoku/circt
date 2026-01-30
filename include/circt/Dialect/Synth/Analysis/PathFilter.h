//===- PathFilter.h - Path filtering for timing analysis --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PathFilter class for filtering timing paths by
// start and/or end point name patterns using glob matching.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_PATHFILTER_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_PATHFILTER_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/GlobPattern.h"
#include <string>

namespace circt {
namespace synth {

/// PathFilter provides glob-based filtering of DataflowPaths by start and/or
/// end point names. It supports shell-like wildcard patterns:
///   - `*` matches zero or more characters
///   - `?` matches exactly one character
///   - `[abc]` matches any character in the set
///   - `[a-z]` matches any character in the range
///
/// Multiple patterns for start/end points use OR semantics (match any).
/// When both start and end patterns are specified, AND semantics apply
/// (path must match at least one pattern from each group).
///
/// Example usage:
/// @code
///   auto filter = PathFilter::create({"data_*", "clk_*"}, {"*_out"});
///   if (!filter) handleError(filter.takeError());
///
///   for (auto &path : paths) {
///     if (filter->matches(path))
///       processPath(path);
///   }
/// @endcode
class PathFilter {
public:
  /// Create a PathFilter from glob pattern strings.
  /// Returns an error if any pattern is invalid.
  /// Empty pattern arrays mean "match all" for that endpoint.
  static llvm::Expected<PathFilter>
  create(llvm::ArrayRef<std::string> startPatterns,
         llvm::ArrayRef<std::string> endPatterns);

  /// Create a passthrough filter that matches all paths.
  static PathFilter createPassthrough() { return PathFilter(); }

  /// Check if a DataflowPath matches the filter criteria.
  /// Returns true if:
  ///   - Start point name matches any start pattern (or no start patterns)
  ///   - AND end point name matches any end pattern (or no end patterns)
  bool matches(const DataflowPath &path) const;

  /// Check if a name matches the start point patterns.
  /// Returns true if no start patterns are specified.
  bool matchesStartPoint(llvm::StringRef name) const;

  /// Check if a name matches the end point patterns.
  /// Returns true if no end patterns are specified.
  bool matchesEndPoint(llvm::StringRef name) const;

  /// Return true if no filtering is active (all paths pass through).
  bool isPassthrough() const {
    return startPatterns.empty() && endPatterns.empty();
  }

  /// Get the number of start patterns.
  size_t getNumStartPatterns() const { return startPatterns.size(); }

  /// Get the number of end patterns.
  size_t getNumEndPatterns() const { return endPatterns.size(); }

  /// Enumerate output ports from a module and compute which ones match
  /// the end patterns. Returns the names of matched ports.
  /// This should be called once per module to pre-compute matching ports.
  llvm::SmallVector<std::string>
  computeMatchingOutputPorts(hw::HWModuleOp module);

  /// Enumerate all objects (ports, registers, combinational nodes) from a
  /// module and compute which ones match the end patterns. Returns the names
  /// of matched objects. This enables matching arbitrary design objects,
  /// not just output ports.
  llvm::SmallVector<std::string>
  computeMatchingEndObjects(hw::HWModuleOp module,
                            const LongestPathAnalysis &analysis);

  /// Get the list of start patterns as strings.
  llvm::SmallVector<std::string> getStartPatternStrings() const;

  /// Get the list of end patterns as strings (for use with
  /// getPathsToMatchingObjects).
  llvm::SmallVector<std::string> getEndPatternStrings() const;

  /// Get the set of matched output port indices after computeMatchingOutputPorts.
  const llvm::DenseSet<size_t> &getMatchedOutputPortIndices() const {
    return matchedOutputPortIndices;
  }

  /// Check if a specific output port index matches the filter.
  /// Only valid after calling computeMatchingOutputPorts.
  bool matchesOutputPortIndex(size_t portIndex) const {
    return matchedOutputPortIndices.empty() || // Empty means match all
           matchedOutputPortIndices.contains(portIndex);
  }

  /// Return true if port enumeration has been performed.
  bool hasEnumeratedPorts() const { return portsEnumerated; }

  /// Return true if object enumeration has been performed.
  bool hasEnumeratedObjects() const { return objectsEnumerated; }

private:
  PathFilter() = default;

  llvm::SmallVector<llvm::GlobPattern, 2> startPatterns;
  llvm::SmallVector<llvm::GlobPattern, 2> endPatterns;

  /// Original pattern strings (needed for getPathsToMatchingObjects API).
  llvm::SmallVector<std::string, 2> startPatternStrings;
  llvm::SmallVector<std::string, 2> endPatternStrings;

  /// Cache of output port indices that match the end patterns.
  /// Computed by computeMatchingOutputPorts.
  llvm::DenseSet<size_t> matchedOutputPortIndices;
  bool portsEnumerated = false;
  bool objectsEnumerated = false;
};

/// Get the full hierarchical path name for a DataflowPath endpoint.
/// Format: "inst1/inst2/signalName"
std::string getEndPointFullName(const DataflowPath &path);

/// Get the full hierarchical path name for a DataflowPath start point.
/// Format: "inst1/inst2/signalName"
std::string getStartPointFullName(const DataflowPath &path);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_PATHFILTER_H

