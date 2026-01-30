//===- PathFilter.cpp - Path filtering for timing analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/PathFilter.h"

using namespace circt;
using namespace synth;

llvm::Expected<PathFilter>
PathFilter::create(llvm::ArrayRef<std::string> startPatternStrs,
                   llvm::ArrayRef<std::string> endPatternStrs) {
  PathFilter filter;

  for (const auto &patStr : startPatternStrs) {
    if (patStr.empty())
      continue;
    auto pat = llvm::GlobPattern::create(patStr);
    if (!pat)
      return pat.takeError();
    filter.startPatterns.push_back(std::move(*pat));
    filter.startPatternStrings.push_back(patStr);
  }

  for (const auto &patStr : endPatternStrs) {
    if (patStr.empty())
      continue;
    auto pat = llvm::GlobPattern::create(patStr);
    if (!pat)
      return pat.takeError();
    filter.endPatterns.push_back(std::move(*pat));
    filter.endPatternStrings.push_back(patStr);
  }

  return filter;
}

bool PathFilter::matchesStartPoint(llvm::StringRef name) const {
  if (startPatterns.empty())
    return true; // No filter = match all
  for (const auto &pat : startPatterns) {
    if (pat.match(name))
      return true;
  }
  return false;
}

bool PathFilter::matchesEndPoint(llvm::StringRef name) const {
  if (endPatterns.empty())
    return true; // No filter = match all
  for (const auto &pat : endPatterns) {
    if (pat.match(name))
      return true;
  }
  return false;
}

bool PathFilter::matches(const DataflowPath &path) const {
  if (isPassthrough())
    return true;

  // Check start point filter.
  std::string startName = getStartPointFullName(path);
  if (!matchesStartPoint(startName))
    return false;

  // Check end point filter.
  // If port enumeration has been done, use index-based matching for OutputPort.
  const auto &ep = path.getEndPoint();
  if (portsEnumerated && std::holds_alternative<DataflowPath::OutputPort>(ep)) {
    auto &[module, resultNumber, bitPos] =
        std::get<DataflowPath::OutputPort>(ep);
    return matchesOutputPortIndex(resultNumber);
  }

  // Fall back to name-based matching.
  std::string endName = getEndPointFullName(path);
  return matchesEndPoint(endName);
}

std::string circt::synth::getStartPointFullName(const DataflowPath &path) {
  return path.getStartPoint().getFullPathName();
}

std::string circt::synth::getEndPointFullName(const DataflowPath &path) {
  const auto &ep = path.getEndPoint();

  if (auto *obj = std::get_if<Object>(&ep))
    return obj->getFullPathName();

  // OutputPort case: (module, resultNumber, bitPos)
  auto &[module, resultNumber, bitPos] =
      *std::get_if<DataflowPath::OutputPort>(&ep);
  return path.getRoot().getOutputName(resultNumber).str();
}

llvm::SmallVector<std::string>
PathFilter::computeMatchingOutputPorts(hw::HWModuleOp module) {
  llvm::SmallVector<std::string> matchedNames;
  portsEnumerated = true;
  matchedOutputPortIndices.clear();

  // If no end patterns, we don't need to enumerate (match all).
  if (endPatterns.empty())
    return matchedNames;

  // Enumerate all output ports and check if they match any end pattern.
  auto moduleType = module.getModuleType();
  size_t numOutputs = moduleType.getNumOutputs();

  for (size_t i = 0; i < numOutputs; ++i) {
    StringRef portName = module.getOutputName(i);
    bool matched = false;
    for (const auto &pat : endPatterns) {
      if (pat.match(portName)) {
        matched = true;
        break;
      }
    }
    if (matched) {
      matchedOutputPortIndices.insert(i);
      matchedNames.push_back(portName.str());
    }
  }

  return matchedNames;
}

llvm::SmallVector<std::string>
PathFilter::computeMatchingEndObjects(hw::HWModuleOp module,
                                      const LongestPathAnalysis &analysis) {
  objectsEnumerated = true;

  // If no end patterns, return empty (match all)
  if (endPatterns.empty())
    return {};

  // Use the analysis's getPathsToMatchingObjects to enumerate matching objects
  llvm::SmallVector<std::string> matchedNames;
  llvm::SmallVector<DataflowPath> matchedPaths;

  auto result = analysis.getPathsToMatchingObjects(
      module.getModuleNameAttr(), endPatternStrings, matchedPaths,
      &matchedNames);

  if (failed(result))
    return matchedNames;

  return matchedNames;
}

llvm::SmallVector<std::string> PathFilter::getStartPatternStrings() const {
  llvm::SmallVector<std::string> result;
  result.reserve(startPatternStrings.size());
  for (const auto &s : startPatternStrings)
    result.push_back(s);
  return result;
}

llvm::SmallVector<std::string> PathFilter::getEndPatternStrings() const {
  llvm::SmallVector<std::string> result;
  result.reserve(endPatternStrings.size());
  for (const auto &s : endPatternStrings)
    result.push_back(s);
  return result;
}

