//===- ResourceUsageAnalysis.cpp - Resource Usage Analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the resource usage analysis for the Synth dialect.
// The analysis computes resource utilization including and-inverter gates,
// DFF bits, and LUTs across module hierarchies.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/ResourceUsageAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_PRINTRESOURCEUSAGEANALYSIS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;

//===----------------------------------------------------------------------===//
// ResourceUsageAnalysis Implementation
//===----------------------------------------------------------------------===//

ResourceUsageAnalysis::ResourceUsageAnalysis(Operation *moduleOp,
                                             mlir::AnalysisManager &am)
    : instanceGraph(&am.getAnalysis<igraph::InstanceGraph>()) {}

/// Get resource usage for a module by name. Returns nullptr if the module
/// is not found or is not a ModuleOpInterface.
ResourceUsageAnalysis::ModuleResourceUsage *
circt::synth::ResourceUsageAnalysis::getResourceUsage(StringAttr moduleName) {
  // Check cache first.
  auto it = designUsageCache.find(moduleName);
  if (it != designUsageCache.end())
    return it->second.get();

  // Lookup module in instance graph.
  auto *node = instanceGraph->lookup(moduleName);
  if (!node || !node->getModule())
    return nullptr;

  auto module = dyn_cast<igraph::ModuleOpInterface>(node->getModule());
  if (!module)
    return nullptr;

  return getResourceUsage(module);
}

/// Compute resource usage for a module. This walks the module's operations
/// and counts resources, then recursively processes child module instances.
ResourceUsageAnalysis::ModuleResourceUsage *
circt::synth::ResourceUsageAnalysis::getResourceUsage(
    igraph::ModuleOpInterface module) {
  // Check cache first.
  {
    auto it = designUsageCache.find(module.getModuleNameAttr());
    if (it != designUsageCache.end())
      return it->second.get();
  }

  auto *node = instanceGraph->lookup(module.getModuleNameAttr());

  // Count local resources by walking all operations in the module.
  llvm::StringMap<uint64_t> counts;
  module->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        // Variadic logic operations (AND, OR, XOR, AIG).
        // The number of gates is (num_inputs - 1) for multi-input ops.
        // Multiply by bitwidth to account for multi-bit operations.
        .Case<synth::aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp>(
            [&](auto logicOp) {
              counts[op->getName().getStringRef()] +=
                  (logicOp.getNumOperands() - 1) *
                  logicOp.getType().getIntOrFloatBitWidth();
            })
        // Majority-inverter gates (MIG).
        .Case<synth::mig::MajorityInverterOp>([&](auto logicOp) {
          counts[op->getName().getStringRef()] +=
              (logicOp.getNumOperands() / 2) *
              logicOp.getType().getIntOrFloatBitWidth();
        })
        // Truth tables and sequential elements (registers/flip-flops).
        .Case<comb::TruthTableOp, seq::CompRegOp, seq::FirRegOp>(
            [&](auto misc) {
              counts[op->getName().getStringRef()] +=
                  misc.getType().getIntOrFloatBitWidth();
              // TODO: Account for and-inverter gates used for reset logic.
            })
        .Default([](auto) {});
  });

  ResourceUsage local(std::move(counts));
  // Initialize module usage with local counts. Total will be updated as we
  // process child instances.
  auto moduleUsage = std::make_unique<ModuleResourceUsage>(
      module.getModuleNameAttr(), local, local);

  // Recursively process child module instances.
  for (auto *child : *node) {
    auto *targetMod = child->getTarget();
    auto childModule =
        dyn_cast_or_null<igraph::ModuleOpInterface>(targetMod->getModule());
    if (!childModule)
      continue;

    auto *instance = child->getInstance().getOperation();
    // Skip instances with no results or marked as "doNotPrint".
    if (instance->getNumResults() == 0 ||
        instance->hasAttrOfType<UnitAttr>("doNotPrint"))
      continue;

    // Recursively compute child usage and accumulate into total.
    auto *childUsage = getResourceUsage(childModule);
    moduleUsage->total += childUsage->total;
    moduleUsage->instances.emplace_back(
        childModule.getModuleNameAttr(),
        child->getInstance().getInstanceNameAttr(), childUsage);
  }

  // Insert into cache and return.
  auto [it, success] = designUsageCache.try_emplace(module.getModuleNameAttr(),
                                                    std::move(moduleUsage));
  assert(success && "module already exists in cache");

  return it->second.get();
}

//===----------------------------------------------------------------------===//
// JSON Serialization
//===----------------------------------------------------------------------===//

/// Convert ResourceUsage to JSON object.
static llvm::json::Object
getModuleResourceUsageJSON(const ResourceUsageAnalysis::ResourceUsage &usage) {
  llvm::json::Object obj;
  for (const auto &count : usage.getCounts())
    obj[count.getKey()] = count.second;
  return obj;
}

/// Convert ModuleResourceUsage to JSON object with full hierarchy.
/// This creates fully-elaborated information including all child instances.
static llvm::json::Object getModuleResourceUsageJSON(
    const ResourceUsageAnalysis::ModuleResourceUsage &usage) {
  llvm::json::Object obj;
  obj["moduleName"] = usage.moduleName.getValue();
  obj["local"] = getModuleResourceUsageJSON(usage.local);
  obj["total"] = getModuleResourceUsageJSON(usage.total);

  // Serialize child instances recursively.
  SmallVector<llvm::json::Value> instances;
  for (const auto &instance : usage.instances) {
    llvm::json::Object child;
    child["instanceName"] = instance.instanceName.getValue();
    child["moduleName"] = instance.moduleName.getValue();
    child["usage"] = getModuleResourceUsageJSON(*instance.usage);
    instances.push_back(std::move(child));
  }
  obj["instances"] = llvm::json::Array(instances);

  return obj;
}

void ResourceUsageAnalysis::ModuleResourceUsage::emitJSON(
    raw_ostream &os) const {
  os << getModuleResourceUsageJSON(*this);
}

namespace {
struct PrintResourceUsageAnalysisPass
    : public impl::PrintResourceUsageAnalysisBase<
          PrintResourceUsageAnalysisPass> {
  using PrintResourceUsageAnalysisBase::PrintResourceUsageAnalysisBase;

  void runOnOperation() override;
  LogicalResult printAnalysisResult(ResourceUsageAnalysis &analysis,
                                    igraph::ModuleOpInterface top,
                                    llvm::raw_ostream *os,
                                    llvm::json::OStream *jsonOS);
};
} // namespace

LogicalResult PrintResourceUsageAnalysisPass::printAnalysisResult(
    ResourceUsageAnalysis &analysis, igraph::ModuleOpInterface top,
    llvm::raw_ostream *os, llvm::json::OStream *jsonOS) {
  auto *usage = analysis.getResourceUsage(top);
  if (!usage)
    return failure();

  if (jsonOS) {
    usage->emitJSON(jsonOS->rawValueBegin());
    jsonOS->rawValueEnd();
  } else if (os) {
    auto &stream = *os;
    stream << "Resource Usage Analysis for module: "
           << usage->moduleName.getValue() << "\n";
    stream << "========================================\n";
    stream << "Total:\n";
    // Sort.
    SmallVector<std::pair<StringRef, uint64_t>> sortedCounts;
    for (const auto &count : usage->getTotal().getCounts())
      sortedCounts.emplace_back(count.getKey(), count.second);
    llvm::sort(sortedCounts, [](const auto &a, const auto &b) {
      return a.first.compare(b.first) <= 0;
    });
    for (const auto &[name, count] : sortedCounts)
      stream << "  " << name << ": " << count << "\n";

    stream << "\n";
  }

  return success();
}

/// Run the PrintResourceUsageAnalysis pass.
/// This pass computes and prints resource usage for top-level modules.
void PrintResourceUsageAnalysisPass::runOnOperation() {
  auto mod = getOperation();

  auto &resourceUsage = getAnalysis<ResourceUsageAnalysis>();
  auto *instanceGraph = resourceUsage.instanceGraph;

  // Determine which modules to analyze.
  SmallVector<igraph::ModuleOpInterface> tops;
  if (topModuleName.getValue().empty()) {
    // Automatically infer top modules from instance graph.
    auto topLevelNodes = instanceGraph->getInferredTopLevelNodes();
    if (failed(topLevelNodes)) {
      mod.emitError()
          << "failed to infer top-level modules from instance graph";
      return signalPassFailure();
    }

    // Collect all ModuleOpInterface instances from top-level nodes.
    for (auto *node : *topLevelNodes) {
      if (auto module = dyn_cast<igraph::ModuleOpInterface>(node->getModule()))
        tops.push_back(module);
    }

    if (tops.empty()) {
      mod.emitError() << "no top-level modules found in instance graph";
      return signalPassFailure();
    }
  } else {
    // Use user-specified top module name.
    auto *node = instanceGraph->lookup(
        mlir::StringAttr::get(mod.getContext(), topModuleName.getValue()));
    if (!node || !node->getModule()) {
      mod.emitError() << "top module '" << topModuleName.getValue()
                      << "' not found";
      return signalPassFailure();
    }
    auto top = dyn_cast<igraph::ModuleOpInterface>(node->getModule());
    if (!top) {
      mod.emitError() << "module '" << topModuleName.getValue()
                      << "' is not a ModuleOpInterface";
      return signalPassFailure();
    }
    tops.push_back(top);
  }

  // Open output file.
  std::string error;
  auto file = mlir::openOutputFile(outputFile.getValue(), &error);
  if (!file) {
    llvm::errs() << error;
    return signalPassFailure();
  }

  auto &os = file->os();
  std::unique_ptr<llvm::json::OStream> jsonOS;
  if (emitJSON.getValue()) {
    // Initialize JSON array for multiple modules.
    jsonOS = std::make_unique<llvm::json::OStream>(os);
    jsonOS->arrayBegin();
  }

  // Ensure JSON array is properly closed on exit.
  auto closeJson = llvm::make_scope_exit([&]() {
    if (jsonOS)
      jsonOS->arrayEnd();
  });

  // Print resource usage for each top module.
  for (auto top : tops) {
    if (failed(printAnalysisResult(resourceUsage, top, jsonOS ? nullptr : &os,
                                   jsonOS.get())))
      return signalPassFailure();
  }

  file->keep();
  return markAllAnalysesPreserved();
}
