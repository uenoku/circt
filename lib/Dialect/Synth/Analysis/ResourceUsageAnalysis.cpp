//===- ResourceUsageAnalysis.cpp - resource usage analysis ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the resource usage analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/ResourceUsageAnalysis.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/Analysis/ResourceUsageAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/IR/OperationSupport.h>
namespace circt {
namespace synth {
#define GEN_PASS_DEF_PRINTRESOURCEUSAGEANALYSIS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;

ResourceUsageAnalysis::ResourceUsageAnalysis(Operation *moduleOp,
                                             mlir::AnalysisManager &am)
    : instanceGraph(&am.getAnalysis<igraph::InstanceGraph>()) {}

ResourceUsageAnalysis::ModuleResourceUsage *
circt::synth::ResourceUsageAnalysis::getResourceUsage(StringAttr moduleName) {
  auto it = designUsageCache.find(moduleName);
  if (it != designUsageCache.end())
    return it->second.get();

  auto *node = instanceGraph->lookup(moduleName);
  if (!node || !node->getModule() || !isa<hw::HWModuleOp>(node->getModule()))
    return nullptr;
  return getResourceUsage(node->getModule<hw::HWModuleOp>());
}

ResourceUsageAnalysis::ModuleResourceUsage *
circt::synth::ResourceUsageAnalysis::getResourceUsage(hw::HWModuleOp module) {
  {
    auto it = designUsageCache.find(module.getModuleNameAttr());
    if (it != designUsageCache.end())
      return it->second.get();
  }
  auto *node = instanceGraph->lookup(module.getModuleNameAttr());
  llvm::StringMap<uint64_t> counts;
  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        // Variadic.
        .Case<synth::aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp>(
            [&](auto logicOp) {
              // The number of and-inverter gates is the number of inputs minus
              // one, except for the case where there is only one input, which
              // is a pass-through inverter. We multiply by the bitwidth to
              // account for multiple bits.
              counts[op->getName().getStringRef()] +=
                  (logicOp.getNumOperands() - 1) *
                  logicOp.getType().getIntOrFloatBitWidth();
            })
        .Case<synth::mig::MajorityInverterOp>([&](auto logicOp) {
          counts[op->getName().getStringRef()] +=
              (logicOp.getNumOperands() / 2) *
              logicOp.getType().getIntOrFloatBitWidth();
        })
        .Case<comb::TruthTableOp, seq::CompRegOp, seq::FirRegOp>(
            [&](auto misc) {
              // NOTE: We might want to reject seq-level registers in Synth at
              // this point and instead only support synth-level flops.
              // numDFFBits += regOp.getType().getIntOrFloatBitWidth();
              counts[op->getName().getStringRef()] +=
                  misc.getType().getIntOrFloatBitWidth();
              // TODO: We need to take into account the number of
              // andInverterGates used for reset.
            })
        // TODO: Add support for memory for more accurate resource usage.
        // TODO: Consider rejecting operations that could introduce
        // AndInverter.
        .Default([](auto) {});
  });
  ResourceUsage local(std::move(counts));
  auto moduleUsage = std::make_unique<ModuleResourceUsage>(
      module.getModuleNameAttr(), local, local);

  for (auto *child : *node) {
    auto *targetMod = child->getTarget();
    if (!isa_and_nonnull<hw::HWModuleOp>(targetMod->getModule()))
      continue;

    auto *instance = child->getInstance().getOperation();
    if (instance->getNumResults() == 0 ||
        instance->hasAttrOfType<UnitAttr>("doNotPrint"))
      continue;

    auto childModule = targetMod->getModule<hw::HWModuleOp>();
    auto *childUsage = getResourceUsage(childModule);
    moduleUsage->total += childUsage->total;
    moduleUsage->instances.emplace_back(
        childModule.getModuleNameAttr(),
        child->getInstance().getInstanceNameAttr(), childUsage);
  }

  // Insert into cache first
  auto [it, success] = designUsageCache.try_emplace(module.getModuleNameAttr(),
                                                    std::move(moduleUsage));
  assert(success && "module already exists in cache");

  return it->second.get();
}

static llvm::json::Object
getModuleResourceUsageJSON(const ResourceUsageAnalysis::ResourceUsage &usage) {
  llvm::json::Object obj;
  for (const auto &count : usage.getCounts())
    obj[count.getKey()] = count.second;

  return obj;
}

// This creates a fully-elaborated information but should be ok for now.
static llvm::json::Object getModuleResourceUsageJSON(
    const ResourceUsageAnalysis::ModuleResourceUsage &usage) {
  llvm::json::Object obj;
  obj["local"] = getModuleResourceUsageJSON(usage.local);
  obj["total"] = getModuleResourceUsageJSON(usage.total);
  obj["moduleName"] = usage.moduleName.getValue();
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
                                    hw::HWModuleOp top, llvm::raw_ostream *os,
                                    llvm::json::OStream *jsonOS);
};
} // namespace

LogicalResult PrintResourceUsageAnalysisPass::printAnalysisResult(
    ResourceUsageAnalysis &analysis, hw::HWModuleOp top, llvm::raw_ostream *os,
    llvm::json::OStream *jsonOS) {
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
    for (const auto &count : usage->getTotal().getCounts())
      stream << "  " << count.getKey() << ": " << count.second << "\n";
    stream << "\n";
  }

  return success();
}

void PrintResourceUsageAnalysisPass::runOnOperation() {
  auto mod = getOperation();

  auto &resourceUsage = getAnalysis<ResourceUsageAnalysis>();
  auto *instanceGraph = resourceUsage.instanceGraph;

  SmallVector<hw::HWModuleOp> tops;
  if (topModuleName.getValue().empty()) {
    // Automatically infer top modules from instance graph
    auto topLevelNodes = instanceGraph->getInferredTopLevelNodes();
    if (failed(topLevelNodes)) {
      mod.emitError()
          << "failed to infer top-level modules from instance graph";
      return signalPassFailure();
    }

    for (auto *node : *topLevelNodes) {
      if (auto hwMod = dyn_cast<hw::HWModuleOp>(node->getModule()))
        tops.push_back(hwMod);
    }

    if (tops.empty()) {
      mod.emitError() << "no top-level HWModuleOp found in instance graph";
      return signalPassFailure();
    }
  } else {
    // Use specified top module name
    auto &symTbl = getAnalysis<mlir::SymbolTable>();
    auto top = symTbl.lookup<hw::HWModuleOp>(topModuleName.getValue());
    if (!top) {
      mod.emitError() << "top module '" << topModuleName.getValue()
                      << "' not found";
      return signalPassFailure();
    }
    tops.push_back(top);
  }

  std::string error;
  auto file = mlir::openOutputFile(outputFile.getValue(), &error);
  if (!file) {
    llvm::errs() << error;
    return signalPassFailure();
  }

  auto &os = file->os();
  std::unique_ptr<llvm::json::OStream> jsonOS;
  if (emitJSON.getValue()) {
    jsonOS = std::make_unique<llvm::json::OStream>(os);
    jsonOS->arrayBegin();
  }

  auto closeJson = llvm::make_scope_exit([&]() {
    if (jsonOS)
      jsonOS->arrayEnd();
  });

  for (auto top : tops) {
    if (failed(printAnalysisResult(resourceUsage, top, jsonOS ? nullptr : &os,
                                   jsonOS.get())))
      return signalPassFailure();
  }

  file->keep();
  return markAllAnalysesPreserved();
}
