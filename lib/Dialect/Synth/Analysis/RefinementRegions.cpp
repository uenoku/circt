//===- RefinementRegions.cpp - Datapath refinement regions ---------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/RefinementRegions.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::synth;
using namespace mlir;

FailureOr<DatapathRefinementPolicy>
circt::synth::parseRefinementPolicy(StringRef name) {
  if (name == "baseline")
    return DatapathRefinementPolicy::Baseline;
  if (name == "reduce-delay")
    return DatapathRefinementPolicy::ReduceDelay;
  if (name == "booth")
    return DatapathRefinementPolicy::ForceBooth;
  if (name == "compress-to-add")
    return DatapathRefinementPolicy::CompressToAdd;
  if (name == "fastest")
    return DatapathRefinementPolicy::FastestKnown;
  if (name == "auto")
    return DatapathRefinementPolicy::ReduceDelay;
  return failure();
}

StringRef circt::synth::refinementPolicyName(DatapathRefinementPolicy policy) {
  switch (policy) {
  case DatapathRefinementPolicy::Baseline:
    return "baseline";
  case DatapathRefinementPolicy::ReduceDelay:
    return "reduce-delay";
  case DatapathRefinementPolicy::ForceBooth:
    return "booth";
  case DatapathRefinementPolicy::CompressToAdd:
    return "compress-to-add";
  case DatapathRefinementPolicy::FastestKnown:
    return "fastest";
  }
  return "baseline";
}

std::optional<DatapathRefinementPolicy>
circt::synth::getRecordedRefinePolicy(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>(refinePolicyAttrName());
  if (!attr)
    return std::nullopt;
  auto policy = parseRefinementPolicy(attr.getValue());
  if (failed(policy))
    return std::nullopt;
  return *policy;
}

namespace {

bool isLibModule(hw::HWModuleOp module) {
  return module->hasAttr("synth.liberty.cell") ||
         module->hasAttr("synth.dont_use") ||
         module->hasAttr("synth.mapping_cost") ||
         module->hasAttr("synth.supergate_candidate") ||
         module.getModuleName().starts_with("__supergate_");
}

// Region-capable operations: datapath operations plus explicitly
// transparent arithmetic glue (extracts, concats, constants).
bool isRegionCapable(Operation *op) {
  if (isa<datapath::DatapathDialect>(op->getDialect()))
    return true;
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  return isa<comb::ExtractOp, comb::ConcatOp>(op);
}

bool isDatapathMember(Operation *op) {
  return isa<datapath::DatapathDialect>(op->getDialect());
}

// Simple union-find over operation pointers (deterministic: unions happen
// in preorder, so the component representative is preorder-stable).
struct UnionFind {
  DenseMap<Operation *, Operation *> parent;
  Operation *find(Operation *op) {
    Operation *root = op;
    while (parent.lookup(root) != root)
      root = parent.lookup(root);
    while (parent.lookup(op) != root) {
      Operation *next = parent.lookup(op);
      parent[op] = root;
      op = next;
    }
    return root;
  }
  void makeSet(Operation *op) { parent.try_emplace(op, op); }
  void unite(Operation *a, Operation *b) {
    Operation *ra = find(a);
    Operation *rb = find(b);
    if (ra == rb)
      return;
    // Representative is deterministic: unions happen in preorder with
    // operand-order traversal, so repeated runs choose the same root.
    parent[ra] = rb;
  }
};

} // namespace

RefinementPlan circt::synth::discoverRefinementRegions(
    ModuleOp top, TimingPressureAnalysis &analysis,
    const RefinementConfig &config,
    function_ref<void(const Twine &)> diagnose) {
  RefinementPlan plan;

  // Resolve violating feedback endpoints to constraining probes.
  DenseSet<Operation *> violatingProbes;
  DenseMap<Operation *, std::vector<std::string>> probeEndpoints;
  for (const auto &record : config.feedback) {
    if (!record.slackPs || *record.slackPs >= 0)
      continue;
    auto resolved = resolveFeedbackEndpoint(top, record.endpoint);
    if (failed(resolved)) {
      plan.unresolvedEndpoints.push_back(record.endpoint);
      continue;
    }
    // The probe feeding the endpoint's D operand, if any.
    Value driver = resolved->instance.getOperand(resolved->dataOperandIdx);
    auto probe = driver.getDefiningOp<TimingProbeOp>();
    if (!probe) {
      plan.unresolvedEndpoints.push_back(record.endpoint);
      continue;
    }
    violatingProbes.insert(probe);
    auto &endpointList = probeEndpoints[probe];
    if (!llvm::is_contained(endpointList, record.endpoint))
      endpointList.push_back(record.endpoint);
    ++plan.numFeedbackResolved;
    ++plan.numFeedbackViolated;
  }

  // Stable probe IDs (@Module:preorder) across all design modules, for
  // reports and driver round-tripping. Deterministic source order.
  DenseMap<Operation *, std::string> probeIds;
  for (auto module : top.getOps<hw::HWModuleOp>()) {
    if (isLibModule(module) || !module.getBodyBlock())
      continue;
    unsigned preorder = 0;
    for (auto &op : module.getBodyBlock()->getOperations()) {
      if (isa<synth::TimingProbeOp>(op))
        probeIds[&op] =
            ("@" + module.getModuleName().str() + ":" + Twine(preorder)).str();
      ++preorder;
    }
  }

  // Deterministic module order: source order.
  for (auto module : top.getOps<hw::HWModuleOp>()) {
    if (isLibModule(module) || !module.getBodyBlock())
      continue;
    Block *body = module.getBodyBlock();

    // Component discovery over region-capable operations.
    UnionFind components;
    SmallVector<Operation *> preorderOps;
    DenseMap<Operation *, unsigned> preorderIndex;
    for (auto &op : body->getOperations()) {
      preorderIndex[&op] = (unsigned)preorderOps.size();
      preorderOps.push_back(&op);
      if (isRegionCapable(&op))
        components.makeSet(&op);
    }
    for (auto &op : body->getOperations()) {
      if (!isRegionCapable(&op))
        continue;
      for (Value operand : op.getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (defOp && isRegionCapable(defOp))
          components.unite(&op, defOp);
      }
    }

    // Group components in first-seen preorder; keep those with a datapath
    // operation. Region ID: @Module:firstDatapathPreorder.
    MapVector<Operation *, SmallVector<Operation *>> groups;
    for (Operation *op : preorderOps) {
      if (!isRegionCapable(op))
        continue;
      groups[components.find(op)].push_back(op);
    }
    for (auto &[root, members] : groups) {
      (void)root;
      Operation *firstDatapath = nullptr;
      for (Operation *op : members) {
        if (isDatapathMember(op)) {
          firstDatapath = op;
          break;
        }
      }
      if (!firstDatapath)
        continue;
      RefinementRegion region;
      region.id = ("@" + module.getModuleName().str() + ":" +
                   Twine(preorderIndex[firstDatapath]))
                      .str();
      region.module = module.getModuleName().str();
      region.members = members;

      // Probe relation, tightest propagated requirement, worst slack.
      DenseSet<Operation *> seenProbes;
      for (Operation *member : members) {
        for (Value result : member->getResults()) {
          for (auto probe : analysis.getConstrainingProbes(result)) {
            if (seenProbes.insert(probe).second) {
              region.probes.push_back(probe);
              auto probeIdIt = probeIds.find(probe);
              region.probeIds.push_back(probeIdIt == probeIds.end()
                                            ? std::string("@unknown")
                                            : probeIdIt->second);
            }
          }
          TimingPoint point = analysis.getPoint(result);
          if (point.slackLevels) {
            if (!region.worstSlackLevels ||
                *point.slackLevels < *region.worstSlackLevels)
              region.worstSlackLevels = *point.slackLevels;
            region.confidence = TimingConfidence::StructuralEstimate;
          }
          if (!point.arrivalLevels || !point.requiredLevels)
            region.unresolvedBoundary = true;
        }
      }
      for (auto probe : region.probes) {
        if (!violatingProbes.contains(probe))
          continue;
        region.violated = true;
        auto endpointIt = probeEndpoints.find(probe);
        if (endpointIt != probeEndpoints.end())
          for (const auto &endpoint : endpointIt->second)
            if (!llvm::is_contained(region.endpoints, endpoint))
              region.endpoints.push_back(endpoint);
      }

      // Selection: manifest > violating cone > structural pressure.
      auto manifestIt = config.manifest.find(region.id);
      if (manifestIt != config.manifest.end()) {
        region.policy = manifestIt->second;
        region.reason = region.policy == DatapathRefinementPolicy::Baseline
                            ? "policy manifest (baseline)"
                            : "policy manifest";
      } else if (region.violated &&
                 config.defaultPolicy != DatapathRefinementPolicy::Baseline) {
        region.policy = config.defaultPolicy;
        region.reason = "violating endpoint cone";
      } else if (!region.probes.empty() && config.pressureMarginLevels >= 0 &&
                 config.defaultPolicy != DatapathRefinementPolicy::Baseline) {
        bool pressured = false;
        for (Operation *member : members) {
          for (Value result : member->getResults()) {
            if (analysis.isPressured(result, config.pressureMarginLevels)) {
              pressured = true;
              break;
            }
          }
          if (pressured)
            break;
        }
        if (pressured) {
          region.policy = config.defaultPolicy;
          region.reason = "structural pressure";
        } else {
          region.reason = "relaxed (no pressure)";
        }
      } else if (region.probes.empty()) {
        region.reason = "no reachable probe";
      } else {
        region.reason = "relaxed (no pressure)";
      }
      plan.regions.push_back(std::move(region));
    }
  }

  // Manifest IDs that matched nothing.
  for (const auto &[id, policy] : config.manifest) {
    (void)policy;
    bool found = false;
    for (const auto &region : plan.regions) {
      if (region.id == id) {
        found = true;
        break;
      }
    }
    if (!found)
      plan.unknownManifestIds.push_back(id);
  }
  if (!plan.unknownManifestIds.empty()) {
    for (const auto &id : plan.unknownManifestIds)
      diagnose("datapath policy manifest: unknown region '" + id + "'");
    if (config.manifestStrict)
      diagnose("datapath policy manifest: strict mode forbids unknown "
               "region IDs");
  }
  return plan;
}

unsigned circt::synth::recordRefinementPlan(const RefinementPlan &plan) {
  unsigned numAnnotated = 0;
  auto record = [&](Operation *op, DatapathRefinementPolicy policy) {
    if (op->hasAttr(refinePolicyAttrName()))
      return;
    op->setAttr(
        refinePolicyAttrName(),
        StringAttr::get(op->getContext(), refinementPolicyName(policy)));
    ++numAnnotated;
  };
  // Phase 1: member datapath operations of selected regions.
  for (const auto &region : plan.regions) {
    if (region.policy == DatapathRefinementPolicy::Baseline)
      continue;
    for (Operation *member : region.members) {
      if (isDatapathMember(member))
        record(member, region.policy);
    }
  }
  // Phase 2: comb add/compare roots whose operand cone (through comb
  // arithmetic, datapath operations, and probes) reaches a recorded
  // operation. First recorded policy in deterministic DFS order wins.
  // Parent modules are visited in first-seen region order; roots in
  // block order.
  SmallVector<hw::HWModuleOp> parents;
  DenseSet<Operation *> parentSeen;
  for (const auto &region : plan.regions) {
    if (region.policy == DatapathRefinementPolicy::Baseline)
      continue;
    for (Operation *member : region.members) {
      auto module = member->getParentOfType<hw::HWModuleOp>();
      if (module && parentSeen.insert(module).second)
        parents.push_back(module);
    }
  }
  for (auto module : parents) {
    if (!module.getBodyBlock())
      continue;
    for (auto &op : module.getBodyBlock()->getOperations()) {
      if (!isa<comb::AddOp, comb::ICmpOp>(op))
        continue;
      Operation *root = &op;
      if (root->hasAttr(refinePolicyAttrName()))
        continue;
      // Bounded DFS through allowed operations.
      SmallVector<Value> worklist(root->getOperands().begin(),
                                  root->getOperands().end());
      DenseSet<Value> visited(worklist.begin(), worklist.end());
      std::optional<DatapathRefinementPolicy> found;
      unsigned steps = 0;
      constexpr unsigned kMaxSteps = 256;
      while (!worklist.empty() && steps++ < kMaxSteps && !found) {
        Value current = worklist.pop_back_val();
        Operation *defOp = current.getDefiningOp();
        if (!defOp)
          continue;
        if (auto policy = getRecordedRefinePolicy(defOp)) {
          found = policy;
          break;
        }
        if (isa<synth::TimingProbeOp>(defOp) ||
            isa<comb::CombDialect>(defOp->getDialect()) ||
            isa<datapath::DatapathDialect>(defOp->getDialect())) {
          for (Value operand : defOp->getOperands()) {
            if (visited.insert(operand).second)
              worklist.push_back(operand);
          }
        }
      }
      if (found)
        record(root, *found);
    }
  }
  return numAnnotated;
}

FailureOr<std::map<std::string, DatapathRefinementPolicy>>
circt::synth::parsePolicyManifest(StringRef jsonText, bool strict,
                                  function_ref<void(const Twine &)> diagnose) {
  std::map<std::string, DatapathRefinementPolicy> manifest;
  auto parsed = llvm::json::parse(jsonText);
  if (!parsed) {
    diagnose("datapath policy manifest: malformed JSON");
    return failure();
  }
  const auto *obj = parsed->getAsObject();
  if (!obj) {
    diagnose("datapath policy manifest: top-level JSON must be an object");
    return failure();
  }
  if (auto schema = obj->getString("schema")) {
    if (*schema != "circt.synth.datapath-policy.v1") {
      diagnose("datapath policy manifest: unsupported schema '" +
               schema->str() + "'");
      return failure();
    }
  }
  const auto *regions = obj->getArray("regions");
  if (!regions) {
    diagnose("datapath policy manifest: object has no 'regions' array");
    return failure();
  }
  for (const auto &entry : *regions) {
    const auto *regionObj = entry.getAsObject();
    if (!regionObj) {
      diagnose("datapath policy manifest: region entry is not an object");
      if (strict)
        return failure();
      continue;
    }
    auto id = regionObj->getString("id");
    auto policyName = regionObj->getString("policy");
    if (!id || !policyName) {
      diagnose("datapath policy manifest: region entry needs 'id' and "
               "'policy'");
      if (strict)
        return failure();
      continue;
    }
    auto policy = parseRefinementPolicy(*policyName);
    if (failed(policy)) {
      diagnose("datapath policy manifest: unknown policy '" +
               policyName->str() + "' for region '" + id->str() + "'");
      if (strict)
        return failure();
      continue;
    }
    manifest[id->str()] = *policy;
  }
  return manifest;
}

std::string
circt::synth::printRefinementReport(const RefinementPlan &plan,
                                    const TimingPressureAnalysis &analysis) {
  std::string out;
  llvm::raw_string_ostream os(out);
  auto quote = [&](const std::string &text) {
    os << '"';
    for (char c : text) {
      if (c == '"' || c == '\\')
        os << '\\';
      os << c;
    }
    os << '"';
  };
  os << "{\"schema\": \"circt.synth.datapath-refinement-report.v1\", "
        "\"regions\": [";
  bool first = true;
  for (const auto &region : plan.regions) {
    if (!first)
      os << ", ";
    first = false;
    os << "{\"id\": ";
    quote(region.id);
    os << ", \"module\": ";
    quote(region.module);
    os << ", \"policy\": \"" << refinementPolicyName(region.policy)
       << "\", \"reason\": ";
    quote(region.reason);
    os << ", \"violated\": " << (region.violated ? "true" : "false")
       << ", \"confidence\": \"";
    switch (region.confidence) {
    case TimingConfidence::STAFeedback:
      os << "STAFeedback";
      break;
    case TimingConfidence::LibraryEstimate:
      os << "LibraryEstimate";
      break;
    case TimingConfidence::StructuralEstimate:
      os << "StructuralEstimate";
      break;
    case TimingConfidence::Unknown:
      os << "Unknown";
      break;
    }
    os << "\", \"worst_slack_levels\": ";
    if (region.worstSlackLevels)
      os << *region.worstSlackLevels;
    else
      os << "null";
    os << ", \"probes\": [";
    bool firstProbe = true;
    for (const auto &probeId : region.probeIds) {
      if (!firstProbe)
        os << ", ";
      firstProbe = false;
      quote(probeId);
    }
    os << "], \"endpoints\": [";
    bool firstEndpoint = true;
    for (const auto &endpoint : region.endpoints) {
      if (!firstEndpoint)
        os << ", ";
      firstEndpoint = false;
      quote(endpoint);
    }
    os << "]}";
  }
  os << "], \"stats\": {\"regions\": " << plan.regions.size()
     << ", \"values_with_arrival\": " << analysis.getNumValuesWithArrival()
     << ", \"values_with_required\": " << analysis.getNumValuesWithRequired()
     << ", \"values_with_slack\": " << analysis.getNumValuesWithSlack()
     << ", \"unknown_delay_boundaries\": " << analysis.getNumUnknownBoundaries()
     << ", \"hierarchical_edges\": " << analysis.getNumHierarchicalEdges()
     << ", \"shared_contexts\": " << analysis.getNumSharedContexts()
     << ", \"feedback_resolved\": " << plan.numFeedbackResolved
     << ", \"feedback_violated\": " << plan.numFeedbackViolated
     << ", \"feedback_unresolved\": " << plan.unresolvedEndpoints.size()
     << ", \"manifest_unknown\": " << plan.unknownManifestIds.size() << "}}";
  return out;
}
