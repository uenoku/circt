//===- TimingFeedback.cpp - Timing feedback parsing and resolution -------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/TimingFeedback.h"
#include "circt/Dialect/Synth/Timing/TimingJSON.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include <cmath>

using namespace circt;
using namespace circt::synth;
using namespace llvm;

namespace {

// Unit multipliers to picoseconds for the versioned schema.
FailureOr<double> timeUnitToPs(StringRef unit) {
  if (unit == "ps")
    return 1.0;
  if (unit == "ns")
    return 1000.0;
  if (unit == "us")
    return 1e6;
  if (unit == "ms")
    return 1e9;
  if (unit == "s")
    return 1e12;
  return failure();
}

// Checked double -> int64 picoseconds conversion.
FailureOr<int64_t> toPs(double value, double unitMultiplier) {
  if (!std::isfinite(value) || !std::isfinite(unitMultiplier))
    return failure();
  double ps = value * unitMultiplier;
  if (!std::isfinite(ps) || ps < (double)std::numeric_limits<int64_t>::min() ||
      ps > (double)std::numeric_limits<int64_t>::max())
    return failure();
  return (int64_t)llround(ps);
}

struct RawPath {
  std::string startpoint;
  std::string endpoint;
  std::string pathGroup;
  std::optional<double> required;
  std::optional<double> arrival;
  std::optional<double> slack;
};

} // namespace

FailureOr<TimingFeedback>
TimingFeedback::parse(StringRef jsonText, StringRef expectedTop, bool strict,
                      function_ref<void(const Twine &)> diagnose) {
  TimingFeedback feedback;
  auto parsed = json::parse(jsonText);
  if (!parsed) {
    diagnose("timing feedback: malformed JSON");
    return failure();
  }

  // Collect raw records from either schema.
  std::vector<RawPath> raw;
  double unitMultiplier = 1000.0; // Legacy records are nanoseconds.
  if (const auto *list = parsed->getAsArray()) {
    // Legacy flat list.
    for (const auto &entry : *list) {
      const auto *obj = entry.getAsObject();
      if (!obj)
        return failure();
      RawPath path;
      if (auto start = obj->getString("startpoint"))
        path.startpoint = start->str();
      if (auto end = obj->getString("endpoint"))
        path.endpoint = end->str();
      if (auto group = obj->getString("path_group"))
        path.pathGroup = group->str();
      if (auto required = obj->getNumber("required_ns"))
        path.required = *required;
      if (auto arrival = obj->getNumber("arrival_ns"))
        path.arrival = *arrival;
      if (auto slack = obj->getNumber("slack_ns"))
        path.slack = *slack;
      raw.push_back(std::move(path));
    }
  } else if (const auto *obj = parsed->getAsObject()) {
    auto schema = obj->getString("schema");
    if (!schema || *schema != "circt.synth.timing-feedback.v1") {
      diagnose("timing feedback: unsupported schema '" +
               (schema ? schema->str() : std::string("<missing>")) + "'");
      return failure();
    }
    auto unit = obj->getString("time_unit");
    auto unitPs = timeUnitToPs(unit ? *unit : StringRef("ns"));
    if (failed(unitPs)) {
      diagnose("timing feedback: unsupported time unit '" +
               (unit ? unit->str() : std::string("<missing>")) + "'");
      return failure();
    }
    unitMultiplier = *unitPs;
    if (auto top = obj->getString("top"))
      feedback.top = top->str();
    if (!expectedTop.empty() && !feedback.top.empty() &&
        feedback.top != expectedTop) {
      diagnose("timing feedback: top '" + feedback.top +
               "' does not match expected '" + expectedTop + "'");
      if (strict)
        return failure();
      return feedback;
    }
    const auto *paths = obj->getArray("paths");
    if (!paths) {
      diagnose("timing feedback: schema object has no 'paths' array");
      return failure();
    }
    for (const auto &entry : *paths) {
      const auto *pathObj = entry.getAsObject();
      if (!pathObj) {
        diagnose("timing feedback: path entry is not an object");
        if (strict)
          return failure();
        ++feedback.numInvalid;
        continue;
      }
      RawPath path;
      if (auto start = pathObj->getString("startpoint"))
        path.startpoint = start->str();
      if (auto end = pathObj->getString("endpoint"))
        path.endpoint = end->str();
      if (auto group = pathObj->getString("path_group"))
        path.pathGroup = group->str();
      if (auto required = pathObj->getNumber("required"))
        path.required = *required;
      if (auto arrival = pathObj->getNumber("arrival"))
        path.arrival = *arrival;
      if (auto slack = pathObj->getNumber("slack"))
        path.slack = *slack;
      raw.push_back(std::move(path));
    }
  } else {
    diagnose("timing feedback: top-level JSON must be an object or array");
    return failure();
  }

  // Validate and convert each record.
  for (const auto &path : raw) {
    ++feedback.numRecords;
    auto invalid = [&](const Twine &reason) -> bool {
      diagnose("timing feedback: ignoring record for endpoint '" +
               path.endpoint + "': " + reason);
      ++feedback.numInvalid;
      return false;
    };
    if (path.endpoint.empty()) {
      if (!invalid("missing endpoint"))
        if (strict)
          return failure();
      continue;
    }
    if (!path.required) {
      if (!invalid("missing required time"))
        if (strict)
          return failure();
      continue;
    }
    auto requiredPs = toPs(*path.required, unitMultiplier);
    if (failed(requiredPs) || *requiredPs <= 0) {
      if (!invalid("non-finite or non-positive required time"))
        if (strict)
          return failure();
      continue;
    }
    std::optional<int64_t> arrivalPs;
    if (path.arrival) {
      auto converted = toPs(*path.arrival, unitMultiplier);
      if (failed(converted)) {
        if (!invalid("non-finite arrival time"))
          if (strict)
            return failure();
        continue;
      }
      arrivalPs = *converted;
    }
    std::optional<int64_t> slackPs;
    if (path.slack) {
      auto converted = toPs(*path.slack, unitMultiplier);
      if (failed(converted)) {
        if (!invalid("non-finite slack"))
          if (strict)
            return failure();
        continue;
      }
      slackPs = *converted;
    }
    // Consistency: required - arrival must agree with slack. Tolerance
    // covers STA report rounding (4-decimal ns ~= 0.1ps per field).
    if (arrivalPs && slackPs) {
      int64_t recomputed = *requiredPs - *arrivalPs;
      int64_t delta = std::abs(recomputed - *slackPs);
      int64_t tolerance =
          std::max<int64_t>(1, llround(0.001 * std::abs(*requiredPs)));
      if (delta > tolerance) {
        if (!invalid("inconsistent required/arrival/slack"))
          if (strict)
            return failure();
        continue;
      }
    }
    TimingFeedbackPath record;
    record.startpoint = path.startpoint;
    record.endpoint = path.endpoint;
    record.pathGroup = path.pathGroup;
    record.requiredPs = *requiredPs;
    record.arrivalPs = arrivalPs;
    record.slackPs = slackPs;
    feedback.paths.push_back(std::move(record));
  }
  feedback.numResolved = (unsigned)feedback.paths.size();
  return feedback;
}

namespace {

// Hierarchical instance lookup by path segments ("a/b/c"). Duplicate
// instance names at any level are ambiguous (null). Library cells are not
// descended into.
// Hierarchical instance lookup by path segments from a root design
// module (never instantiated by another design module). Duplicate
// instance names at any level are ambiguous (null). Library cells are
// not descended into.
hw::InstanceOp findInstance(mlir::ModuleOp top, ArrayRef<StringRef> segments) {
  auto isLibModule = [](hw::HWModuleOp module) {
    return module->hasAttr("synth.liberty.cell") ||
           module->hasAttr("synth.dont_use") ||
           module->hasAttr("synth.mapping_cost") ||
           module->hasAttr("synth.supergate_candidate") ||
           module.getModuleName().starts_with("__supergate_");
  };
  llvm::DenseSet<Operation *> hasDesignParent;
  for (auto module : top.getOps<hw::HWModuleOp>()) {
    if (isLibModule(module))
      continue;
    for (auto &block : module->getRegion(0))
      for (auto &op : block)
        if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
          auto *target = top.lookupSymbol(inst.getModuleName());
          auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
          if (targetModule && !isLibModule(targetModule))
            hasDesignParent.insert(target);
        }
  }
  SmallVector<hw::HWModuleOp, 2> worklist;
  for (auto module : top.getOps<hw::HWModuleOp>())
    if (!isLibModule(module) && !hasDesignParent.contains(module))
      worklist.push_back(module);
  // Deterministic order: source order is already deterministic; the scan
  // below visits parents in that order and takes the unique match.
  for (size_t level = 0; level < segments.size(); ++level) {
    hw::InstanceOp match;
    for (auto parent : worklist) {
      if (!parent.getBodyBlock())
        continue;
      for (auto &op : parent.getBodyBlock()->getOperations()) {
        auto inst = dyn_cast<hw::InstanceOp>(op);
        if (!inst || inst.getInstanceName() != segments[level])
          continue;
        if (match)
          return hw::InstanceOp(); // Ambiguous: fail closed.
        match = inst;
      }
    }
    if (!match)
      return hw::InstanceOp();
    if (level + 1 == segments.size())
      return match;
    auto *target = top.lookupSymbol(match.getModuleName());
    auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
    if (!targetModule || !targetModule.getBodyBlock())
      return hw::InstanceOp();
    worklist.clear();
    worklist.push_back(targetModule);
  }
  return hw::InstanceOp();
}

void splitPorts(hw::HWModuleOp module, SmallVector<hw::PortInfo> &inputs,
                SmallVector<hw::PortInfo> &outputs) {
  for (const auto &port : module.getPortList()) {
    if (port.isInput())
      inputs.push_back(port);
    else if (port.isOutput())
      outputs.push_back(port);
  }
}

// Input port index named "D" (flop data). -1 when absent.
int dataPinIndex(hw::HWModuleOp module) {
  SmallVector<hw::PortInfo> inputs, outputs;
  splitPorts(module, inputs, outputs);
  for (unsigned i = 0; i < inputs.size(); ++i)
    if (inputs[i].name.getValue() == "D")
      return (int)i;
  return -1;
}

} // namespace

FailureOr<ResolvedEndpoint>
circt::synth::resolveFeedbackEndpoint(mlir::ModuleOp top, StringRef endpoint) {
  SmallVector<StringRef> segments;
  endpoint.split(segments, '/');
  if (segments.size() < 2)
    return failure();
  // The final segment names the pin (".../inst:D" or ".../inst/D").
  StringRef pin = segments.back();
  segments.pop_back();
  if (pin == "D") {
    // ".../inst/D" form already split.
  } else if (auto colon = pin.find(':'); colon != StringRef::npos) {
    // ".../inst:D" form: split pin from instance name.
    segments.push_back(pin.substr(0, colon));
    pin = pin.substr(colon + 1);
  } else {
    // Bare ".../inst" defaults to the D input.
    segments.push_back(pin);
    pin = "D";
  }
  if (pin != "D")
    return failure(); // Clock pins and other pins are never endpoints.
  hw::InstanceOp inst = findInstance(top, segments);
  if (!inst)
    return failure();
  auto *target = top.lookupSymbol(inst.getModuleName());
  auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
  if (!targetModule)
    return failure();
  int dataIdx = dataPinIndex(targetModule);
  if (dataIdx < 0 || (unsigned)dataIdx >= inst.getNumOperands())
    return failure();
  return ResolvedEndpoint{inst, (unsigned)dataIdx};
}

FailureOr<Value> circt::synth::resolveFeedbackLaunch(mlir::ModuleOp top,
                                                     StringRef startpoint) {
  // Single segment: a top-level input port (unique across design modules).
  if (!startpoint.contains('/')) {
    Value found;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (!module.getBodyBlock())
        continue;
      unsigned idx = 0;
      for (const auto &port : module.getPortList()) {
        if (port.isInput() && port.name.getValue() == startpoint) {
          if (found)
            return failure(); // Ambiguous: fail closed.
          found = module.getBodyBlock()->getArgument(idx);
        }
        if (port.isInput())
          ++idx;
      }
    }
    if (!found)
      return failure();
    return found;
  }
  SmallVector<StringRef> segments;
  startpoint.split(segments, '/');
  if (segments.size() < 2)
    return failure();
  StringRef pin = segments.back();
  segments.pop_back();
  if (auto colon = pin.find(':'); colon != StringRef::npos) {
    segments.push_back(pin.substr(0, colon));
    pin = pin.substr(colon + 1);
  } else if (pin != "Q" && pin != "QN") {
    segments.push_back(pin);
    pin = "Q";
  }
  if (pin != "Q" && pin != "QN")
    return failure();
  hw::InstanceOp inst = findInstance(top, segments);
  if (!inst)
    return failure();
  auto *target = top.lookupSymbol(inst.getModuleName());
  auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
  if (!targetModule)
    return failure();
  SmallVector<hw::PortInfo> inputs, outputs;
  splitPorts(targetModule, inputs, outputs);
  // Prefer the requested pin; a bare instance path defaults to Q, then QN
  // (some cells expose only the inverted output).
  for (unsigned i = 0; i < outputs.size(); ++i)
    if (outputs[i].name.getValue() == pin)
      return i < inst.getNumResults() ? inst.getResult(i) : Value();
  if (pin == "Q") {
    for (unsigned i = 0; i < outputs.size(); ++i)
      if (outputs[i].name.getValue() == "QN")
        return i < inst.getNumResults() ? inst.getResult(i) : Value();
  }
  return failure();
}

StringRef circt::synth::describeResolveFailure(ResolveFailure failure) {
  switch (failure) {
  case ResolveFailure::Missing:
    return "missing";
  case ResolveFailure::Renamed:
    return "renamed";
  case ResolveFailure::Reordered:
    return "reordered";
  case ResolveFailure::WrongCell:
    return "wrong-cell";
  case ResolveFailure::WrongPinRole:
    return "wrong-pin-role";
  case ResolveFailure::Ambiguous:
    return "ambiguous";
  }
  llvm_unreachable("unknown resolve failure");
}

// Unique instance by name in one module body. Returns null when missing;
// sets ambiguous when duplicated.
static hw::InstanceOp findUniqueInstance(hw::HWModuleOp module, StringRef name,
                                         bool &ambiguous) {
  hw::InstanceOp match;
  if (auto *block = module.getBodyBlock()) {
    for (auto &op : block->getOperations()) {
      auto inst = dyn_cast<hw::InstanceOp>(op);
      if (!inst || inst.getInstanceName() != name)
        continue;
      if (match) {
        ambiguous = true;
        return hw::InstanceOp();
      }
      match = inst;
    }
  }
  return match;
}

FailureOr<ResolvedEndpoint> circt::synth::resolveFeedbackEndpointV2(
    mlir::ModuleOp top, mlir::StringAttr topName,
    const timing::TimingObjectKey &key, ResolveFailure &failureReason) {
  using timing::TimingObjectKind;
  if (key.kind != TimingObjectKind::InstancePin) {
    failureReason = ResolveFailure::WrongPinRole;
    return mlir::failure();
  }
  auto *topSym = top.lookupSymbol(topName);
  auto topModule = dyn_cast_or_null<hw::HWModuleOp>(topSym);
  if (!topModule || !topModule.getBodyBlock()) {
    failureReason = ResolveFailure::Missing;
    return mlir::failure();
  }
  hw::HWModuleOp current = topModule;
  hw::InstanceOp inst;
  for (size_t level = 0; level < key.instancePath.size(); ++level) {
    bool ambiguous = false;
    inst = findUniqueInstance(current, key.instancePath[level].getValue(),
                              ambiguous);
    if (ambiguous) {
      failureReason = ResolveFailure::Ambiguous;
      return mlir::failure();
    }
    if (!inst) {
      failureReason = ResolveFailure::Missing;
      return mlir::failure();
    }
    if (level + 1 == key.instancePath.size())
      break;
    auto *target = top.lookupSymbol(inst.getModuleName());
    auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
    if (!targetModule || !targetModule.getBodyBlock()) {
      failureReason = ResolveFailure::Missing;
      return mlir::failure();
    }
    current = targetModule;
  }
  if (!inst) {
    failureReason = ResolveFailure::Missing;
    return mlir::failure();
  }
  // Empty path means the endpoint is the top module itself: not probale.
  if (key.instancePath.empty()) {
    failureReason = ResolveFailure::WrongPinRole;
    return mlir::failure();
  }
  auto *target = top.lookupSymbol(inst.getModuleName());
  auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
  if (!targetModule ||
      targetModule.getModuleName() != key.moduleName.getValue()) {
    failureReason = ResolveFailure::WrongCell;
    return mlir::failure();
  }
  SmallVector<hw::PortInfo> inputs, outputs;
  splitPorts(targetModule, inputs, outputs);
  if (key.portIndex >= inputs.size()) {
    failureReason = ResolveFailure::Missing;
    return mlir::failure();
  }
  if (inputs[key.portIndex].name != key.portName) {
    bool foundElsewhere = false;
    for (const auto &input : inputs)
      if (input.name == key.portName)
        foundElsewhere = true;
    failureReason =
        foundElsewhere ? ResolveFailure::Reordered : ResolveFailure::Renamed;
    return mlir::failure();
  }
  auto portType = dyn_cast<IntegerType>(inputs[key.portIndex].type);
  if (!portType || key.bit >= (unsigned)portType.getWidth()) {
    failureReason = ResolveFailure::WrongPinRole;
    return mlir::failure();
  }
  if (key.portIndex >= inst.getNumOperands()) {
    failureReason = ResolveFailure::WrongPinRole;
    return mlir::failure();
  }
  return ResolvedEndpoint{inst, key.portIndex};
}

std::vector<circt::synth::TimingFeedbackPath>
circt::synth::toV1Paths(const timing::TimingFeedbackV2 &feedback) {
  std::vector<TimingFeedbackPath> paths;
  for (const auto &record : feedback.paths) {
    TimingFeedbackPath path;
    path.startpoint = record.startpoint.portName.str();
    path.endpoint = record.endpoint.portName.str();
    path.pathGroup = record.pathGroup.str();
    path.requiredPs = record.required.value;
    if (record.arrival)
      path.arrivalPs = record.arrival->value;
    if (record.slack)
      path.slackPs = record.slack->value;
    paths.push_back(std::move(path));
  }
  return paths;
}
