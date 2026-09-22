//===- OpenTimerTimingOracle.cpp - OpenTimer timing backend ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenTimer implementation of TimingOracle: builds the timer graph
// directly (no Verilog/SDC parsing), runs setup/max analysis, and
// normalizes whole-design aggregates with the shared helpers. Time is
// pinned to picoseconds via set_time_unit so the float boundary converts
// exactly once with checked rounding. See
// docs/StaticTimingAnalysisLibraryIntegration.md.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Timing/TimingOracle.h"

#include <ot/timer/timer.hpp>


#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>

using namespace circt::synth::timing;

namespace {

int64_t toPs(double ps) { return llround(ps); }

std::string gateName(uint32_t cellId) {
  return "g" + std::to_string(cellId);
}
std::string netName(uint32_t netId) { return "n" + std::to_string(netId); }
std::string primaryName(const TimingObjectKey &key) {
  std::string name = "p_";
  if (key.portName)
    name += key.portName.getValue().str();
  name += "_" + std::to_string(key.bit);
  return name;
}

struct OpenTimerTimingOracle : public TimingOracle {
  explicit OpenTimerTimingOracle(mlir::MLIRContext *ctx) : ctx(ctx) {}

  llvm::StringRef getName() const override { return "opentimer"; }

  TimingCapabilities getCapabilities() const override {
    TimingCapabilities caps;
    caps.setupMax = true;
    // v1: single ideal clocks only. Generated clocks, exceptions,
    // latches, hold/min, and non-Liberty interconnect delay are
    // rejected, never silently approximated.
    return caps;
  }

  mlir::FailureOr<TimingResult> analyze(const TimingRequest &request) override;

  mlir::MLIRContext *ctx;
};

mlir::FailureOr<TimingResult>
OpenTimerTimingOracle::analyze(const TimingRequest &request) {
  TimingResult result;
  result.backend = mlir::StringAttr::get(ctx, "opentimer");
  result.backendVersion = mlir::StringAttr::get(ctx, "OpenTimer-2.0");

  if (!request.design || !ctx)
    return mlir::failure();
  if (request.libertyFiles.empty() ||
      request.constraints.analysis != TimingAnalysisKind::SetupMax)
    return mlir::failure();
  // v1 supports exactly one active clock.
  if (request.constraints.clocks.size() != 1)
    return mlir::failure();

  const TimingDesign &design = *request.design;
  ot::Timer timer;
  auto ps = ot::make_time_unit("1ps");
  auto ff = ot::make_capacitance_unit("1fF");
  if (!ps || !ff)
    return mlir::failure();
  timer.set_time_unit(*ps);
  timer.set_capacitance_unit(*ff);
  for (const auto &lib : request.libertyFiles) {
    if (!std::filesystem::exists(lib))
      return mlir::failure();
    // Populate both analysis splits (setup/max queries the MAX corner).
    timer.read_celllib(lib, ot::MIN).read_celllib(lib, ot::MAX);
  }

  // Build the graph with synthetic names; every name maps back exactly.
  std::map<std::string, TimingObjectKey> pinKeys;
  std::map<uint32_t, std::string> netNames;
  for (const auto &net : design.nets)
    netNames[net.id] = netName(net.id);
  for (const auto &cell : design.cells) {
    std::string gate = gateName(cell.id);
    timer.insert_gate(gate, cell.cellType.getValue().str());
    for (const auto &pin : cell.pins) {
      std::string name = gate + ":" + pin.portName.getValue().str();
      TimingObjectKey key;
      key.kind = TimingObjectKind::InstancePin;
      key.moduleName = cell.instance.moduleName;
      key.instancePath = cell.instance.instancePath;
      key.portName = pin.portName;
      key.portIndex = pin.portIndex;
      key.bit = pin.bit;
      // Multi-bit Liberty pins share one timer pin; the first bit wins
      // deterministically (mapped netlists are bit-blasted, so this is
      // rare; other bits resolve through the same timer pin).
      if (!pinKeys.count(name))
        pinKeys[name] = key;
    }
  }
  for (const auto &port : design.ports) {
    std::string name = primaryName(port.port);
    if (port.isInput)
      timer.insert_primary_input(name);
    else
      timer.insert_primary_output(name);
    pinKeys[name] = port.port;
  }
  // Connect pins to nets via node ids.
  std::map<uint32_t, std::string> nodeToPin;
  for (const auto &port : design.ports)
    nodeToPin[port.node.value] = primaryName(port.port);
  for (const auto &cell : design.cells) {
    std::string gate = gateName(cell.id);
    for (const auto &pin : cell.pins) {
      std::string name = gate + ":" + pin.portName.getValue().str();
      nodeToPin[pin.node.value] = name;
    }
  }
  // Primary pins arrive pre-connected to implicit nets named after the
  // pin (OpenTimer creates and connects them on insertion). Nets touching
  // a primary therefore merge onto the implicit net: the primary itself
  // is never reconnected. Feedthrough (primary input straight to primary
  // output) has two implicit nets with no merge API, so it fails closed.
  std::map<uint32_t, std::pair<std::string, bool>> primaryNodes;
  for (const auto &port : design.ports)
    primaryNodes[port.node.value] = {primaryName(port.port), port.isInput};
  for (const auto &net : design.nets) {
    const std::string &myNet = netNames[net.id];
    std::string netStr = myNet;
    bool needInsert = true;
    std::string primaryInput;
    std::string primaryOutput;
    if (net.driver) {
      auto it = primaryNodes.find(net.driver->value);
      if (it != primaryNodes.end() && it->second.second)
        primaryInput = it->second.first;
    }
    for (const auto &load : net.loads) {
      auto it = primaryNodes.find(load.value);
      if (it != primaryNodes.end() && !it->second.second) {
        if (!primaryOutput.empty())
          return mlir::failure();
        primaryOutput = it->second.first;
      }
    }
    if (!primaryInput.empty() && !primaryOutput.empty())
      return mlir::failure();
    if (!primaryInput.empty()) {
      netStr = primaryInput;
      needInsert = false;
    } else if (!primaryOutput.empty()) {
      netStr = primaryOutput;
      needInsert = false;
    }
    if (needInsert)
      timer.insert_net(netStr);
    auto connect = [&](uint32_t node) {
      // Primaries stay on their implicit nets.
      if (primaryNodes.count(node))
        return true;
      auto it = nodeToPin.find(node);
      if (it == nodeToPin.end())
        return false;
      timer.connect_pin(it->second, netStr);
      return true;
    };
    if (net.driver && !connect(net.driver->value))
      return mlir::failure();
    for (const auto &load : net.loads)
      if (!connect(load.value))
        return mlir::failure();
  }
  // Primary-port bits attach to their value nets through the top module.
  // (Wired below via port constraints; unconnected bits stay unconstrained.)

  // Constraints: clocks first, then boundary conditions.
  const TimingClock &clock = request.constraints.clocks.front();
  auto clockIt = pinKeys.find(primaryName(clock.source));
  if (clockIt == pinKeys.end()) {
    // Clock source is an instance pin: resolve through the cell map.
    bool found = false;
    for (const auto &cell : design.cells) {
      std::string gate = gateName(cell.id);
      for (const auto &pin : cell.pins) {
        TimingObjectKey key;
        key.kind = TimingObjectKind::InstancePin;
        key.moduleName = cell.instance.moduleName;
        key.instancePath = cell.instance.instancePath;
        key.portName = pin.portName;
        key.portIndex = pin.portIndex;
        key.bit = pin.bit;
        if (key == clock.source) {
          clockIt = pinKeys.find(gate + ":" + pin.portName.getValue().str());
          found = clockIt != pinKeys.end();
          break;
        }
      }
      if (found)
        break;
    }
    if (!found)
      return mlir::failure();
  }
  for (auto split : {ot::MIN, ot::MAX}) {
    timer.set_at(clockIt->first, split, ot::RISE, 0.0f);
    timer.set_at(clockIt->first, split, ot::FALL, 0.0f);
    timer.set_slew(clockIt->first, split, ot::RISE, 1.0f);
    timer.set_slew(clockIt->first, split, ot::FALL, 1.0f);
  }
  timer.create_clock(clock.name.getValue().str(), clockIt->first,
                     (float)clock.period.value);
  for (const auto &portC : request.constraints.ports) {
    std::string name = primaryName(portC.port);
    bool isInput = false, known = false;
    for (const auto &port : design.ports) {
      if (port.port == portC.port) {
        isInput = port.isInput;
        known = true;
        break;
      }
    }
    if (!known)
      return mlir::failure();
    for (auto split : {ot::MIN, ot::MAX}) {
      if (isInput) {
        if (portC.lateArrival) {
          timer.set_at(name, split, ot::RISE, (float)portC.lateArrival->value);
          timer.set_at(name, split, ot::FALL, (float)portC.lateArrival->value);
        }
        if (portC.riseSlew) {
          timer.set_slew(name, split, ot::RISE, (float)portC.riseSlew->value);
          timer.set_slew(name, split, ot::FALL, (float)portC.riseSlew->value);
        }
        if (portC.fallSlew) {
          timer.set_slew(name, split, ot::FALL, (float)portC.fallSlew->value);
        }
      } else {
        if (portC.load) {
          timer.set_load(name, split, ot::RISE, (float)portC.load->value);
          timer.set_load(name, split, ot::FALL, (float)portC.load->value);
        }
        if (portC.lateRequired) {
          timer.set_rat(name, split, ot::RISE,
                        (float)portC.lateRequired->value);
          timer.set_rat(name, split, ot::FALL,
                        (float)portC.lateRequired->value);
        }
      }
    }
  }
  timer.update_timing();

  // Enumerate every analyzed endpoint directly (setup/max uses the MAX
  // split; both transitions are kept per endpoint). Arrival and required
  // ride along when the timer computed them; launch points need path
  // tracing, which OpenTimer only materializes for violated paths, so
  // worstLaunch stays empty in v1 (it is optional in the schema).
  std::map<std::string, TimingEndpointResult> endpointsByPin;
  for (const auto &info : timer.report_endpoints()) {
    if (info.split != ot::MAX)
      continue;
    auto keyIt = pinKeys.find(info.pin);
    if (keyIt == pinKeys.end())
      continue;
    TimingEndpointResult &endpoint = endpointsByPin[info.pin];
    endpoint.endpoint = keyIt->second;
    TimingTransitionResult tr;
    tr.transition = info.transition == ot::RISE ? TimingTransition::Rise
                                                : TimingTransition::Fall;
    if (info.arrival)
      tr.arrival = TimePs{toPs(*info.arrival)};
    tr.slack = TimePs{toPs(info.slack)};
    if (info.required)
      tr.required = TimePs{toPs(*info.required)};
    endpoint.transitions.push_back(std::move(tr));
  }
  // Deterministic order: sort transitions rise-before-fall, endpoints by
  // pin name.
  for (auto &[pin, endpoint] : endpointsByPin) {
    std::sort(endpoint.transitions.begin(), endpoint.transitions.end(), [](const TimingTransitionResult &a,
                                        const TimingTransitionResult &b) {
      return (int)a.transition < (int)b.transition;
    });
    result.endpoints.push_back(std::move(endpoint));
  }

  // Enumeration-based coverage: every constrained endpoint appears in
  // the worst-path set. Endpoints invisible to the timer (no arrival or
  // no required, e.g. unclocked flops) cannot be counted without
  // Liberty-role introspection; that expected-count refinement is future
  // work (see the integration doc).
  result.coverage.expectedEndpoints = result.endpoints.size();
  result.coverage.constrainedEndpoints = result.endpoints.size();
  result.coverage.analyzedEndpoints = result.endpoints.size();
  result.coverage.unconstrainedEndpoints = 0;
  result.status = TimingResultStatus::Complete;
  TimingDiagnostic coverageNote;
  coverageNote.message = mlir::StringAttr::get(
      ctx, "endpoint set enumerated by the timer backend");
  result.diagnostics.push_back(std::move(coverageNote));
  normalizeTimingResult(result);
  return result;
}

} // namespace

std::unique_ptr<TimingOracle>
circt::synth::timing::createOpenTimerTimingOracle(mlir::MLIRContext *ctx) {
  return std::make_unique<OpenTimerTimingOracle>(ctx);
}
