//===- TimingJSON.cpp - Timing result JSON serialization ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Timing/TimingJSON.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::timing;
using llvm::json::Array;
using llvm::json::Object;

namespace {

//--- Ordering -------------------------------------------------------------//

int compareKeys(const TimingObjectKey &a, const TimingObjectKey &b) {
  if (a.kind != b.kind)
    return (int)a.kind - (int)b.kind;
  size_t n = std::min(a.instancePath.size(), b.instancePath.size());
  for (size_t i = 0; i < n; ++i) {
    int c = a.instancePath[i].getValue().compare(b.instancePath[i].getValue());
    if (c != 0)
      return c;
  }
  if (a.instancePath.size() != b.instancePath.size())
    return a.instancePath.size() < b.instancePath.size() ? -1 : 1;
  int c = a.moduleName.getValue().compare(b.moduleName.getValue());
  if (c != 0)
    return c;
  if (a.portIndex != b.portIndex)
    return a.portIndex < b.portIndex ? -1 : 1;
  c = a.portName.getValue().compare(b.portName.getValue());
  if (c != 0)
    return c;
  if (a.bit != b.bit)
    return a.bit < b.bit ? -1 : 1;
  return 0;
}

//--- Writers --------------------------------------------------------------//

const char *transitionName(TimingTransition t) {
  return t == TimingTransition::Rise ? "rise" : "fall";
}

Object writeKey(const TimingObjectKey &key) {
  Object obj;
  obj["kind"] = key.kind == TimingObjectKind::Instance      ? "instance"
                : key.kind == TimingObjectKind::InstancePin ? "instance-pin"
                                                            : "top-port";
  Array path;
  for (auto part : key.instancePath)
    path.push_back(part.getValue().str());
  obj["instance_path"] = std::move(path);
  obj["module"] = key.moduleName.getValue().str();
  obj["port_index"] = (int64_t)key.portIndex;
  obj["port_name"] = key.portName.getValue().str();
  obj["bit"] = (int64_t)key.bit;
  return obj;
}

//--- Readers --------------------------------------------------------------//

struct Reader {
  mlir::MLIRContext *ctx;
  bool ok = true;

  mlir::StringAttr str(const llvm::json::Value *v, const char *what) {
    auto s = v ? v->getAsString() : std::nullopt;
    if (!s) {
      ok = false;
      return {};
    }
    return mlir::StringAttr::get(ctx, *s);
  }

  int64_t integer(const llvm::json::Value *v, const char *what) {
    auto i = v ? v->getAsInteger() : std::nullopt;
    if (!i) {
      ok = false;
      return 0;
    }
    return *i;
  }

  bool transition(llvm::StringRef s, TimingTransition &t) {
    if (s == "rise") {
      t = TimingTransition::Rise;
      return true;
    }
    if (s == "fall") {
      t = TimingTransition::Fall;
      return true;
    }
    ok = false;
    return false;
  }

  bool key(const llvm::json::Value *v, TimingObjectKey &key) {
    auto *obj = v ? v->getAsObject() : nullptr;
    if (!obj) {
      ok = false;
      return false;
    }
    auto kind = obj->getString("kind");
    if (!kind) {
      ok = false;
      return false;
    }
    if (*kind == "instance")
      key.kind = TimingObjectKind::Instance;
    else if (*kind == "instance-pin")
      key.kind = TimingObjectKind::InstancePin;
    else if (*kind == "top-port")
      key.kind = TimingObjectKind::TopPort;
    else {
      ok = false;
      return false;
    }
    auto *path = obj->getArray("instance_path");
    if (path) {
      for (const auto &part : *path) {
        auto s = part.getAsString();
        if (!s) {
          ok = false;
          return false;
        }
        key.instancePath.push_back(mlir::StringAttr::get(ctx, *s));
      }
    }
    auto module = obj->getString("module");
    auto portName = obj->getString("port_name");
    auto portIndex = obj->getInteger("port_index");
    auto bit = obj->getInteger("bit");
    if (!module || !portName || !portIndex || !bit) {
      ok = false;
      return false;
    }
    key.moduleName = mlir::StringAttr::get(ctx, *module);
    key.portName = mlir::StringAttr::get(ctx, *portName);
    key.portIndex = (unsigned)*portIndex;
    key.bit = (unsigned)*bit;
    return true;
  }
};

} // namespace

void circt::synth::timing::printTimingResult(const TimingResult &result,
                                             llvm::raw_ostream &os) {
  Object root;
  root["schema"] = kTimingResultSchema.str();
  root["time_unit"] = "ps";
  // Top is informational in v1; the object keys carry full identity.
  Object backend;
  backend["name"] = result.backend.getValue().str();
  backend["version"] = result.backendVersion.getValue().str();
  root["backend"] = std::move(backend);
  root["analysis"] = "setup-max";
  root["status"] = result.status == TimingResultStatus::Complete ? "complete"
                   : result.status == TimingResultStatus::Incomplete
                       ? "incomplete"
                       : "failed";
  Object summary;
  if (result.wns)
    summary["wns"] = result.wns->value;
  else
    summary["wns"] = nullptr;
  if (result.tns)
    summary["tns"] = result.tns->value;
  else
    summary["tns"] = nullptr;
  summary["failing_endpoints"] = (int64_t)result.failingEndpoints;
  root["summary"] = std::move(summary);

  Object coverage;
  coverage["expected_endpoints"] = (int64_t)result.coverage.expectedEndpoints;
  coverage["constrained_endpoints"] =
      (int64_t)result.coverage.constrainedEndpoints;
  coverage["analyzed_endpoints"] = (int64_t)result.coverage.analyzedEndpoints;
  coverage["unconstrained_endpoints"] =
      (int64_t)result.coverage.unconstrainedEndpoints;
  coverage["unsupported_endpoints"] =
      (int64_t)result.coverage.unsupportedEndpoints;
  root["coverage"] = std::move(coverage);

  // Deterministic endpoint order.
  SmallVector<const TimingEndpointResult *> endpoints;
  for (const auto &endpoint : result.endpoints)
    endpoints.push_back(&endpoint);
  llvm::sort(endpoints, [](const auto *a, const auto *b) {
    return compareKeys(a->endpoint, b->endpoint) < 0;
  });
  Array endpointsArray;
  for (const auto *endpoint : endpoints) {
    Object obj;
    obj["endpoint"] = writeKey(endpoint->endpoint);
    obj["path_group"] = endpoint->pathGroup.getValue().str();
    obj["worst_slack"] = endpoint->worstSlack.value;
    Array transitions;
    for (const auto &tr : endpoint->transitions) {
      Object trobj;
      trobj["transition"] = transitionName(tr.transition);
      trobj["arrival"] = tr.arrival.value;
      trobj["required"] = tr.required.value;
      trobj["slack"] = tr.slack.value;
      if (tr.worstLaunch)
        trobj["worst_launch"] = writeKey(*tr.worstLaunch);
      transitions.push_back(std::move(trobj));
    }
    obj["transitions"] = std::move(transitions);
    endpointsArray.push_back(std::move(obj));
  }
  root["endpoints"] = std::move(endpointsArray);

  // Deterministic path order: slack, endpoint, transition.
  SmallVector<const TimingPathResult *> paths;
  for (const auto &path : result.paths)
    paths.push_back(&path);
  llvm::sort(paths, [](const auto *a, const auto *b) {
    if (a->slack.value != b->slack.value)
      return a->slack.value < b->slack.value;
    int c = compareKeys(a->endpoint, b->endpoint);
    if (c != 0)
      return c < 0;
    return (int)a->transition < (int)b->transition;
  });
  Array pathsArray;
  for (const auto *path : paths) {
    Object obj;
    obj["endpoint"] = writeKey(path->endpoint);
    obj["transition"] = transitionName(path->transition);
    obj["arrival"] = path->arrival.value;
    obj["required"] = path->required.value;
    obj["slack"] = path->slack.value;
    Array points;
    if (!path->points.empty()) {
      for (const auto &point : path->points) {
        Object pobj;
        pobj["object"] = writeKey(point.object);
        pobj["transition"] = transitionName(point.transition);
        pobj["arrival"] = point.arrival.value;
        points.push_back(std::move(pobj));
      }
    }
    obj["points"] = std::move(points);
    pathsArray.push_back(std::move(obj));
  }
  root["paths"] = std::move(pathsArray);

  Array diagnostics;
  for (const auto &diag : result.diagnostics) {
    Object dobj;
    dobj["message"] = diag.message.getValue().str();
    if (diag.object)
      dobj["object"] = writeKey(*diag.object);
    diagnostics.push_back(std::move(dobj));
  }
  root["diagnostics"] = std::move(diagnostics);

  os << llvm::json::Value(std::move(root));
}

mlir::LogicalResult circt::synth::timing::parseTimingResult(
    llvm::StringRef json, TimingResult &result, mlir::MLIRContext *ctx) {
  auto parsed = llvm::json::parse(json);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return mlir::failure();
  }
  Reader reader{ctx};
  auto *root = parsed->getAsObject();
  if (!root)
    return mlir::failure();
  auto schema = root->getString("schema");
  if (!schema || *schema != kTimingResultSchema)
    return mlir::failure();

  result = TimingResult();
  if (auto *backend = root->getObject("backend")) {
    result.backend = reader.str(backend->get("name"), "backend.name");
    result.backendVersion =
        reader.str(backend->get("version"), "backend.version");
  } else {
    reader.ok = false;
  }
  auto status = root->getString("status");
  if (!status)
    reader.ok = false;
  else if (*status == "complete")
    result.status = TimingResultStatus::Complete;
  else if (*status == "incomplete")
    result.status = TimingResultStatus::Incomplete;
  else if (*status == "failed")
    result.status = TimingResultStatus::Failed;
  else
    reader.ok = false;

  if (auto *summary = root->getObject("summary")) {
    if (auto *wns = summary->get("wns")) {
      if (!wns->getAsNull())
        result.wns = TimePs{reader.integer(wns, "summary.wns")};
    }
    if (auto *tns = summary->get("tns")) {
      if (!tns->getAsNull())
        result.tns = TimePs{reader.integer(tns, "summary.tns")};
    }
    if (auto failing = summary->getInteger("failing_endpoints"))
      result.failingEndpoints = (uint64_t)*failing;
    else
      reader.ok = false;
  } else {
    reader.ok = false;
  }

  if (auto *coverage = root->getObject("coverage")) {
    auto get = [&](const char *name, uint64_t &out) {
      if (auto v = coverage->getInteger(name))
        out = (uint64_t)*v;
      else
        reader.ok = false;
    };
    get("expected_endpoints", result.coverage.expectedEndpoints);
    get("constrained_endpoints", result.coverage.constrainedEndpoints);
    get("analyzed_endpoints", result.coverage.analyzedEndpoints);
    get("unconstrained_endpoints", result.coverage.unconstrainedEndpoints);
    get("unsupported_endpoints", result.coverage.unsupportedEndpoints);
  } else {
    reader.ok = false;
  }

  if (auto *endpoints = root->getArray("endpoints")) {
    for (const auto &entry : *endpoints) {
      auto *obj = entry.getAsObject();
      if (!obj) {
        reader.ok = false;
        break;
      }
      TimingEndpointResult endpoint;
      if (auto *ep = obj->get("endpoint")) {
        if (!reader.key(ep, endpoint.endpoint))
          continue;
      } else {
        reader.ok = false;
        continue;
      }
      endpoint.pathGroup = reader.str(obj->get("path_group"), "path_group");
      if (auto worst = obj->getInteger("worst_slack"))
        endpoint.worstSlack.value = *worst;
      else
        reader.ok = false;
      if (auto *transitions = obj->getArray("transitions")) {
        for (const auto &trEntry : *transitions) {
          auto *trObj = trEntry.getAsObject();
          if (!trObj) {
            reader.ok = false;
            break;
          }
          TimingTransitionResult tr;
          auto transition = trObj->getString("transition");
          if (!transition || !reader.transition(*transition, tr.transition))
            continue;
          if (auto arrival = trObj->getInteger("arrival"))
            tr.arrival.value = *arrival;
          else {
            reader.ok = false;
            continue;
          }
          if (auto required = trObj->getInteger("required"))
            tr.required.value = *required;
          else {
            reader.ok = false;
            continue;
          }
          if (auto slack = trObj->getInteger("slack"))
            tr.slack.value = *slack;
          else {
            reader.ok = false;
            continue;
          }
          if (auto *launch = trObj->get("worst_launch")) {
            TimingObjectKey key;
            if (reader.key(launch, key))
              tr.worstLaunch = key;
          }
          endpoint.transitions.push_back(std::move(tr));
        }
      } else {
        reader.ok = false;
      }
      result.endpoints.push_back(std::move(endpoint));
    }
  }

  if (auto *paths = root->getArray("paths")) {
    for (const auto &entry : *paths) {
      auto *obj = entry.getAsObject();
      if (!obj) {
        reader.ok = false;
        break;
      }
      TimingPathResult path;
      if (auto *ep = obj->get("endpoint")) {
        if (!reader.key(ep, path.endpoint))
          continue;
      } else {
        reader.ok = false;
        continue;
      }
      auto transition = obj->getString("transition");
      if (!transition || !reader.transition(*transition, path.transition))
        continue;
      if (auto arrival = obj->getInteger("arrival"))
        path.arrival.value = *arrival;
      else {
        reader.ok = false;
        continue;
      }
      if (auto required = obj->getInteger("required"))
        path.required.value = *required;
      else {
        reader.ok = false;
        continue;
      }
      if (auto slack = obj->getInteger("slack"))
        path.slack.value = *slack;
      else {
        reader.ok = false;
        continue;
      }
      if (auto *points = obj->getArray("points")) {
        for (const auto &ptEntry : *points) {
          auto *ptObj = ptEntry.getAsObject();
          if (!ptObj) {
            reader.ok = false;
            break;
          }
          TimingPathPoint point;
          if (auto *o = ptObj->get("object")) {
            if (!reader.key(o, point.object))
              continue;
          } else {
            reader.ok = false;
            continue;
          }
          auto ptTransition = ptObj->getString("transition");
          if (!ptTransition ||
              !reader.transition(*ptTransition, point.transition))
            continue;
          if (auto arrival = ptObj->getInteger("arrival"))
            point.arrival.value = *arrival;
          else {
            reader.ok = false;
            continue;
          }
          path.points.push_back(std::move(point));
        }
      }
      result.paths.push_back(std::move(path));
    }
  }

  if (auto *diagnostics = root->getArray("diagnostics")) {
    for (const auto &entry : *diagnostics) {
      auto *obj = entry.getAsObject();
      if (!obj) {
        reader.ok = false;
        break;
      }
      TimingDiagnostic diag;
      diag.message = reader.str(obj->get("message"), "message");
      if (auto *o = obj->get("object")) {
        TimingObjectKey key;
        if (reader.key(o, key))
          diag.object = key;
      }
      result.diagnostics.push_back(std::move(diag));
    }
  }

  if (!reader.ok)
    return mlir::failure();
  return mlir::success();
}
