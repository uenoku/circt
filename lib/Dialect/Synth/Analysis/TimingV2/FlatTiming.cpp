//===- FlatTiming.cpp - Flat programmable timing framework ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/TimingV2/FlatTiming.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <queue>

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::timingv2;
using namespace mlir;

static size_t getBitWidth(Value value) {
  if (auto intType = dyn_cast<IntegerType>(value.getType()))
    return intType.getWidth();
  return 1;
}

static std::string getValueName(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    return ("arg" + Twine(arg.getArgNumber())).str();
  }

  auto *op = value.getDefiningOp();
  if (!op)
    return "value";
  if (auto name = op->getAttrOfType<StringAttr>("sv.namehint"))
    return name.getValue().str();
  if (auto name = op->getAttrOfType<StringAttr>("hw.name"))
    return name.getValue().str();
  return op->getName().stripDialect().str();
}

static std::string makeBitName(StringRef base, uint32_t bit) {
  return (base + "[" + Twine(bit) + "]").str();
}

static TimingDelayObjective toDelayObjective(TimingObjective objective) {
  return objective == TimingObjective::HoldMin ? TimingDelayObjective::Min
                                               : TimingDelayObjective::Max;
}

static DelayContext makeDelayContext(Operation *op, int32_t inputIndex,
                                     int32_t outputIndex,
                                     TimingObjective objective) {
  DelayContext ctx;
  ctx.op = op;
  ctx.inputIndex = inputIndex;
  ctx.outputIndex = outputIndex;
  ctx.objective = toDelayObjective(objective);
  if (op && inputIndex >= 0 &&
      static_cast<unsigned>(inputIndex) < op->getNumOperands())
    ctx.inputValue = op->getOperand(inputIndex);
  if (op && outputIndex >= 0 &&
      static_cast<unsigned>(outputIndex) < op->getNumResults())
    ctx.outputValue = op->getResult(outputIndex);
  return ctx;
}

static int64_t getDefaultOperationDelayCost(Operation *op) {
  if (!op)
    return 0;

  StringRef opName = op->getName().getStringRef();
  if (opName == "synth.aig.and_inv" || opName == "comb.and" ||
      opName == "comb.or" || opName == "comb.xor")
    return llvm::Log2_64_Ceil(op->getNumOperands());

  if (opName == "comb.mux" || opName == "comb.icmp" ||
      opName == "comb.truth_table")
    return 1;

  if (opName == "comb.extract" || opName == "comb.concat" ||
      opName == "comb.replicate" || opName == "hw.wire")
    return 0;

  return 1;
}

namespace {
class DefaultDelayModel final : public DelayModel {
public:
  DelayResult computeDelay(const DelayContext &ctx) const override {
    DelayResult result;
    result.delay = getDefaultOperationDelayCost(ctx.op);
    result.outputSlew = ctx.inputSlew;
    return result;
  }

  StringRef getName() const override { return "aig-level"; }
};
} // namespace

std::unique_ptr<DelayModel> circt::synth::timingv2::createDefaultDelayModel() {
  return std::make_unique<DefaultDelayModel>();
}

//===----------------------------------------------------------------------===//
// TimingNetwork
//===----------------------------------------------------------------------===//

void TimingNetwork::clear() {
  root = nullptr;
  delayModel = nullptr;
  ownedDefaultDelayModel.reset();
  delayModelName.clear();
  points.clear();
  arcs.clear();
  startPoints.clear();
  endPoints.clear();
  topologicalOrder.clear();
  reverseTopologicalOrder.clear();
  valueLookup.clear();
}

const TimingPoint *TimingNetwork::getPoint(TimingPointId id) const {
  if (id.index >= points.size())
    return nullptr;
  return &points[id.index];
}

TimingPoint *TimingNetwork::getPoint(TimingPointId id) {
  if (id.index >= points.size())
    return nullptr;
  return &points[id.index];
}

const TimingArc *TimingNetwork::getArc(uint32_t index) const {
  if (index >= arcs.size())
    return nullptr;
  return &arcs[index];
}

TimingPointId TimingNetwork::findValueBit(Value value, uint32_t bit) const {
  auto it = valueLookup.find({value, bit});
  if (it == valueLookup.end())
    return {};
  return it->second;
}

void TimingPropagationOptions::setRequiredTime(TimingPointId point,
                                               int64_t requiredTime) {
  requiredTimes.push_back({point, requiredTime});
}

LogicalResult TimingPropagationOptions::setRequiredTime(
    const TimingNetwork &network, Value value, uint32_t bit,
    int64_t requiredTime) {
  auto point = network.findValueBit(value, bit);
  if (!point.isValid())
    return failure();
  setRequiredTime(point, requiredTime);
  return success();
}

TimingPointId
TimingNetwork::getOrCreateValuePoint(Value value, uint32_t bit,
                                     TimingPointKind preferredKind,
                                     StringRef name, Operation *owner) {
  auto key = std::make_pair(value, bit);
  auto it = valueLookup.find(key);
  if (it != valueLookup.end())
    return it->second;

  TimingPointId id{static_cast<uint32_t>(points.size())};
  TimingPoint point;
  point.id = id;
  point.kind = preferredKind;
  point.value = value;
  point.bit = bit;
  point.owner = owner ? owner : value.getDefiningOp();
  point.name = name.empty() ? makeBitName(getValueName(value), bit) : name.str();

  if (point.isStartPoint())
    startPoints.push_back(id);
  if (point.isEndPoint())
    endPoints.push_back(id);
  valueLookup[key] = id;
  points.push_back(std::move(point));
  return id;
}

TimingPointId TimingNetwork::createBoundaryPoint(TimingPointKind kind,
                                                 Value value, uint32_t bit,
                                                 StringRef name,
                                                 Operation *owner) {
  TimingPointId id{static_cast<uint32_t>(points.size())};
  TimingPoint point;
  point.id = id;
  point.kind = kind;
  point.value = value;
  point.bit = bit;
  point.owner = owner;
  point.name = name.empty() ? makeBitName(getValueName(value), bit) : name.str();
  if (point.isStartPoint())
    startPoints.push_back(id);
  if (point.isEndPoint())
    endPoints.push_back(id);
  points.push_back(std::move(point));
  return id;
}

TimingPointId TimingNetwork::createSyntheticPoint(StringRef name,
                                                  Operation *owner) {
  TimingPointId id{static_cast<uint32_t>(points.size())};
  TimingPoint point;
  point.id = id;
  point.kind = TimingPointKind::Synthetic;
  point.owner = owner;
  point.name = name.str();
  points.push_back(std::move(point));
  return id;
}

uint32_t TimingNetwork::createArc(TimingPointId from, TimingPointId to,
                                  int64_t maxDelay, int64_t minDelay,
                                  Operation *op, int32_t inputIndex,
                                  int32_t outputIndex, TimingArcKind kind,
                                  StringRef token) {
  assert(from.index < points.size() && to.index < points.size() &&
         "invalid timing arc endpoints");
  uint32_t index = static_cast<uint32_t>(arcs.size());
  TimingArc arc;
  arc.index = index;
  arc.from = from;
  arc.to = to;
  arc.delay = maxDelay;
  arc.minDelay = minDelay;
  arc.op = op;
  arc.inputIndex = inputIndex;
  arc.outputIndex = outputIndex;
  arc.kind = kind;
  arc.token = token.str();
  arcs.push_back(std::move(arc));
  points[from.index].fanout.push_back(index);
  points[to.index].fanin.push_back(index);
  return index;
}

void TimingNetwork::removeArcsOwnedBy(Operation *op) {
  if (!op)
    return;

  SmallVector<TimingArc, 128> keptArcs;
  keptArcs.reserve(arcs.size());
  for (auto &arc : arcs)
    if (arc.op != op)
      keptArcs.push_back(std::move(arc));

  arcs.clear();
  for (auto &point : points) {
    point.fanin.clear();
    point.fanout.clear();
  }

  for (auto &arc : keptArcs) {
    arc.index = static_cast<uint32_t>(arcs.size());
    points[arc.from.index].fanout.push_back(arc.index);
    points[arc.to.index].fanin.push_back(arc.index);
    arcs.push_back(std::move(arc));
  }
}

void TimingNetwork::dropValuePointsOwnedBy(Operation *op) {
  if (!op)
    return;

  for (auto result : op->getResults()) {
    for (uint32_t bit = 0, e = getBitWidth(result); bit < e; ++bit) {
      auto it = valueLookup.find({result, bit});
      if (it == valueLookup.end())
        continue;
      auto *point = getPoint(it->second);
      valueLookup.erase(it);
      if (!point)
        continue;
      point->value = {};
      point->owner = nullptr;
      point->name = std::string("stale:") + point->name;
    }
  }
}

LogicalResult TimingNetwork::build(Operation *newRoot,
                                   const DelayModel *newDelayModel,
                                   const TimingSemanticsProvider
                                       *semanticsProvider) {
  clear();
  root = newRoot;
  if (!root)
    return failure();

  if (!newDelayModel) {
    ownedDefaultDelayModel = createDefaultDelayModel();
    newDelayModel = ownedDefaultDelayModel.get();
  }
  delayModel = newDelayModel;
  delayModelName = delayModel->getName().str();

  DefaultTimingSemanticsProvider defaultSemantics;
  const TimingSemanticsProvider &semantics =
      semanticsProvider ? *semanticsProvider : defaultSemantics;
  if (failed(processFlatRoot(semantics)))
    return failure();
  computeTopologicalOrder();
  return success();
}

static LogicalResult
processTimingOp(Operation *op, const TimingSemanticsProvider &semantics,
                TimingNetworkBuilder &builder);

LogicalResult TimingNetwork::processFlatRoot(
    const TimingSemanticsProvider &semanticsProvider) {
  TimingNetworkBuilder builder(*this, *delayModel);
  for (auto [operandIndex, operand] : llvm::enumerate(root->getOperands()))
    for (uint32_t bit = 0, e = getBitWidth(operand); bit < e; ++bit)
      getOrCreateValuePoint(
          operand, bit, TimingPointKind::RootInput,
          makeBitName(("operand" + Twine(operandIndex)).str(), bit), root);

  for (auto &region : root->getRegions())
    for (auto &block : region)
      for (auto argument : block.getArguments())
        for (uint32_t bit = 0, e = getBitWidth(argument); bit < e; ++bit)
          getOrCreateValuePoint(argument, bit, TimingPointKind::RootInput,
                                makeBitName(getValueName(argument), bit),
                                root);

  auto createRootResultOutputs = [&]() {
    for (auto [resultIndex, result] : llvm::enumerate(root->getResults())) {
      for (uint32_t bit = 0, e = getBitWidth(result); bit < e; ++bit) {
        auto from = getOrCreateValuePoint(result, bit, TimingPointKind::ValueBit,
                                          {}, root);
        auto to = createBoundaryPoint(
            TimingPointKind::RootOutput, result, bit,
            makeBitName(("result" + Twine(resultIndex)).str(), bit), root);
        createArc(from, to, 0, 0, root, -1,
                  static_cast<int32_t>(resultIndex), TimingArcKind::Boundary,
                  "root_output");
      }
    }
  };

  if (root->getNumRegions() == 0) {
    if (failed(processTimingOp(root, semanticsProvider, builder)))
      return failure();
    createRootResultOutputs();
    return success();
  }

  WalkResult result = root->walk([&](Operation *op) {
    if (op == root)
      return WalkResult::advance();
    if (failed(processTimingOp(op, semanticsProvider, builder)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  createRootResultOutputs();
  return result.wasInterrupted() ? failure() : success();
}

void TimingNetwork::computeTopologicalOrder() {
  topologicalOrder.clear();
  reverseTopologicalOrder.clear();

  SmallVector<unsigned, 64> indegree(points.size(), 0);
  std::queue<TimingPointId> queue;
  for (auto &point : points) {
    indegree[point.id.index] = point.fanin.size();
    if (point.fanin.empty())
      queue.push(point.id);
  }

  while (!queue.empty()) {
    auto id = queue.front();
    queue.pop();
    topologicalOrder.push_back(id);
    for (auto arcIndex : points[id.index].fanout) {
      const auto &arc = arcs[arcIndex];
      if (--indegree[arc.to.index] == 0)
        queue.push(arc.to);
    }
  }

  if (topologicalOrder.size() != points.size()) {
    llvm::DenseSet<uint32_t> seen;
    for (auto id : topologicalOrder)
      seen.insert(id.index);
    for (auto &point : points)
      if (!seen.contains(point.id.index))
        topologicalOrder.push_back(point.id);
  }

  reverseTopologicalOrder.assign(topologicalOrder.rbegin(),
                                 topologicalOrder.rend());
}

//===----------------------------------------------------------------------===//
// TimingNetworkBuilder
//===----------------------------------------------------------------------===//

TimingPointId TimingNetworkBuilder::getValueBit(Value value, uint32_t bit,
                                                TimingPointKind preferredKind,
                                                StringRef name,
                                                Operation *owner) {
  return network.getOrCreateValuePoint(value, bit, preferredKind, name, owner);
}

TimingPointId TimingNetworkBuilder::createBoundary(TimingPointKind kind,
                                                   Value value, uint32_t bit,
                                                   StringRef name,
                                                   Operation *owner) {
  return network.createBoundaryPoint(kind, value, bit, name, owner);
}

TimingPointId TimingNetworkBuilder::createSynthetic(StringRef name,
                                                    Operation *owner) {
  return network.createSyntheticPoint(name, owner);
}

uint32_t TimingNetworkBuilder::addArc(TimingPointId from, TimingPointId to,
                                      Operation *op, int32_t inputIndex,
                                      int32_t outputIndex, TimingArcKind kind,
                                      StringRef token) {
  int64_t maxDelay = 0;
  int64_t minDelay = 0;
  if (op) {
    auto ctx = makeDelayContext(op, inputIndex, outputIndex, objective);
    ctx.objective = TimingDelayObjective::Max;
    maxDelay = delayModel.computeDelay(ctx).delay;
    ctx.objective = TimingDelayObjective::Min;
    minDelay = delayModel.computeDelay(ctx).delay;
  }
  uint32_t arcIndex = network.createArc(from, to, maxDelay, minDelay, op,
                                        inputIndex, outputIndex, kind, token);
  return arcIndex;
}

uint32_t TimingNetworkBuilder::addArc(TimingPointId from, TimingPointId to,
                                      int64_t delay, Operation *op,
                                      int32_t inputIndex, int32_t outputIndex,
                                      TimingArcKind kind, StringRef token) {
  return network.createArc(from, to, delay, delay, op, inputIndex, outputIndex,
                           kind, token);
}

//===----------------------------------------------------------------------===//
// Timing semantics lowering
//===----------------------------------------------------------------------===//

static TimingPointId lowerSemanticPoint(TimingNetworkBuilder &builder,
                                        const TimingSemanticPoint &point,
                                        Operation *fallbackOwner) {
  Operation *owner = point.owner ? point.owner : fallbackOwner;
  switch (point.kind) {
  case TimingPointKind::RootOutput:
  case TimingPointKind::CutEnd:
    return builder.createBoundary(point.kind, point.value, point.bit,
                                  point.name, owner);
  case TimingPointKind::Synthetic:
    return builder.createSynthetic(point.name, owner);
  case TimingPointKind::ValueBit:
  case TimingPointKind::RootInput:
  case TimingPointKind::CutStart:
    return builder.getValueBit(point.value, point.bit, point.kind, point.name,
                               owner);
  }
  llvm_unreachable("unknown timing point kind");
}

static LogicalResult lowerTimingSemantics(const TimingSemantics &semantics,
                                          TimingNetworkBuilder &builder) {
  for (const auto &point : semantics.points)
    lowerSemanticPoint(builder, point, semantics.op);

  for (const auto &arc : semantics.arcs) {
    auto from = lowerSemanticPoint(builder, arc.from, semantics.op);
    auto to = lowerSemanticPoint(builder, arc.to, semantics.op);
    Operation *op = arc.op ? arc.op : semantics.op;
    if (arc.fixedDelay) {
      builder.addArc(from, to, *arc.fixedDelay, op, arc.inputIndex,
                     arc.outputIndex, arc.kind, arc.token);
      continue;
    }
    builder.addArc(from, to, op, arc.inputIndex, arc.outputIndex, arc.kind,
                   arc.token);
  }
  return success();
}

static LogicalResult
processTimingOp(Operation *op, const TimingSemanticsProvider &semantics,
                TimingNetworkBuilder &builder) {
  auto description = semantics.describe(op);
  if (failed(description))
    return failure();
  return lowerTimingSemantics(*description, builder);
}

//===----------------------------------------------------------------------===//
// Timing propagation
//===----------------------------------------------------------------------===//

TimingPropagationResult::TimingPropagationResult(const TimingNetwork &network)
    : network(network), states(network.getNumPoints()) {}

const TimingPointState *
TimingPropagationResult::getState(TimingPointId id) const {
  if (id.index >= states.size())
    return nullptr;
  return &states[id.index];
}

TimingPointState *TimingPropagationResult::getState(TimingPointId id) {
  if (id.index >= states.size())
    return nullptr;
  return &states[id.index];
}

static bool isBetterRequired(TimingObjective objective, int64_t candidate,
                             const TimingPointState &state) {
  if (!state.hasRequired)
    return true;
  return objective == TimingObjective::HoldMin ? candidate > state.required
                                               : candidate < state.required;
}

static void setRequiredIfBetter(TimingPropagationResult &result,
                                TimingPointId id,
                                TimingObjective objective,
                                int64_t requiredTime) {
  auto *state = result.getState(id);
  if (!state)
    return;
  if (!isBetterRequired(objective, requiredTime, *state))
    return;
  state->hasRequired = true;
  state->required = requiredTime;
}

FailureOr<TimingPropagationResult>
TimingPropagator::run(const TimingNetwork &network,
                      TimingPropagationOptions options) {
  TimingPropagationResult result(network);
  bool minMode = options.objective == TimingObjective::HoldMin;

  for (auto id : network.getStartPoints()) {
    auto *state = result.getState(id);
    state->hasArrival = true;
    state->arrival = 0;
  }

  for (auto id : network.getTopologicalOrder()) {
    auto *state = result.getState(id);
    if (!state || !state->hasArrival)
      continue;
    const auto *point = network.getPoint(id);
    if (!point)
      continue;
    for (auto arcIndex : point->fanout) {
      const auto *arc = network.getArc(arcIndex);
      auto *succ = result.getState(arc->to);
      int64_t arcDelay = minMode ? arc->minDelay : arc->delay;
      int64_t candidate = state->arrival + arcDelay;
      bool better = !succ->hasArrival ||
                    (minMode ? candidate < succ->arrival
                             : candidate > succ->arrival);
      if (!better)
        continue;
      succ->hasArrival = true;
      succ->arrival = candidate;
      succ->predecessorArc = arcIndex;
    }
  }

  for (auto id : network.getEndPoints()) {
    setRequiredIfBetter(result, id, options.objective,
                        options.defaultRequiredTime);
  }
  for (auto requiredTime : options.requiredTimes)
    setRequiredIfBetter(result, requiredTime.point, options.objective,
                        requiredTime.requiredTime);

  for (auto id : network.getReverseTopologicalOrder()) {
    auto *state = result.getState(id);
    const auto *point = network.getPoint(id);
    if (!state || !point)
      continue;
    for (auto arcIndex : point->fanout) {
      const auto *arc = network.getArc(arcIndex);
      auto *succ = result.getState(arc->to);
      if (!succ || !succ->hasRequired)
        continue;
      int64_t arcDelay = minMode ? arc->minDelay : arc->delay;
      int64_t candidate = succ->required - arcDelay;
      bool better = isBetterRequired(options.objective, candidate, *state);
      if (!better)
        continue;
      state->hasRequired = true;
      state->required = candidate;
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Dynamic timing semantics
//===----------------------------------------------------------------------===//

const TimingPointState *
TimingDynamicContext::getState(Value value, uint32_t bit) const {
  auto id = network.findValueBit(value, bit);
  if (!id.isValid())
    return nullptr;
  return propagation.getState(id);
}

FailureOr<int64_t> TimingDynamicContext::getArrival(Value value,
                                                    uint32_t bit) const {
  auto *state = getState(value, bit);
  if (!state || !state->hasArrival)
    return failure();
  return state->arrival;
}

FailureOr<int64_t> TimingDynamicContext::getRequired(Value value,
                                                     uint32_t bit) const {
  auto *state = getState(value, bit);
  if (!state || !state->hasRequired)
    return failure();
  return state->required;
}

FailureOr<int64_t> TimingDynamicContext::getSlack(Value value,
                                                  uint32_t bit) const {
  auto *state = getState(value, bit);
  if (!state || !state->hasArrival || !state->hasRequired)
    return failure();
  return state->required - state->arrival;
}

LogicalResult TimingNetwork::refineDynamicSemantics(
    const TimingDynamicSemanticsProvider &dynamicProvider,
    TimingPropagationOptions options) {
  if (!root || !delayModel)
    return failure();

  auto propagation = TimingPropagator::run(*this, options);
  if (failed(propagation))
    return failure();
  TimingDynamicContext context(*this, *propagation, options);

  struct DynamicUpdate {
    Operation *op = nullptr;
    TimingSemantics semantics;
  };
  SmallVector<DynamicUpdate, 8> updates;

  auto refineOp = [&](Operation *op) -> LogicalResult {
    if (!dynamicProvider.handles(op))
      return success();
    auto semantics = dynamicProvider.refine(op, context);
    if (failed(semantics))
      return failure();
    if (!semantics->op)
      semantics->op = op;
    updates.push_back({op, std::move(*semantics)});
    return success();
  };

  if (root->getNumRegions() == 0) {
    if (failed(refineOp(root)))
      return failure();
  } else {
    WalkResult walk = root->walk([&](Operation *op) {
      if (op == root)
        return WalkResult::advance();
      return failed(refineOp(op)) ? WalkResult::interrupt()
                                  : WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return failure();
  }

  TimingNetworkBuilder builder(*this, *delayModel, options.objective);
  for (auto &update : updates) {
    Operation *owner = update.semantics.op ? update.semantics.op : update.op;
    removeArcsOwnedBy(owner);
    if (failed(lowerTimingSemantics(update.semantics, builder)))
      return failure();
  }

  computeTopologicalOrder();
  return success();
}

//===----------------------------------------------------------------------===//
// Timing path reconstruction
//===----------------------------------------------------------------------===//

std::optional<ReconstructedTimingPath>
TimingPathReconstructor::reconstructTo(TimingPointId end) const {
  const auto *endState = result.getState(end);
  if (!endState || !endState->hasArrival)
    return std::nullopt;

  SmallVector<ReconstructedTimingStep, 16> reversed;
  TimingPointId current = end;
  while (current.isValid()) {
    const auto *state = result.getState(current);
    if (!state)
      return std::nullopt;
    uint32_t incoming = state->predecessorArc;
    reversed.push_back({current, incoming});
    if (incoming == UINT32_MAX)
      break;
    const auto *arc = network.getArc(incoming);
    if (!arc)
      return std::nullopt;
    current = arc->from;
  }

  ReconstructedTimingPath path;
  path.end = end;
  path.start = reversed.back().point;
  path.delay = endState->arrival;
  path.steps.assign(reversed.rbegin(), reversed.rend());
  return path;
}

//===----------------------------------------------------------------------===//
// Timing speculation
//===----------------------------------------------------------------------===//

static bool isBetterArrival(TimingObjective objective, int64_t candidate,
                            int64_t current) {
  return objective == TimingObjective::HoldMin ? candidate < current
                                               : candidate > current;
}

static FailureOr<TimingPointId>
findBestEndpoint(const TimingNetwork &network,
                 const TimingPropagationResult &result,
                 TimingObjective objective) {
  TimingPointId bestEnd;
  int64_t bestArrival = 0;
  for (auto id : network.getEndPoints()) {
    auto *state = result.getState(id);
    if (!state || !state->hasArrival)
      continue;
    if (bestEnd.isValid() &&
        !isBetterArrival(objective, state->arrival, bestArrival))
      continue;
    bestEnd = id;
    bestArrival = state->arrival;
  }
  if (!bestEnd.isValid())
    return failure();
  return bestEnd;
}

FailureOr<TimingSpeculationContext>
TimingSpeculationContext::create(const TimingNetwork &network,
                                 TimingPropagationOptions options) {
  auto result = TimingPropagator::run(network, options);
  if (failed(result))
    return failure();

  auto bestEnd = findBestEndpoint(network, *result, options.objective);
  if (failed(bestEnd))
    return failure();

  TimingPathReconstructor reconstructor(network, *result);
  auto path = reconstructor.reconstructTo(*bestEnd);
  if (!path)
    return failure();

  return TimingSpeculationContext(network, std::move(options),
                                  std::move(*result), std::move(*path));
}

FailureOr<int64_t>
TimingSpeculationContext::getArrival(Value value, uint32_t bit) const {
  auto id = network.findValueBit(value, bit);
  if (!id.isValid())
    return failure();
  auto *state = propagation.getState(id);
  if (!state || !state->hasArrival)
    return failure();
  return state->arrival;
}

bool TimingSpeculationContext::isOnWorstEndpointPath(Value value,
                                                     uint32_t bit) const {
  auto id = network.findValueBit(value, bit);
  if (!id.isValid())
    return false;
  return llvm::any_of(worstEndpointPath.steps,
                      [&](ReconstructedTimingStep step) {
                        return step.point == id;
                      });
}

FailureOr<int64_t> TimingSpeculationContext::speculateEndpointDelay(
    TimingPointId endpoint,
    ArrayRef<TimingArrivalReplacement> replacements) const {
  DenseMap<TimingPointId, int64_t> replacementArrivals;
  for (const auto &replacement : replacements) {
    auto point = network.findValueBit(replacement.value, replacement.bit);
    if (!point.isValid())
      return failure();
    auto *state = propagation.getState(point);
    if (!state || !state->hasArrival)
      return failure();
    replacementArrivals[point] = replacement.arrival;
  }

  TimingPathReconstructor reconstructor(network, propagation);
  auto path = reconstructor.reconstructTo(endpoint);
  if (!path)
    return failure();

  auto *endpointState = propagation.getState(endpoint);
  if (!endpointState || !endpointState->hasArrival)
    return failure();

  TimingPointId selectedPoint;
  int64_t selectedReplacementArrival = 0;
  for (auto step : path->steps) {
    auto replacement = replacementArrivals.find(step.point);
    if (replacement == replacementArrivals.end())
      continue;
    selectedPoint = step.point;
    selectedReplacementArrival = replacement->second;
  }

  if (!selectedPoint.isValid())
    return endpointState->arrival;

  auto *selectedState = propagation.getState(selectedPoint);
  if (!selectedState || !selectedState->hasArrival)
    return failure();
  return endpointState->arrival - selectedState->arrival +
         selectedReplacementArrival;
}

FailureOr<TimingEndpointSpeculation>
TimingSpeculationContext::speculateWorstEndpointDelay(
    ArrayRef<TimingArrivalReplacement> replacements) const {
  TimingEndpointSpeculation speculation;
  speculation.baselineDelay = worstEndpointPath.delay;
  speculation.predictedDelay = worstEndpointPath.delay;

  DenseMap<TimingPointId, int64_t> replacementArrivals;
  for (const auto &replacement : replacements) {
    auto point = network.findValueBit(replacement.value, replacement.bit);
    if (!point.isValid())
      return failure();
    auto *state = propagation.getState(point);
    if (!state || !state->hasArrival)
      return failure();
    replacementArrivals[point] = replacement.arrival;
  }

  std::optional<int64_t> predictedBestEndpoint;
  TimingPathReconstructor reconstructor(network, propagation);
  for (auto endpoint : network.getEndPoints()) {
    auto *endpointState = propagation.getState(endpoint);
    if (!endpointState || !endpointState->hasArrival)
      continue;

    auto path = reconstructor.reconstructTo(endpoint);
    if (!path)
      return failure();

    TimingPointId selectedPoint;
    int64_t selectedReplacementArrival = 0;
    for (auto step : path->steps) {
      auto replacement = replacementArrivals.find(step.point);
      if (replacement == replacementArrivals.end())
        continue;
      selectedPoint = step.point;
      selectedReplacementArrival = replacement->second;
    }

    int64_t predictedEndpoint = endpointState->arrival;
    if (selectedPoint.isValid()) {
      auto *selectedState = propagation.getState(selectedPoint);
      if (!selectedState || !selectedState->hasArrival)
        return failure();
      predictedEndpoint =
          endpointState->arrival - selectedState->arrival +
          selectedReplacementArrival;

      if (endpointState->arrival == worstEndpointPath.delay) {
        speculation.affectedWorstEndpointPath = true;
        speculation.affectedPoint = selectedPoint;
        speculation.oldArrival = selectedState->arrival;
        speculation.newArrival = selectedReplacementArrival;
      }
    }

    if (predictedBestEndpoint &&
        !isBetterArrival(options.objective, predictedEndpoint,
                         *predictedBestEndpoint))
      continue;
    predictedBestEndpoint = predictedEndpoint;
  }

  if (predictedBestEndpoint)
    speculation.predictedDelay = *predictedBestEndpoint;

  return speculation;
}

//===----------------------------------------------------------------------===//
// Timing reports
//===----------------------------------------------------------------------===//

static StringRef stringifyObjective(TimingObjective objective) {
  switch (objective) {
  case TimingObjective::SetupMax:
    return "setup_max";
  case TimingObjective::HoldMin:
    return "hold_min";
  }
  return "unknown";
}

static StringRef stringifyPointKind(TimingPointKind kind) {
  switch (kind) {
  case TimingPointKind::ValueBit:
    return "value";
  case TimingPointKind::RootInput:
    return "root_input";
  case TimingPointKind::RootOutput:
    return "root_output";
  case TimingPointKind::CutStart:
    return "cut_start";
  case TimingPointKind::CutEnd:
    return "cut_end";
  case TimingPointKind::Synthetic:
    return "synthetic";
  }
  return "unknown";
}

static StringRef stringifyArcKind(TimingArcKind kind) {
  switch (kind) {
  case TimingArcKind::Data:
    return "data";
  case TimingArcKind::Boundary:
    return "boundary";
  case TimingArcKind::Cut:
    return "cut";
  case TimingArcKind::Synthetic:
    return "synthetic";
  }
  return "unknown";
}

static int64_t getArcDelayForObjective(const TimingArc &arc,
                                       TimingObjective objective) {
  return objective == TimingObjective::HoldMin ? arc.minDelay : arc.delay;
}

static void printPointLabel(const TimingNetwork &network, TimingPointId id,
                            llvm::raw_ostream &os) {
  auto *point = network.getPoint(id);
  if (!point) {
    os << "<invalid:" << id.index << ">";
    return;
  }
  os << point->name << " (" << stringifyPointKind(point->kind) << ")";
}

FailureOr<ReconstructedTimingPath>
circt::synth::timingv2::reconstructCriticalPath(
    const TimingNetwork &network, TimingPropagationOptions options) {
  auto result = TimingPropagator::run(network, options);
  if (failed(result))
    return failure();

  auto bestEnd = findBestEndpoint(network, *result, options.objective);
  if (failed(bestEnd))
    return failure();

  TimingPathReconstructor reconstructor(network, *result);
  auto path = reconstructor.reconstructTo(*bestEnd);
  if (!path)
    return failure();
  return *path;
}

LogicalResult circt::synth::timingv2::printCriticalTimingReport(
    const TimingNetwork &network, llvm::raw_ostream &os,
    TimingPropagationOptions options) {
  auto result = TimingPropagator::run(network, options);
  if (failed(result))
    return failure();

  auto bestEnd = findBestEndpoint(network, *result, options.objective);
  if (failed(bestEnd))
    return failure();

  TimingPathReconstructor reconstructor(network, *result);
  auto path = reconstructor.reconstructTo(*bestEnd);
  if (!path)
    return failure();

  os << "TimingV2 critical path report\n";
  os << "objective: " << stringifyObjective(options.objective) << "\n";
  os << "delay: " << path->delay << "\n";
  os << "start: ";
  printPointLabel(network, path->start, os);
  os << "\nend: ";
  printPointLabel(network, path->end, os);
  os << "\npath:\n";

  for (auto step : path->steps) {
    auto *state = result->getState(step.point);
    os << "  ";
    printPointLabel(network, step.point, os);
    if (state && state->hasArrival)
      os << " arrival=" << state->arrival;
    os << "\n";

    if (step.incomingArc == UINT32_MAX)
      continue;
    auto *arc = network.getArc(step.incomingArc);
    if (!arc)
      continue;
    os << "    via " << (arc->token.empty() ? StringRef("<untagged>")
                                            : StringRef(arc->token))
       << " kind=" << stringifyArcKind(arc->kind)
       << " delay=" << getArcDelayForObjective(*arc, options.objective);
    if (arc->op)
      os << " op=" << arc->op->getName().getStringRef();
    os << "\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TimingRepairSession
//===----------------------------------------------------------------------===//

TimingRepairSession::TimingRepairSession(Operation *root,
                                         const DelayModel *delayModel,
                                         const TimingSemanticsProvider
                                             *semanticsProvider)
    : root(root), delayModel(delayModel), semanticsProvider(semanticsProvider) {}

LogicalResult TimingRepairSession::initialize() {
  network = std::make_unique<TimingNetwork>();
  if (failed(network->build(root, delayModel, semanticsProvider)))
    return failure();
  dirtyOps.clear();
  removedOps.clear();
  pendingChanges = false;
  fullRebuildRequired = false;
  initialized = true;
  return success();
}

LogicalResult TimingRepairSession::repair() {
  if (!initialized)
    return initialize();
  if (!pendingChanges)
    return success();

  if (!fullRebuildRequired && succeeded(repairLocalEdits()))
    return success();

  // If local repair cannot cover the edit set, rebuild conservatively.
  if (failed(initialize()))
    return failure();
  return success();
}

bool TimingRepairSession::canRepairLocally(Operation *op) const {
  if (!initialized || !network || !op || op == root)
    return false;
  if (op->getNumRegions() != 0)
    return false;

  DefaultTimingSemanticsProvider defaultSemantics;
  const TimingSemanticsProvider &semantics =
      semanticsProvider ? *semanticsProvider : defaultSemantics;
  auto description = semantics.describe(op);
  if (failed(description))
    return false;

  auto isLocalPoint = [](const TimingSemanticPoint &point) {
    switch (point.kind) {
    case TimingPointKind::ValueBit:
    case TimingPointKind::RootOutput:
    case TimingPointKind::CutStart:
    case TimingPointKind::CutEnd:
    case TimingPointKind::Synthetic:
      return true;
    case TimingPointKind::RootInput:
      return false;
    }
    llvm_unreachable("unknown timing point kind");
  };
  for (const auto &point : description->points)
    if (!isLocalPoint(point))
      return false;
  for (const auto &arc : description->arcs)
    if (!isLocalPoint(arc.from) || !isLocalPoint(arc.to))
      return false;

  return true;
}

LogicalResult TimingRepairSession::repairLocalEdits() {
  if (!network || !network->delayModel)
    return failure();

  DefaultTimingSemanticsProvider defaultSemantics;
  const TimingSemanticsProvider &semantics =
      semanticsProvider ? *semanticsProvider : defaultSemantics;

  TimingNetworkBuilder builder(*network, *network->delayModel);
  DenseSet<Operation *> removed;
  for (auto *op : removedOps) {
    if (!op || !removed.insert(op).second)
      continue;
    network->removeArcsOwnedBy(op);
  }

  DenseSet<Operation *> repairedOps;
  for (auto *op : dirtyOps) {
    if (!op || removed.contains(op) || !repairedOps.insert(op).second)
      continue;
    if (!canRepairLocally(op))
      return failure();
    network->removeArcsOwnedBy(op);
    if (failed(processTimingOp(op, semantics, builder)))
      return failure();
  }

  network->computeTopologicalOrder();
  dirtyOps.clear();
  removedOps.clear();
  pendingChanges = false;
  fullRebuildRequired = false;
  return success();
}

void TimingRepairSession::recordInserted(Operation *op) {
  if (!op)
    return;
  pendingChanges = true;
  if (!canRepairLocally(op)) {
    fullRebuildRequired = true;
    return;
  }
  dirtyOps.push_back(op);
}

void TimingRepairSession::recordModified(Operation *op) {
  if (!op)
    return;
  pendingChanges = true;
  if (!canRepairLocally(op)) {
    fullRebuildRequired = true;
    return;
  }
  dirtyOps.push_back(op);
}

void TimingRepairSession::recordAffectedUsers(Operation *op) {
  for (Value result : op->getResults()) {
    for (Operation *user : llvm::make_early_inc_range(result.getUsers())) {
      if (user == op)
        continue;
      dirtyOps.push_back(user);
      if (!canRepairLocally(user))
        fullRebuildRequired = true;
    }
  }
}

void TimingRepairSession::recordReplacement(Operation *op,
                                            ValueRange replacement) {
  if (!op)
    return;
  pendingChanges = true;
  if (!canRepairLocally(op))
    fullRebuildRequired = true;
  recordAffectedUsers(op);
  removedOps.push_back(op);
  if (network)
    network->dropValuePointsOwnedBy(op);
  for (Value value : replacement)
    if (auto *def = value.getDefiningOp())
      dirtyOps.push_back(def);
}

void TimingRepairSession::recordErasure(Operation *op) {
  if (!op)
    return;
  pendingChanges = true;
  if (!canRepairLocally(op))
    fullRebuildRequired = true;
  recordAffectedUsers(op);
  removedOps.push_back(op);
  if (network)
    network->dropValuePointsOwnedBy(op);
}

void TimingRepairSession::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
  recordInserted(op);
}

void TimingRepairSession::notifyOperationModified(Operation *op) {
  recordModified(op);
}

void TimingRepairSession::notifyOperationReplaced(Operation *op,
                                                  Operation *replacement) {
  recordReplacement(op, replacement ? replacement->getResults() : ValueRange());
}

void TimingRepairSession::notifyOperationReplaced(Operation *op,
                                                  ValueRange replacement) {
  recordReplacement(op, replacement);
}

void TimingRepairSession::notifyOperationErased(Operation *op) {
  recordErasure(op);
}
