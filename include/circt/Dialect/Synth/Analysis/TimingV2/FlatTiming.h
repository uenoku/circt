//===- FlatTiming.h - Flat programmable timing framework --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a flat, bit-level timing substrate. The core framework is
// intentionally dialect-agnostic: operations are described through generic
// timing semantics, and the network lowers those semantics into bit-level arcs.
// Dialect-specific knowledge belongs in TimingSemanticsProvider
// implementations, such as the default CIRCT provider.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGV2_FLATTIMING_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGV2_FLATTIMING_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace circt {
namespace synth {
namespace timingv2 {

/// Analysis objective for programmable delay models. Setup/max-delay analysis
/// asks for conservative maximum arc cost while hold/min-delay analysis asks
/// for conservative minimum arc cost.
enum class TimingDelayObjective : uint8_t { Max, Min };

/// Context passed to delay model for each arc computation.
struct DelayContext {
  /// Operation that owns the timing arc being costed.
  mlir::Operation *op = nullptr;
  /// Operand value driving the arc, if the arc corresponds to an operand.
  mlir::Value inputValue;
  /// Result value driven by the arc, if the arc corresponds to a result.
  mlir::Value outputValue;
  /// Operand/result ordinals used by per-pin delay models. `-1` means unknown
  /// or not applicable.
  int32_t inputIndex = -1;
  int32_t outputIndex = -1;
  /// Optional physical timing quantities. Scalar models may ignore these.
  double inputSlew = 0.0;
  double outputLoad = 0.0;
  /// Whether the caller is asking for max/setup or min/hold cost.
  TimingDelayObjective objective = TimingDelayObjective::Max;
};

/// Result of delay computation.
struct DelayResult {
  int64_t delay = 0;
  double outputSlew = 0.0;
};

/// Abstract base class for delay models.
class DelayModel {
public:
  virtual ~DelayModel() = default;

  /// Compute a scalar delay for one TimingV2 arc.
  virtual DelayResult computeDelay(const DelayContext &ctx) const = 0;
  /// Human-readable model name for reports.
  virtual llvm::StringRef getName() const = 0;
};

/// Create the default scalar delay model used by TimingV2.
std::unique_ptr<DelayModel> createDefaultDelayModel();

struct TimingPointId {
  uint32_t index = UINT32_MAX;

  /// Return true if this ID names a point in some TimingNetwork.
  bool isValid() const { return index != UINT32_MAX; }
  bool operator==(const TimingPointId &other) const {
    return index == other.index;
  }
  bool operator!=(const TimingPointId &other) const {
    return !(*this == other);
  }
};

enum class TimingPointKind : uint8_t {
  /// Bit of an SSA value.
  ValueBit,
  /// Root operand or region argument launch point.
  RootInput,
  /// Root result or output-like capture point.
  RootOutput,
  /// Sequential element output launch point.
  CutStart,
  /// Sequential element input capture point.
  CutEnd,
  /// Provider-authored virtual timing point.
  Synthetic,
};

enum class TimingArcKind : uint8_t {
  /// Ordinary data dependency arc.
  Data,
  /// Root input/output boundary arc.
  Boundary,
  /// Sequential cut arc.
  Cut,
  /// Provider-authored virtual timing arc.
  Synthetic,
};

enum class TimingObjective : uint8_t {
  SetupMax,
  HoldMin,
};

class TimingNetwork;

struct TimingRequiredTime {
  /// Point whose required time is constrained.
  TimingPointId point;
  /// Required arrival time at `point`.
  int64_t requiredTime = 0;
};

/// Options for arrival and required-time propagation.
struct TimingPropagationOptions {
  /// Select max/setup or min/hold propagation.
  TimingObjective objective = TimingObjective::SetupMax;
  /// Required time used for all end points unless a tighter explicit
  /// constraint applies.
  int64_t defaultRequiredTime = 0;
  /// Explicit per-point required-time constraints, typically supplied by a
  /// transform that wants to budget internal datapath unit outputs.
  llvm::SmallVector<TimingRequiredTime, 4> requiredTimes;

  /// Add a required-time constraint by point ID.
  void setRequiredTime(TimingPointId point, int64_t requiredTime);
  /// Add a required-time constraint for a value bit, returning failure if the
  /// bit is not present in `network`.
  mlir::LogicalResult setRequiredTime(const TimingNetwork &network,
                                      mlir::Value value, uint32_t bit,
                                      int64_t requiredTime);
};

/// A concrete node in the flat timing graph.
struct TimingPoint {
  /// Stable ID within the owning TimingNetwork.
  TimingPointId id;
  /// Role of this timing point.
  TimingPointKind kind = TimingPointKind::ValueBit;
  /// SSA value represented by this point, if any.
  mlir::Value value;
  /// Bit index within `value`.
  uint32_t bit = 0;
  /// Operation that conceptually owns this point.
  mlir::Operation *owner = nullptr;
  /// Human-readable label used in reports.
  std::string name;
  /// Incoming and outgoing arc indices in the owning TimingNetwork.
  llvm::SmallVector<uint32_t, 4> fanin;
  llvm::SmallVector<uint32_t, 4> fanout;

  /// Return true if propagation should seed this point with arrival zero.
  bool isStartPoint() const {
    return kind == TimingPointKind::RootInput ||
           kind == TimingPointKind::CutStart;
  }
  /// Return true if propagation should seed this point with the default
  /// required time.
  bool isEndPoint() const {
    return kind == TimingPointKind::RootOutput ||
           kind == TimingPointKind::CutEnd;
  }
};

/// A concrete directed edge in the flat timing graph.
struct TimingArc {
  /// Stable index within the owning TimingNetwork.
  uint32_t index = UINT32_MAX;
  /// Source and destination timing points.
  TimingPointId from;
  TimingPointId to;
  /// Structural role of the arc.
  TimingArcKind kind = TimingArcKind::Data;
  /// Max/setup delay.
  int64_t delay = 0;
  /// Min/hold delay.
  int64_t minDelay = 0;
  /// Operation that owns the arc. Repair and dynamic refinement use this to
  /// remove/rebuild an operation's timing behavior.
  mlir::Operation *op = nullptr;
  /// Operand/result ordinals represented by this arc, if known.
  int32_t inputIndex = -1;
  int32_t outputIndex = -1;
  /// Optional debug/path token used by reports and path reconstruction.
  std::string token;
};

class TimingNetworkBuilder;
class TimingDynamicSemanticsProvider;

/// A provider-side point description. The core lowers these into stable
/// TimingPointIds and owns the final graph storage.
struct TimingSemanticPoint {
  /// SSA value and bit described by the provider.
  mlir::Value value;
  uint32_t bit = 0;
  /// Requested point kind. Value-like kinds are uniqued by `(value, bit)`;
  /// boundary and synthetic kinds create standalone graph points.
  TimingPointKind kind = TimingPointKind::ValueBit;
  /// Optional report label. If empty, the network derives one.
  std::string name;
  /// Optional owner override for the lowered point.
  mlir::Operation *owner = nullptr;
};

/// A provider-side arc description. If `fixedDelay` is unset, the network
/// builder queries the DelayModel using `op`, `inputIndex`, and `outputIndex`.
/// If it is set, the arc is treated as structural or policy-authored cost and
/// bypasses the DelayModel.
struct TimingSemanticArc {
  /// Provider-side source and destination.
  TimingSemanticPoint from;
  TimingSemanticPoint to;
  /// Role of the lowered arc.
  TimingArcKind kind = TimingArcKind::Data;
  /// Optional fixed scalar cost. When absent, TimingNetworkBuilder asks the
  /// active DelayModel for max/min delay.
  std::optional<int64_t> fixedDelay;
  /// Optional owner override. Defaults to the containing TimingSemantics op.
  mlir::Operation *op = nullptr;
  /// Operand/result ordinals represented by this arc, if known.
  int32_t inputIndex = -1;
  int32_t outputIndex = -1;
  /// Optional debug/path token.
  std::string token;
};

/// Complete timing behavior for one operation. Implementations may emit no
/// points/arcs, direct value-bit arcs, cut boundaries, or synthetic virtual
/// points. The framework does not inspect the operation type after this point.
struct TimingSemantics {
  /// Operation whose behavior is described.
  mlir::Operation *op = nullptr;
  /// Additional points to materialize before arcs are lowered.
  llvm::SmallVector<TimingSemanticPoint, 4> points;
  /// Provider-authored arcs for this operation.
  llvm::SmallVector<TimingSemanticArc, 8> arcs;
};

/// Converts an operation into generic bit-level timing semantics.
///
/// This is the only extension point the flat core needs for operation-specific
/// behavior. A provider can wrap the default implementation, replace selected
/// operations, or define timing for an entirely different dialect/root form.
class TimingSemanticsProvider {
public:
  virtual ~TimingSemanticsProvider() = default;

  /// Return generic TimingV2 semantics for `op`. Returning failure aborts graph
  /// construction or local repair.
  virtual mlir::FailureOr<TimingSemantics>
  describe(mlir::Operation *op) const = 0;
};

/// Default CIRCT semantics provider.
///
/// This is the built-in policy for common CIRCT operations. It lives in a
/// separate implementation file so the TimingV2 core remains independent of
/// concrete dialect operation classes. Users can replace this provider to
/// override any operation, including plain value transforms, datapath-style
/// virtual units, calls, memories, or custom dialects.
class DefaultTimingSemanticsProvider : public TimingSemanticsProvider {
public:
  /// Describe common CIRCT operations and fall back to same-bit data arcs for
  /// unknown value-transforming operations.
  mlir::FailureOr<TimingSemantics>
  describe(mlir::Operation *op) const override;
};

/// Owns a flat bit-level timing graph rooted at one MLIR operation.
///
/// Point IDs and arc indices are stable until the network is rebuilt or repaired
/// in a way that removes/recreates graph storage. TimingNetwork is dialect
/// agnostic after TimingSemantics are lowered.
class TimingNetwork {
public:
  TimingNetwork() = default;

  /// Build a fresh flat graph rooted at `root`.
  ///
  /// If `delayModel` is null, TimingV2 creates and owns a default scalar model.
  /// If `semanticsProvider` is null, DefaultTimingSemanticsProvider is used.
  mlir::LogicalResult build(
      mlir::Operation *root, const DelayModel *delayModel = nullptr,
      const TimingSemanticsProvider *semanticsProvider = nullptr);
  /// Run one propagation pass, ask `dynamicProvider` for replacement semantics
  /// for handled ops, and rebuild those ops' arcs in place.
  mlir::LogicalResult refineDynamicSemantics(
      const TimingDynamicSemanticsProvider &dynamicProvider,
      TimingPropagationOptions options = {});

  /// Return the root operation used for the current graph.
  mlir::Operation *getRoot() const { return root; }
  /// Return the active delay model name for reports.
  llvm::StringRef getDelayModelName() const { return delayModelName; }

  /// Return graph size.
  size_t getNumPoints() const { return points.size(); }
  size_t getNumArcs() const { return arcs.size(); }

  /// Look up graph storage by ID/index. Returns null for invalid IDs.
  const TimingPoint *getPoint(TimingPointId id) const;
  TimingPoint *getPoint(TimingPointId id);
  const TimingArc *getArc(uint32_t index) const;

  /// Return graph storage and propagation boundary sets.
  llvm::ArrayRef<TimingPoint> getPoints() const { return points; }
  llvm::ArrayRef<TimingArc> getArcs() const { return arcs; }
  llvm::ArrayRef<TimingPointId> getStartPoints() const { return startPoints; }
  llvm::ArrayRef<TimingPointId> getEndPoints() const { return endPoints; }
  /// Return forward and reverse topological traversal orders used by
  /// propagation. Cyclic leftovers, if any, are appended deterministically.
  llvm::ArrayRef<TimingPointId> getTopologicalOrder() const {
    return topologicalOrder;
  }
  llvm::ArrayRef<TimingPointId> getReverseTopologicalOrder() const {
    return reverseTopologicalOrder;
  }

  /// Find the canonical graph point for an SSA value bit. Returns an invalid ID
  /// if the value bit is not materialized in this network.
  TimingPointId findValueBit(mlir::Value value, uint32_t bit) const;

private:
  friend class TimingNetworkBuilder;
  friend class TimingRepairSession;

  TimingPointId getOrCreateValuePoint(mlir::Value value, uint32_t bit,
                                      TimingPointKind preferredKind,
                                      llvm::StringRef name,
                                      mlir::Operation *owner);
  TimingPointId createBoundaryPoint(TimingPointKind kind, mlir::Value value,
                                    uint32_t bit, llvm::StringRef name,
                                    mlir::Operation *owner);
  TimingPointId createSyntheticPoint(llvm::StringRef name,
                                     mlir::Operation *owner);
  uint32_t createArc(TimingPointId from, TimingPointId to, int64_t maxDelay,
                     int64_t minDelay, mlir::Operation *op,
                     int32_t inputIndex, int32_t outputIndex,
                     TimingArcKind kind, llvm::StringRef token);
  void removeArcsOwnedBy(mlir::Operation *op);
  void dropValuePointsOwnedBy(mlir::Operation *op);
  void clear();
  void computeTopologicalOrder();
  mlir::LogicalResult
  processFlatRoot(const TimingSemanticsProvider &semanticsProvider);

  mlir::Operation *root = nullptr;
  const DelayModel *delayModel = nullptr;
  std::unique_ptr<DelayModel> ownedDefaultDelayModel;
  std::string delayModelName;

  llvm::SmallVector<TimingPoint, 64> points;
  llvm::SmallVector<TimingArc, 128> arcs;
  llvm::SmallVector<TimingPointId, 16> startPoints;
  llvm::SmallVector<TimingPointId, 16> endPoints;
  llvm::SmallVector<TimingPointId, 64> topologicalOrder;
  llvm::SmallVector<TimingPointId, 64> reverseTopologicalOrder;
  llvm::DenseMap<std::pair<mlir::Value, uint32_t>, TimingPointId> valueLookup;
};

class TimingNetworkBuilder {
public:
  /// Create a lowering helper for provider-authored TimingSemantics.
  TimingNetworkBuilder(TimingNetwork &network, const DelayModel &delayModel,
                       TimingObjective objective = TimingObjective::SetupMax)
      : network(network), delayModel(delayModel), objective(objective) {}

  /// Get or create a value-bit point. Value-bit-like points are uniqued by
  /// `(value, bit)`.
  TimingPointId getValueBit(mlir::Value value, uint32_t bit,
                            TimingPointKind preferredKind =
                                TimingPointKind::ValueBit,
                            llvm::StringRef name = {},
                            mlir::Operation *owner = nullptr);
  /// Create a standalone boundary point.
  TimingPointId createBoundary(TimingPointKind kind, mlir::Value value,
                               uint32_t bit, llvm::StringRef name,
                               mlir::Operation *owner);
  /// Create a standalone synthetic point.
  TimingPointId createSynthetic(llvm::StringRef name, mlir::Operation *owner);

  /// Add an arc whose max/min delays are computed by the active DelayModel.
  uint32_t addArc(TimingPointId from, TimingPointId to, mlir::Operation *op,
                  int32_t inputIndex, int32_t outputIndex,
                  TimingArcKind kind = TimingArcKind::Data,
                  llvm::StringRef token = {});
  /// Add an arc with the same fixed delay for max/setup and min/hold.
  uint32_t addArc(TimingPointId from, TimingPointId to, int64_t delay,
                  mlir::Operation *op = nullptr, int32_t inputIndex = -1,
                  int32_t outputIndex = -1,
                  TimingArcKind kind = TimingArcKind::Data,
                  llvm::StringRef token = {});

  /// Return the delay model and objective used while lowering semantics.
  const DelayModel &getDelayModel() const { return delayModel; }
  TimingObjective getObjective() const { return objective; }

private:
  TimingNetwork &network;
  const DelayModel &delayModel;
  TimingObjective objective;
};

/// Per-point propagation state.
struct TimingPointState {
  /// Arrival state and predecessor arc used for path reconstruction.
  bool hasArrival = false;
  int64_t arrival = 0;
  uint32_t predecessorArc = UINT32_MAX;
  /// Required-time state.
  bool hasRequired = false;
  int64_t required = 0;
};

/// Result of one propagation run over a TimingNetwork.
class TimingPropagationResult {
public:
  /// Allocate one state slot per point in `network`.
  explicit TimingPropagationResult(const TimingNetwork &network);

  /// Return the network this result belongs to.
  const TimingNetwork &getNetwork() const { return network; }
  /// Look up mutable/immutable point state. Returns null for invalid IDs.
  const TimingPointState *getState(TimingPointId id) const;
  TimingPointState *getState(TimingPointId id);
  /// Return all point states in point-ID order.
  llvm::ArrayRef<TimingPointState> getStates() const { return states; }

private:
  const TimingNetwork &network;
  llvm::SmallVector<TimingPointState, 64> states;
};

class TimingPropagator {
public:
  /// Compute arrival and required times over `network`.
  ///
  /// SetupMax propagates maximum arrivals and tightest minimum required times.
  /// HoldMin propagates minimum arrivals and the corresponding reverse bound.
  static mlir::FailureOr<TimingPropagationResult>
  run(const TimingNetwork &network, TimingPropagationOptions options = {});
};

/// Query context passed to dynamic timing policies after an initial propagation
/// over the current graph. This lets datapath policies inspect input arrivals,
/// output required times, and objective options before replacing an operation's
/// timing arcs.
class TimingDynamicContext {
public:
  TimingDynamicContext(const TimingNetwork &network,
                       const TimingPropagationResult &propagation,
                       TimingPropagationOptions options)
      : network(network), propagation(propagation), options(options) {}

  /// Return the graph, propagation result, and options used to create this
  /// context.
  const TimingNetwork &getNetwork() const { return network; }
  const TimingPropagationResult &getPropagation() const { return propagation; }
  TimingPropagationOptions getOptions() const { return options; }

  /// Query propagated state by value bit. Returns null/failure if the value bit
  /// is not present or the requested quantity was not computed.
  const TimingPointState *getState(mlir::Value value, uint32_t bit) const;
  mlir::FailureOr<int64_t> getArrival(mlir::Value value, uint32_t bit) const;
  mlir::FailureOr<int64_t> getRequired(mlir::Value value, uint32_t bit) const;
  /// Return `required - arrival` for a value bit.
  mlir::FailureOr<int64_t> getSlack(mlir::Value value, uint32_t bit) const;

private:
  const TimingNetwork &network;
  const TimingPropagationResult &propagation;
  TimingPropagationOptions options;
};

/// Optional second-phase operation policy. Static TimingSemanticsProvider
/// implementations describe an initial graph. Dynamic providers can then
/// replace selected operations' arcs after arrival and required-time propagation
/// has produced the context needed by scheduling-sensitive datapath models.
class TimingDynamicSemanticsProvider {
public:
  virtual ~TimingDynamicSemanticsProvider() = default;

  /// Return true when this provider wants a second-phase refinement callback
  /// for `op`.
  virtual bool handles(mlir::Operation *op) const = 0;
  /// Return replacement semantics for `op` using propagated timing context.
  /// The network removes previous arcs owned by `op` before lowering the
  /// returned semantics.
  virtual mlir::FailureOr<TimingSemantics>
  refine(mlir::Operation *op, const TimingDynamicContext &context) const = 0;
};

/// Tunable delays and policy knobs for DatapathTimingSemanticsProvider.
struct DatapathTimingSemanticsOptions {
  /// Per-stage delay used by the fast compressor policy.
  int64_t fastCompressorStageDelay = 1;
  /// Per-stage delay used by the area-oriented compressor policy.
  int64_t areaCompressorStageDelay = 2;
  /// Delay assigned to compact partial-product virtual arcs.
  int64_t partialProductDelay = 1;
  /// Prefer area-oriented compressor arcs when propagated slack can absorb
  /// their delay; otherwise choose fast arcs.
  bool preferAreaWhenSlackAllows = true;
};

/// Dynamic timing policy for datapath dialect operations.
///
/// This provider replaces coarse seed arcs for datapath units with compact
/// virtual arcs. Compressor timing builds an arrival-greedy compression
/// schedule that uses 4:2 compressors when possible and 3:2 full adders
/// otherwise, models carry movement across bit columns, and uses required times
/// to choose an area-oriented or fast policy. Partial-product operations use a
/// compact bit-sensitive model instead of dense carry-prefix fallback arcs.
class DatapathTimingSemanticsProvider final
    : public TimingDynamicSemanticsProvider {
public:
  DatapathTimingSemanticsProvider() = default;
  /// Create a datapath provider with explicit policy options.
  explicit DatapathTimingSemanticsProvider(
      DatapathTimingSemanticsOptions options)
      : options(options) {}

  /// Handle datapath compressor and partial-product operations.
  bool handles(mlir::Operation *op) const override;
  /// Emit compact virtual datapath arcs using the current dynamic context.
  mlir::FailureOr<TimingSemantics>
  refine(mlir::Operation *op, const TimingDynamicContext &context) const override;

private:
  DatapathTimingSemanticsOptions options;
};

/// One step in a reconstructed timing path.
struct ReconstructedTimingStep {
  /// Point reached by this step.
  TimingPointId point;
  /// Arc used to enter `point`, or UINT32_MAX for the path start.
  uint32_t incomingArc = UINT32_MAX;
};

/// Reconstructed point/arc sequence for one propagated path.
struct ReconstructedTimingPath {
  /// First and last points in the path.
  TimingPointId start;
  TimingPointId end;
  /// Arrival at `end` under the propagation objective.
  int64_t delay = 0;
  /// Ordered steps from start to end.
  llvm::SmallVector<ReconstructedTimingStep, 16> steps;
};

/// Reconstruct paths from predecessor arcs stored in a propagation result.
class TimingPathReconstructor {
public:
  /// Create a reconstructor for `network` and its matching propagation result.
  TimingPathReconstructor(const TimingNetwork &network,
                          const TimingPropagationResult &result)
      : network(network), result(result) {}

  /// Reconstruct the selected arrival path ending at `end`.
  std::optional<ReconstructedTimingPath> reconstructTo(TimingPointId end) const;

private:
  const TimingNetwork &network;
  const TimingPropagationResult &result;
};

/// One side-effect-free arrival replacement used for timing-guided rewrite
/// speculation. A transform computes the predicted arrival for a value bit after
/// a candidate rewrite, without first mutating IR, and asks TimingV2 to estimate
/// the impact on endpoint arrivals.
struct TimingArrivalReplacement {
  /// Value bit whose arrival would change.
  mlir::Value value;
  uint32_t bit = 0;
  /// Speculated arrival time for the value bit after the candidate rewrite.
  int64_t arrival = 0;
};

/// Estimated effect of a set of side-effect-free arrival replacements on the
/// current worst endpoint under the active objective.
struct TimingEndpointSpeculation {
  /// Worst endpoint delay before applying replacements.
  int64_t baselineDelay = 0;
  /// Estimated worst endpoint delay after applying replacements to every endpoint
  /// path that contains a replaced point. Unaffected endpoints keep their
  /// current propagated arrivals.
  int64_t predictedDelay = 0;
  /// True when at least one replacement matched a point on a current worst
  /// endpoint path.
  bool affectedWorstEndpointPath = false;
  /// Latest replaced point on a current worst endpoint path. Invalid when
  /// `affectedWorstEndpointPath` is false.
  TimingPointId affectedPoint;
  /// Old and new arrivals at `affectedPoint`.
  int64_t oldArrival = 0;
  int64_t newArrival = 0;
};

/// Cached propagation and endpoint query context for side-effect-free
/// transform speculation.
///
/// This is intentionally not a clone-based API. The transform owns the
/// candidate model: it computes replacement arrivals for the values it would
/// rewrite, while TimingV2 handles propagation state, endpoint-path membership,
/// and endpoint-delay estimation against the unmodified network.
class TimingSpeculationContext {
public:
  /// Propagate `network` and reconstruct the current worst endpoint path for
  /// `options`.
  static mlir::FailureOr<TimingSpeculationContext>
  create(const TimingNetwork &network, TimingPropagationOptions options = {});

  /// Return the graph, propagation result, options, and current worst endpoint
  /// path.
  const TimingNetwork &getNetwork() const { return network; }
  const TimingPropagationResult &getPropagation() const { return propagation; }
  TimingPropagationOptions getOptions() const { return options; }
  const ReconstructedTimingPath &getWorstEndpointPath() const {
    return worstEndpointPath;
  }
  /// Return the current worst endpoint delay under the active objective.
  int64_t getWorstEndpointDelay() const { return worstEndpointPath.delay; }

  /// Query propagated arrival by value bit.
  mlir::FailureOr<int64_t> getArrival(mlir::Value value, uint32_t bit) const;
  /// Return true when the value bit is present on the reconstructed worst
  /// endpoint path.
  bool isOnWorstEndpointPath(mlir::Value value, uint32_t bit) const;

  /// Estimate one endpoint arrival after applying replacement arrivals to that
  /// endpoint path. If multiple replacements are on the path, the latest path
  /// point wins; this matches the common rewrite use case where the transform
  /// supplies arrivals for rewritten outputs rather than every internal
  /// candidate node.
  mlir::FailureOr<int64_t> speculateEndpointDelay(
      TimingPointId endpoint,
      llvm::ArrayRef<TimingArrivalReplacement> replacements) const;
  /// Estimate the current worst endpoint delay after applying replacement
  /// arrivals across all endpoint paths.
  mlir::FailureOr<TimingEndpointSpeculation>
  speculateWorstEndpointDelay(
      llvm::ArrayRef<TimingArrivalReplacement> replacements) const;

private:
  TimingSpeculationContext(const TimingNetwork &network,
                           TimingPropagationOptions options,
                           TimingPropagationResult propagation,
                           ReconstructedTimingPath worstEndpointPath)
      : network(network), options(std::move(options)),
        propagation(std::move(propagation)),
        worstEndpointPath(std::move(worstEndpointPath)) {}

  const TimingNetwork &network;
  TimingPropagationOptions options;
  TimingPropagationResult propagation;
  ReconstructedTimingPath worstEndpointPath;
};

/// Propagate timing and reconstruct the best endpoint path for `options`.
mlir::FailureOr<ReconstructedTimingPath>
reconstructCriticalPath(const TimingNetwork &network,
                        TimingPropagationOptions options = {});

/// Print a compact textual critical path report.
mlir::LogicalResult printCriticalTimingReport(
    const TimingNetwork &network, llvm::raw_ostream &os,
    TimingPropagationOptions options = {});

/// PatternRewriter listener that keeps a TimingNetwork synchronized with local
/// IR edits when possible and falls back to full rebuilds for structural edits.
class TimingRepairSession : public mlir::PatternRewriter::Listener {
public:
  /// Create a repair session for `root`. The delay model and semantics provider
  /// must outlive the session when supplied.
  TimingRepairSession(
      mlir::Operation *root, const DelayModel *delayModel = nullptr,
      const TimingSemanticsProvider *semanticsProvider = nullptr);

  /// Build the initial network.
  mlir::LogicalResult initialize();
  /// Apply pending listener edits to the network.
  mlir::LogicalResult repair();

  /// Query repair-session status.
  bool isInitialized() const { return initialized; }
  bool hasPendingChanges() const { return pendingChanges; }
  bool needsFullRebuild() const { return fullRebuildRequired; }

  /// Return the current network, or null before successful initialization.
  const TimingNetwork *getNetwork() const { return network.get(); }

  /// PatternRewriter::Listener hooks.
  void notifyOperationInserted(mlir::Operation *op,
                               mlir::OpBuilder::InsertPoint previous) override;
  void notifyOperationModified(mlir::Operation *op) override;
  void notifyOperationReplaced(mlir::Operation *op,
                               mlir::Operation *replacement) override;
  void notifyOperationReplaced(mlir::Operation *op,
                               mlir::ValueRange replacement) override;
  void notifyOperationErased(mlir::Operation *op) override;

private:
  bool canRepairLocally(mlir::Operation *op) const;
  mlir::LogicalResult repairLocalEdits();
  void recordInserted(mlir::Operation *op);
  void recordModified(mlir::Operation *op);
  void recordReplacement(mlir::Operation *op, mlir::ValueRange replacement);
  void recordErasure(mlir::Operation *op);
  void recordAffectedUsers(mlir::Operation *op);

  mlir::Operation *root = nullptr;
  const DelayModel *delayModel = nullptr;
  const TimingSemanticsProvider *semanticsProvider = nullptr;
  std::unique_ptr<TimingNetwork> network;
  llvm::SmallVector<mlir::Operation *, 8> dirtyOps;
  llvm::SmallVector<mlir::Operation *, 8> removedOps;
  bool initialized = false;
  bool pendingChanges = false;
  bool fullRebuildRequired = false;
};

} // namespace timingv2
} // namespace synth
} // namespace circt

namespace llvm {
template <> struct DenseMapInfo<circt::synth::timingv2::TimingPointId> {
  static circt::synth::timingv2::TimingPointId getEmptyKey() {
    return {UINT32_MAX};
  }
  static circt::synth::timingv2::TimingPointId getTombstoneKey() {
    return {UINT32_MAX - 1};
  }
  static unsigned
  getHashValue(circt::synth::timingv2::TimingPointId id) {
    return DenseMapInfo<uint32_t>::getHashValue(id.index);
  }
  static bool isEqual(circt::synth::timingv2::TimingPointId lhs,
                      circt::synth::timingv2::TimingPointId rhs) {
    return lhs.index == rhs.index;
  }
};
} // namespace llvm

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGV2_FLATTIMING_H
