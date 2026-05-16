#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/Analysis/TimingV2/FlatTiming.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace circt {
namespace synth {

#define GEN_PASS_DEF_TIMINGV2GUIDEDREWRITE
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"

} // namespace synth
} // namespace circt

namespace {

using circt::comb::MuxOp;
using circt::hw::HWModuleOp;
namespace timingv2 = circt::synth::timingv2;

constexpr llvm::StringLiteral kTimingV2GuidedRewriteReportAttrName =
    "synth.timing_v2_guided_rewrite_report";

struct PriorityMuxChain {
  llvm::SmallVector<MuxOp, 8> muxes;
  llvm::SmallVector<mlir::Value, 8> conditions;
  llvm::SmallVector<mlir::Value, 8> results;
  llvm::SmallVector<mlir::Location, 8> locations;
};

struct RewriteTimingRecord {
  unsigned conditions = 0;
  int64_t before = 0;
  int64_t predicted = 0;
  int64_t after = 0;
};

struct MuxBalanceSpeculation {
  int64_t before = 0;
  int64_t predicted = 0;
};

static unsigned getBitWidth(mlir::Value value) {
  if (auto type = mlir::dyn_cast<mlir::IntegerType>(value.getType()))
    return type.getWidth();
  return 1;
}

struct TimingGuidedRewriteState {
  TimingGuidedRewriteState(timingv2::TimingRepairSession &timingSession,
                           int64_t requiredTime)
      : timingSession(timingSession), requiredTime(requiredTime) {}

  timingv2::TimingRepairSession &timingSession;
  int64_t requiredTime = 0;
  llvm::SmallVector<RewriteTimingRecord, 8> records;

  mlir::FailureOr<int64_t> getCurrentDelay(mlir::Location loc) const {
    auto *network = timingSession.getNetwork();
    if (!network) {
      mlir::emitError(loc) << "TimingV2 network is not initialized";
      return mlir::failure();
    }
    timingv2::TimingPropagationOptions options;
    options.defaultRequiredTime = requiredTime;
    auto speculation = timingv2::TimingSpeculationContext::create(*network,
                                                                  options);
    if (llvm::failed(speculation)) {
      mlir::emitError(loc) << "failed to reconstruct TimingV2 critical path";
      return mlir::failure();
    }
    return speculation->getCriticalPath().delay;
  }

  mlir::LogicalResult repair(mlir::Location loc) {
    if (llvm::failed(timingSession.repair()))
      return mlir::emitError(loc) << "failed to repair TimingV2 network";
    return mlir::success();
  }
};

static bool isFalseSidePriorityMuxRoot(MuxOp op) {
  if (op->hasOneUse() && mlir::isa<MuxOp>(*op->user_begin()))
    return false;
  auto falseMux = op.getFalseValue().getDefiningOp<MuxOp>();
  auto trueMux = op.getTrueValue().getDefiningOp<MuxOp>();
  return falseMux && !trueMux;
}

static PriorityMuxChain collectFalseSidePriorityChain(MuxOp rootMux) {
  PriorityMuxChain chain;
  chain.muxes.push_back(rootMux);
  chain.conditions.push_back(rootMux.getCond());
  chain.results.push_back(rootMux.getTrueValue());
  chain.locations.push_back(rootMux.getLoc());

  MuxOp current = rootMux.getFalseValue().getDefiningOp<MuxOp>();
  while (current) {
    chain.muxes.push_back(current);
    chain.conditions.push_back(current.getCond());
    chain.results.push_back(current.getTrueValue());
    chain.locations.push_back(current.getLoc());

    auto nextValue = current.getFalseValue();
    auto nextMux = nextValue.getDefiningOp<MuxOp>();
    if (!nextMux || !nextMux->hasOneUse()) {
      chain.results.push_back(nextValue);
      break;
    }
    current = nextMux;
  }

  return chain;
}

// NOLINTNEXTLINE(misc-no-recursion)
static mlir::Value buildBalancedPriorityMux(mlir::OpBuilder &rewriter,
                                            llvm::ArrayRef<mlir::Value> conds,
                                            llvm::ArrayRef<mlir::Value> results,
                                            mlir::Value defaultValue,
                                            llvm::ArrayRef<mlir::Location> locs) {
  size_t size = conds.size();
  if (size == 0)
    return defaultValue;
  if (size == 1)
    return rewriter.createOrFold<MuxOp>(locs.front(), conds.front(),
                                        results.front(), defaultValue);

  unsigned mid = llvm::divideCeil(size, 2);
  auto loc = rewriter.getFusedLoc(locs.take_front(mid));
  auto leftTree = buildBalancedPriorityMux(
      rewriter, conds.take_front(mid), results.take_front(mid),
      results.take_front(mid).back(), locs.take_front(mid));
  auto rightTree = buildBalancedPriorityMux(
      rewriter, conds.drop_front(mid), results.drop_front(mid), defaultValue,
      locs.drop_front(mid));
  auto combinedCond =
      rewriter.createOrFold<circt::comb::OrOp>(loc, conds.take_front(mid),
                                               /*twoState=*/true);
  return MuxOp::create(rewriter, loc, combinedCond, leftTree, rightTree);
}

static mlir::LogicalResult rewritePriorityMuxChain(MuxOp rootMux,
                                                   mlir::PatternRewriter
                                                       &rewriter) {
  if (!isFalseSidePriorityMuxRoot(rootMux))
    return mlir::failure();
  auto chain = collectFalseSidePriorityChain(rootMux);
  if (chain.conditions.size() < 2 ||
      chain.conditions.size() + 1 != chain.results.size())
    return mlir::failure();

  rewriter.setInsertionPoint(rootMux);
  auto balanced = buildBalancedPriorityMux(
      rewriter, chain.conditions, llvm::ArrayRef(chain.results).drop_back(),
      chain.results.back(), chain.locations);
  rewriter.replaceOp(rootMux, balanced);
  for (unsigned i = 1, e = chain.muxes.size(); i < e; ++i)
    if (chain.muxes[i]->use_empty())
      rewriter.eraseOp(chain.muxes[i]);
  return mlir::success();
}

static mlir::FailureOr<int64_t>
speculateBalancedMuxArrival(const timingv2::TimingSpeculationContext &context,
                            llvm::ArrayRef<mlir::Value> conds,
                            llvm::ArrayRef<mlir::Value> results,
                            mlir::Value defaultValue, uint32_t bit,
                            mlir::Location loc) {
  size_t size = conds.size();
  if (size == 0)
    return context.getArrival(defaultValue, bit);

  if (size == 1) {
    auto resultArrival = context.getArrival(results.front(), bit);
    if (llvm::failed(resultArrival))
      return mlir::emitError(loc) << "missing TimingV2 result arrival";
    if (results.front() == defaultValue)
      return *resultArrival;

    auto defaultArrival = context.getArrival(defaultValue, bit);
    if (llvm::failed(defaultArrival))
      return mlir::emitError(loc) << "missing TimingV2 default arrival";
    auto condArrival = context.getArrival(conds.front(), 0);
    if (llvm::failed(condArrival))
      return mlir::emitError(loc) << "missing TimingV2 condition arrival";
    return std::max({*condArrival + 1, *resultArrival + 1,
                     *defaultArrival + 1});
  }

  unsigned mid = llvm::divideCeil(size, 2);
  auto left = speculateBalancedMuxArrival(
      context, conds.take_front(mid), results.take_front(mid),
      results.take_front(mid).back(), bit, loc);
  if (llvm::failed(left))
    return mlir::failure();
  auto right = speculateBalancedMuxArrival(
      context, conds.drop_front(mid), results.drop_front(mid), defaultValue,
      bit, loc);
  if (llvm::failed(right))
    return mlir::failure();

  int64_t combinedCondArrival = 0;
  bool hasCondArrival = false;
  for (auto cond : conds.take_front(mid)) {
    auto condArrival = context.getArrival(cond, 0);
    if (llvm::failed(condArrival))
      return mlir::emitError(loc) << "missing TimingV2 condition arrival";
    combinedCondArrival =
        hasCondArrival ? std::max(combinedCondArrival, *condArrival)
                       : *condArrival;
    hasCondArrival = true;
  }
  combinedCondArrival += llvm::Log2_64_Ceil(mid);
  return std::max({combinedCondArrival + 1, *left + 1, *right + 1});
}

static mlir::FailureOr<MuxBalanceSpeculation>
speculatePriorityMuxBalance(TimingGuidedRewriteState &state, MuxOp rootMux,
                            const PriorityMuxChain &chain) {
  auto loc = rootMux.getLoc();
  auto *network = state.timingSession.getNetwork();
  if (!network)
    return mlir::emitError(loc) << "TimingV2 network is not initialized";

  timingv2::TimingPropagationOptions options;
  options.defaultRequiredTime = state.requiredTime;
  auto context = timingv2::TimingSpeculationContext::create(*network, options);
  if (llvm::failed(context))
    return mlir::failure();

  auto result = rootMux.getResult();
  llvm::SmallVector<timingv2::TimingArrivalReplacement, 8> replacements;
  for (uint32_t bit = 0, width = getBitWidth(result); bit < width; ++bit) {
    auto newArrival = speculateBalancedMuxArrival(
        *context, chain.conditions, llvm::ArrayRef(chain.results).drop_back(),
        chain.results.back(), bit, loc);
    if (llvm::failed(newArrival))
      return mlir::failure();
    replacements.push_back({result, bit, *newArrival});
  }

  auto speculation = context->speculateCriticalPathDelay(replacements);
  if (llvm::failed(speculation))
    return mlir::failure();
  return MuxBalanceSpeculation{speculation->baselineDelay,
                               speculation->predictedDelay};
}

class BalancePriorityMuxPattern : public mlir::OpRewritePattern<MuxOp> {
public:
  BalancePriorityMuxPattern(mlir::MLIRContext *context,
                            TimingGuidedRewriteState &state)
      : mlir::OpRewritePattern<MuxOp>(context), state(state) {}

  mlir::LogicalResult
  matchAndRewrite(MuxOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (llvm::failed(state.repair(loc)))
      return mlir::failure();

    if (!isFalseSidePriorityMuxRoot(op))
      return rewriter.notifyMatchFailure(
          op, "not a false-side priority mux chain root");
    auto chain = collectFalseSidePriorityChain(op);
    if (chain.conditions.size() < 2 ||
        chain.conditions.size() + 1 != chain.results.size())
      return rewriter.notifyMatchFailure(op, "not a balanceable mux chain");

    auto speculation = speculatePriorityMuxBalance(state, op, chain);
    if (llvm::failed(speculation))
      return mlir::failure();
    if (speculation->predicted >= speculation->before)
      return rewriter.notifyMatchFailure(
          op, "speculative TimingV2 estimate does not improve critical delay");

    if (llvm::failed(rewritePriorityMuxChain(op, rewriter)))
      return mlir::failure();
    if (llvm::failed(state.repair(loc)))
      return mlir::failure();

    auto after = state.getCurrentDelay(loc);
    if (llvm::failed(after))
      return mlir::failure();
    state.records.push_back(
        RewriteTimingRecord{static_cast<unsigned>(chain.conditions.size()),
                            speculation->before, speculation->predicted,
                            *after});
    return mlir::success();
  }

private:
  TimingGuidedRewriteState &state;
};

static mlir::FailureOr<std::string>
runTimingGuidedRewrite(HWModuleOp module, int64_t requiredTime) {
  timingv2::TimingRepairSession timingSession(module);
  if (llvm::failed(timingSession.initialize()))
    return module.emitError() << "failed to build TimingV2 network";

  TimingGuidedRewriteState state{timingSession, requiredTime};

  auto initialDelay = state.getCurrentDelay(module.getLoc());
  if (llvm::failed(initialDelay))
    return mlir::failure();

  std::string report;
  llvm::raw_string_ostream os(report);
  os << "TimingV2 guided rewrite report\n";
  os << "module: @" << module.getName() << "\n";
  os << "initial_delay: " << *initialDelay << "\n";
  os << "pattern: balance false-side priority mux chain\n";

  mlir::RewritePatternSet patterns(module.getContext());
  patterns.add<BalancePriorityMuxPattern>(module.getContext(), state);

  mlir::GreedyRewriteConfig config;
  config.setListener(&timingSession).setUseTopDownTraversal(true);
  bool changed = false;
  if (llvm::failed(
          mlir::applyPatternsGreedily(module, std::move(patterns), config,
                                      &changed)))
    return module.emitError() << "failed to apply TimingV2 rewrite patterns";
  if (changed && llvm::failed(timingSession.repair()))
    return module.emitError() << "failed to repair TimingV2 network";

  auto finalDelay = state.getCurrentDelay(module.getLoc());
  if (llvm::failed(finalDelay))
    return mlir::failure();

  for (const auto &record : state.records)
    os << "rewrite: priority_mux_chain conditions=" << record.conditions
       << " before=" << record.before
       << " predicted=" << record.predicted << " after=" << record.after
       << "\n";
  os << "rewrites_applied: " << state.records.size() << "\n";
  os << "final_delay: " << *finalDelay << "\n";
  os.flush();
  return report;
}

class TimingV2GuidedRewritePass
    : public circt::synth::impl::TimingV2GuidedRewriteBase<
          TimingV2GuidedRewritePass> {
public:
  using TimingV2GuidedRewriteBase::TimingV2GuidedRewriteBase;

  void runOnOperation() final {
    auto module = getOperation();

    auto report = runTimingGuidedRewrite(module, requiredTime);
    if (llvm::failed(report))
      return signalPassFailure();

    module->setAttr(
        kTimingV2GuidedRewriteReportAttrName,
        mlir::StringAttr::get(module.getContext(), *report));
    if (printReport) {
      llvm::outs() << *report;
      if (!llvm::StringRef(*report).ends_with("\n"))
        llvm::outs() << "\n";
    }
  }
};

} // namespace
