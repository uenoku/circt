//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SOP (Sum-of-Products) balancing for delay optimization
// based on "Delay Optimization Using SOP Balancing" by Mishchenko et al.
// (ICCAD 2011).
//
// NOTE: Currently supports AIG but should be extended to other logic forms
// (MIG, comb or/and) in the future.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "synth-sop-balancing"

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace circt {
namespace synth {
#define GEN_PASS_DEF_SOPBALANCING
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt::synth;

//===----------------------------------------------------------------------===//
// SOP Cache
//===----------------------------------------------------------------------===//

namespace {

/// Expected maximum number of inputs for ISOP extraction, used as a hint for
/// SmallVector capacity to avoid reallocations in common cases.
constexpr unsigned expectedISOPInputs = 8;

/// Cache for SOP extraction results, keyed by truth table.
class SOPCache {
public:
  const SOPForm &getOrCompute(const APInt &truthTable, unsigned numVars) {
    auto it = cache.find(truthTable);
    if (it != cache.end())
      return it->second;
    return cache
        .try_emplace(truthTable, circt::extractISOP(truthTable, numVars))
        .first->second;
  }

private:
  DenseMap<APInt, SOPForm> cache;
};

//===----------------------------------------------------------------------===//
// Tree Building Helpers
//===----------------------------------------------------------------------===//

struct SOPSignal {
  unsigned index = 0;
  bool isInput = true;
  bool inverted = false;
};

struct SOPAndNode {
  SOPSignal lhs;
  SOPSignal rhs;
};

struct SOPImplementation : MatchImplementation {
  SmallVector<SOPAndNode, 8> nodes;
  SOPSignal output;
};

struct SOPPlanNode {
  DelayType arrivalTime = 0;
  size_t valueNumbering = 0;
  uint64_t usedMask = 0;
  SmallVector<DelayType, expectedISOPInputs> inputDelays;
  SOPSignal signal;

  SOPPlanNode flipInversion() const {
    SOPPlanNode node = *this;
    node.signal.inverted = !node.signal.inverted;
    return node;
  }

  bool operator>(const SOPPlanNode &other) const {
    return std::tie(arrivalTime, valueNumbering) >
           std::tie(other.arrivalTime, other.valueNumbering);
  }
};

static SOPPlanNode combineSOPPlanNodes(const SOPPlanNode &lhs,
                                       const SOPPlanNode &rhs,
                                       unsigned numVars,
                                       size_t &valueNumbering,
                                       SOPImplementation &implementation) {
  SOPPlanNode node;
  node.arrivalTime = std::max(lhs.arrivalTime, rhs.arrivalTime) + 1;
  node.valueNumbering = valueNumbering++;
  node.usedMask = lhs.usedMask | rhs.usedMask;
  node.inputDelays.assign(numVars, 0);
  implementation.nodes.push_back({lhs.signal, rhs.signal});
  node.signal = SOPSignal{
      static_cast<unsigned>(implementation.nodes.size() - 1),
      /*isInput=*/false, /*inverted=*/false};

  auto accumulateDelays = [&](const SOPPlanNode &source) {
    for (unsigned i = 0; i < numVars; ++i) {
      if (!(source.usedMask & (uint64_t{1} << i)))
        continue;
      node.inputDelays[i] =
          std::max(node.inputDelays[i], static_cast<DelayType>(
                                            source.inputDelays[i] + 1));
    }
  };
  accumulateDelays(lhs);
  accumulateDelays(rhs);
  return node;
}

static std::shared_ptr<const SOPImplementation>
buildSOPImplementation(const SOPForm &sop, ArrayRef<DelayType> inputArrivalTimes,
                       SmallVectorImpl<DelayType> &delays) {
  assert(sop.numVars < 64 && "SOP delay model supports at most 63 variables");

  auto implementation = std::make_shared<SOPImplementation>();
  SmallVector<SOPPlanNode, expectedISOPInputs> productTerms, literals;
  size_t valueNumbering = 0;
  for (const auto &cube : sop.cubes) {
    for (unsigned i = 0; i < sop.numVars; ++i)
      // Literal polarity does not affect structural path length.
      if (cube.hasLiteral(i)) {
        SOPPlanNode literal;
        literal.arrivalTime = inputArrivalTimes[i];
        literal.valueNumbering = valueNumbering++;
        literal.usedMask = uint64_t{1} << i;
        literal.inputDelays.assign(sop.numVars, 0);
        literal.signal =
            SOPSignal{i, /*isInput=*/true, cube.isLiteralInverted(i)};
        literals.push_back(std::move(literal));
      }
    if (!literals.empty()) {
      productTerms.push_back(
          buildBalancedTreeWithArrivalTimes<SOPPlanNode>(
              literals, [&](const SOPPlanNode &lhs, const SOPPlanNode &rhs) {
                return combineSOPPlanNodes(lhs, rhs, sop.numVars,
                                           valueNumbering, *implementation);
              })
              .flipInversion());
      literals.clear();
    }
  }

  delays.resize(sop.numVars, 0);
  if (productTerms.empty())
    return implementation;

  auto outputNode = buildBalancedTreeWithArrivalTimes<SOPPlanNode>(
                        productTerms,
                        [&](const SOPPlanNode &lhs, const SOPPlanNode &rhs) {
                          return combineSOPPlanNodes(lhs, rhs, sop.numVars,
                                                     valueNumbering,
                                                     *implementation);
                        })
                        .flipInversion();
  implementation->output = outputNode.signal;

  for (unsigned i = 0; i < sop.numVars; ++i)
    if (outputNode.usedMask & (uint64_t{1} << i))
      delays[i] = outputNode.inputDelays[i];
  return implementation;
}

//===----------------------------------------------------------------------===//
// SOP Balancing Pattern
//===----------------------------------------------------------------------===//

/// Pattern that performs SOP balancing on cuts.
struct SOPBalancingPattern : public CutRewritePattern {
  SOPBalancingPattern(MLIRContext *context) : CutRewritePattern(context) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    const auto &network = enumerator.getLogicNetwork();
    if (cut.isTrivialCut() || cut.getOutputSize(network) != 1)
      return std::nullopt;

    const auto &tt = *cut.getTruthTable();
    const SOPForm &sop = sopCache.getOrCompute(tt.table, tt.numInputs);
    if (sop.cubes.empty())
      return std::nullopt;

    SmallVector<DelayType, expectedISOPInputs> arrivalTimes;
    if (failed(cut.getInputArrivalTimes(enumerator, arrivalTimes))) {
      assert(false);
      return std::nullopt;
    }

    // Compute area estimate
    unsigned totalGates = 0;
    for (const auto &cube : sop.cubes)
      if (cube.size() > 1)
        totalGates += cube.size() - 1;
    if (sop.cubes.size() > 1)
      totalGates += sop.cubes.size() - 1;


    SmallVector<DelayType, expectedISOPInputs> delays;
    auto implementation = buildSOPImplementation(sop, arrivalTimes, delays);

    llvm::errs() << "SOP with " << sop.cubes.size() << " cubes and "
                 << totalGates << " " << cut.getInputSize() << "-input gates\n";
    llvm::errs() << "Truth table: " << tt.table << "\n";
    llvm::errs() << "Extracted SOP:\n";
    sop.dump(llvm::errs());
    for (unsigned i = 0; i < cut.getInputSize(); ++i)
      llvm::errs() << "Input " << i << " arrival time: " << arrivalTimes[i]
                   << ", delay contribution: " << delays[i] << "\n";
    MatchResult result;
    result.area = static_cast<double>(totalGates);
    result.setOwnedDelays(std::move(delays));
    result.setImplementation(std::move(implementation));
    return result;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut,
                                 const MatchedPattern &matched) const override {
    const auto &network = enumerator.getLogicNetwork();
    auto *implementation =
        static_cast<const SOPImplementation *>(matched.getImplementation());
    if (!implementation)
      return failure();

    // Construct the fused location.
    SetVector<Location> inputLocs;
    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");
    inputLocs.insert(rootOp->getLoc());

    SmallVector<Value> inputValues;
    network.getValues(cut.inputs, inputValues);
    for (auto input : inputValues)
      inputLocs.insert(input.getLoc());

    auto loc = builder.getFusedLoc(inputLocs.getArrayRef());

    SmallVector<Value> nodeValues;
    nodeValues.reserve(implementation->nodes.size());
    auto resolveSignal = [&](const SOPSignal &signal) -> Value {
      return signal.isInput ? inputValues[signal.index] : nodeValues[signal.index];
    };

    for (const auto &node : implementation->nodes) {
      Value lhs = resolveSignal(node.lhs);
      Value rhs = resolveSignal(node.rhs);
      nodeValues.push_back(aig::AndInverterOp::create(
          builder, loc, lhs, rhs, node.lhs.inverted, node.rhs.inverted));
    }

    Value result = resolveSignal(implementation->output);
    if (implementation->output.inverted)
      result = aig::AndInverterOp::create(builder, loc, result, true);

    auto *op = result.getDefiningOp();
    if (!op)
      op = aig::AndInverterOp::create(builder, loc, result, false);
    return op;
  }

  unsigned getNumOutputs() const override { return 1; }
  StringRef getPatternName() const override { return "sop-balancing"; }

private:
  // Cache for SOP extraction results. Hence the pattern is stateful and must
  // not be used in parallelly.
  mutable SOPCache sopCache;
};

struct NativeOpPattern : public CutRewritePattern {
  NativeOpPattern(MLIRContext *context) : CutRewritePattern(context) {}
  constexpr static DelayType delays[3] = {1, 1, 1};

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    const auto &network = enumerator.getLogicNetwork();
    auto size = cut.getInputSize();
    if (cut.isTrivialCut() || size >= 4 || size <= 1)
      return std::nullopt;

    const auto &gate = network.getGate(cut.getRootIndex());
    auto *rootOp = gate.getOperation();
    if (!rootOp || !gate.matchesCutExactly(cut))
      return std::nullopt;

    // For exact 2-input AIG cuts this mostly acts as a tie-breaker, since SOP
    // extraction can already encode literal polarity and rebuild the same gate.
    // The more visible effect is preserving native XOR/MAJ nodes, or mixed
    // cones where exact native cuts should win over a generic SOP rewrite.
    return MatchResult(1.0, ArrayRef<DelayType>(delays, cut.getInputSize()));
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 const Cut &cut,
                                 const MatchedPattern &) const override {
    const auto &network = enumerator.getLogicNetwork();
    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    if (!rootOp)
      return failure();

    auto gate = network.getGate(cut.getRootIndex());
    SmallVector<Value> inputValues;
    network.getValues(cut.inputs, inputValues);
    auto getEdgeTT = [&](const Signal &edge) -> std::pair<Value, bool> {
      Value v = network.getValue(edge.getIndex());
      bool inverted = edge.isInverted();
      return {v, inverted};
    };
    switch (gate.getKind()) {
    case LogicNetworkGate::And2: {
      auto [a, aInv] = getEdgeTT(gate.edges[0]);
      auto [b, bInv] = getEdgeTT(gate.edges[1]);
      auto newOp = aig::AndInverterOp::create(builder, rootOp->getLoc(), a, b,
                                              aInv, bInv);
      return newOp.getOperation();
    }
    case LogicNetworkGate::Xor2: {
      SmallVector<Value> operands;
      for (unsigned i = 0; i < 2; ++i) {
        auto [v, inv] = getEdgeTT(gate.edges[i]);
        if (inv)
          v = comb::createOrFoldNot(builder, rootOp->getLoc(), v);
        operands.push_back(v);
      }
      auto newOp =
          comb::XorOp::create(builder, rootOp->getLoc(), operands, true);
      return newOp.getOperation();
    }
    case LogicNetworkGate::Maj3: {
      auto [a, aInv] = getEdgeTT(gate.edges[0]);
      auto [b, bInv] = getEdgeTT(gate.edges[1]);
      auto [c, cInv] = getEdgeTT(gate.edges[2]);
      auto newOp = synth::mig::MajorityInverterOp::create(
          builder, rootOp->getLoc(), ArrayRef<Value>{a, b, c},
          ArrayRef<bool>{aInv, bInv, cInv});
      return newOp.getOperation();
    }
    default:
      llvm_unreachable("unexpected gate kind for native op pattern");
    }
  }

  unsigned getNumOutputs() const override { return 1; }
  StringRef getPatternName() const override { return "native-op"; }
};
} // namespace

//===----------------------------------------------------------------------===//
// SOP Balancing Pass
//===----------------------------------------------------------------------===//

struct SOPBalancingPass
    : public circt::synth::impl::SOPBalancingBase<SOPBalancingPass> {
  using SOPBalancingBase::SOPBalancingBase;

  void runOnOperation() override {
    auto module = getOperation();

    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = maxCutInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.allowNoMatch = true;

    SmallVector<std::unique_ptr<CutRewritePattern>, 2> patterns;
    if (!std::getenv("DISABLE_NATIVE_PATTERN"))
      patterns.push_back(std::make_unique<NativeOpPattern>(module->getContext()));
    patterns.push_back(
        std::make_unique<SOPBalancingPattern>(module->getContext()));

    CutRewritePatternSet patternSet(std::move(patterns));
    CutRewriter rewriter(options, patternSet);
    if (failed(rewriter.run(module)))
      return signalPassFailure();
  }
};
