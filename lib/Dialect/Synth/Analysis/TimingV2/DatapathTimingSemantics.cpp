//===- DatapathTimingSemantics.cpp - Datapath timing policy -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/TimingV2/FlatTiming.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::timingv2;
using namespace mlir;

static size_t getBitWidth(Value value) {
  if (auto intType = dyn_cast<IntegerType>(value.getType()))
    return intType.getWidth();
  return 1;
}

static TimingSemanticPoint point(Value value, uint32_t bit,
                                 Operation *owner = nullptr) {
  TimingSemanticPoint point;
  point.value = value;
  point.bit = bit;
  point.kind = TimingPointKind::ValueBit;
  point.owner = owner;
  return point;
}

static void addArc(TimingSemantics &semantics, Value from, uint32_t fromBit,
                   Value to, uint32_t toBit, Operation *op,
                   int32_t inputIndex, int32_t outputIndex, int64_t delay,
                   StringRef token) {
  TimingSemanticArc arc;
  arc.from = point(from, fromBit);
  arc.to = point(to, toBit, op);
  arc.kind = TimingArcKind::Synthetic;
  arc.fixedDelay = delay;
  arc.op = op;
  arc.inputIndex = inputIndex;
  arc.outputIndex = outputIndex;
  arc.token = token.str();
  semantics.arcs.push_back(std::move(arc));
}

struct SourceBit {
  Value value;
  uint32_t bit = 0;
  int32_t operandIndex = -1;
  int64_t arrival = 0;
};

struct ScheduledBit {
  int64_t arrival = 0;
  SmallVector<SourceBit, 4> sources;
};

struct CompressorOutputBit {
  int32_t resultIndex = -1;
  uint32_t bit = 0;
  ScheduledBit scheduled;
};

struct GreedyCompressorSchedule {
  SmallVector<CompressorOutputBit, 16> outputs;
};

static bool hasSource(ArrayRef<SourceBit> sources, const SourceBit &source) {
  return llvm::any_of(sources, [&](const SourceBit &existing) {
    return existing.value == source.value && existing.bit == source.bit &&
           existing.operandIndex == source.operandIndex;
  });
}

static void appendUniqueSources(SmallVectorImpl<SourceBit> &dest,
                                ArrayRef<SourceBit> sources) {
  for (const auto &source : sources) {
    if (!hasSource(dest, source))
      dest.push_back(source);
  }
}

static ScheduledBit combineCompressorInputs(ArrayRef<ScheduledBit> inputs,
                                            int64_t compressorDelay) {
  ScheduledBit result;
  for (const auto &input : inputs) {
    result.arrival = std::max(result.arrival, input.arrival);
    appendUniqueSources(result.sources, input.sources);
  }
  result.arrival += compressorDelay;
  return result;
}

static GreedyCompressorSchedule
buildGreedyCompressorSchedule(Operation *op,
                              const TimingDynamicContext &context,
                              int64_t stageDelay) {
  GreedyCompressorSchedule schedule;
  unsigned width = 0;
  for (auto result : op->getResults())
    width = std::max<unsigned>(width, getBitWidth(result));
  for (auto operand : op->getOperands())
    width = std::max<unsigned>(width, getBitWidth(operand));

  SmallVector<SmallVector<ScheduledBit, 8>, 16> columns(width);
  for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
    for (uint32_t bit = 0, e = getBitWidth(operand); bit < e; ++bit) {
      auto arrival = context.getArrival(operand, bit);
      ScheduledBit item;
      item.arrival = succeeded(arrival) ? *arrival : 0;
      item.sources.push_back(
          {operand, bit, static_cast<int32_t>(operandIndex), item.arrival});
      columns[bit].push_back(std::move(item));
    }
  }

  unsigned targetRows = op->getNumResults();
  if (targetRows == 0)
    return schedule;

  bool changed = true;
  for (unsigned stage = 0; changed && stage < width + op->getNumOperands();
       ++stage) {
    changed = false;
    SmallVector<SmallVector<ScheduledBit, 8>, 16> next(columns.size());
    for (unsigned columnIndex = 0, e = columns.size(); columnIndex < e;
         ++columnIndex) {
      auto column = std::move(columns[columnIndex]);
      llvm::sort(column, [](const ScheduledBit &lhs, const ScheduledBit &rhs) {
        return lhs.arrival > rhs.arrival;
      });

      while (column.size() > targetRows && column.size() >= 3) {
        unsigned inputCount = column.size() >= 4 ? 4 : 3;
        SmallVector<ScheduledBit, 4> compressorInputs;
        compressorInputs.reserve(inputCount);
        for (unsigned i = 0; i < inputCount; ++i)
          compressorInputs.push_back(column.pop_back_val());
        int64_t compressorDelay =
            stageDelay * static_cast<int64_t>(inputCount == 4 ? 2 : 1);
        auto compressed =
            combineCompressorInputs(compressorInputs, compressorDelay);
        next[columnIndex].push_back(compressed);
        if (columnIndex + 1 < e)
          next[columnIndex + 1].push_back(std::move(compressed));
        changed = true;
      }

      for (auto &item : column)
        next[columnIndex].push_back(std::move(item));
    }
    columns = std::move(next);
  }

  for (unsigned bit = 0, e = columns.size(); bit < e; ++bit) {
    auto &column = columns[bit];
    llvm::sort(column, [](const ScheduledBit &lhs, const ScheduledBit &rhs) {
      return lhs.arrival > rhs.arrival;
    });

    for (unsigned resultIndex = 0,
                  resultEnd = std::min<unsigned>(targetRows, column.size());
         resultIndex < resultEnd; ++resultIndex) {
      schedule.outputs.push_back({static_cast<int32_t>(resultIndex), bit,
                                  std::move(column[resultIndex])});
    }
  }

  return schedule;
}

static bool scheduleMeetsRequiredTimes(Operation *op,
                                       const TimingDynamicContext &context,
                                       const GreedyCompressorSchedule &schedule) {
  for (const auto &output : schedule.outputs) {
    if (output.resultIndex < 0 ||
        static_cast<unsigned>(output.resultIndex) >= op->getNumResults())
      continue;
    auto result = op->getResult(output.resultIndex);
    if (output.bit >= getBitWidth(result))
      continue;
    auto required = context.getRequired(result, output.bit);
    if (failed(required))
      continue;
    if (output.scheduled.arrival > *required)
      return false;
  }
  return true;
}

static void addGreedyCompressorArcs(TimingSemantics &semantics, Operation *op,
                                    const GreedyCompressorSchedule &schedule,
                                    StringRef token) {
  for (const auto &output : schedule.outputs) {
    if (output.resultIndex < 0 ||
        static_cast<unsigned>(output.resultIndex) >= op->getNumResults())
      continue;
    auto result = op->getResult(output.resultIndex);
    if (output.bit >= getBitWidth(result))
      continue;
    for (const auto &source : output.scheduled.sources) {
      int64_t delay = output.scheduled.arrival - source.arrival;
      addArc(semantics, source.value, source.bit, result, output.bit, op,
             source.operandIndex, output.resultIndex, delay, token);
    }
  }
}

static TimingSemantics
describeCompressor(Operation *op, const TimingDynamicContext &context,
                   const DatapathTimingSemanticsOptions &options) {
  TimingSemantics semantics;
  semantics.op = op;

  auto fastSchedule = buildGreedyCompressorSchedule(
      op, context, options.fastCompressorStageDelay);
  auto areaSchedule = buildGreedyCompressorSchedule(
      op, context, options.areaCompressorStageDelay);

  bool useArea = options.preferAreaWhenSlackAllows &&
                 scheduleMeetsRequiredTimes(op, context, areaSchedule);
  addGreedyCompressorArcs(semantics, op, useArea ? areaSchedule : fastSchedule,
                          useArea ? "compressor_greedy_area"
                                  : "compressor_greedy_fast");
  return semantics;
}

static TimingSemantics
describePartialProduct(Operation *op,
                       const DatapathTimingSemanticsOptions &options) {
  TimingSemantics semantics;
  semantics.op = op;
  if (op->getNumOperands() < 2)
    return semantics;

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(op->getNumOperands() - 1);
  size_t lhsWidth = getBitWidth(lhs);
  size_t rhsWidth = getBitWidth(rhs);

  for (auto [resultIndex, result] : llvm::enumerate(op->getResults())) {
    uint32_t lhsBit =
        lhsWidth ? std::min<uint32_t>(resultIndex, lhsWidth - 1) : 0;
    for (uint32_t bit = 0, e = getBitWidth(result); bit < e; ++bit) {
      if (lhsWidth)
        addArc(semantics, lhs, lhsBit, result, bit, op, 0,
               static_cast<int32_t>(resultIndex), options.partialProductDelay,
               "partial_product");
      if (rhsWidth)
        addArc(semantics, rhs, std::min<uint32_t>(bit, rhsWidth - 1), result,
               bit, op, static_cast<int32_t>(op->getNumOperands() - 1),
               static_cast<int32_t>(resultIndex), options.partialProductDelay,
               "partial_product");
    }
  }

  return semantics;
}

static TimingSemantics
describePosPartialProduct(Operation *op,
                          const DatapathTimingSemanticsOptions &options) {
  TimingSemantics semantics;
  semantics.op = op;

  for (auto [resultIndex, result] : llvm::enumerate(op->getResults())) {
    for (uint32_t bit = 0, e = getBitWidth(result); bit < e; ++bit) {
      for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
        size_t operandWidth = getBitWidth(operand);
        if (operandWidth == 0)
          continue;
        addArc(semantics, operand, std::min<uint32_t>(bit, operandWidth - 1),
               result, bit, op, static_cast<int32_t>(operandIndex),
               static_cast<int32_t>(resultIndex), options.partialProductDelay,
               "partial_product");
      }
    }
  }

  return semantics;
}

bool DatapathTimingSemanticsProvider::handles(Operation *op) const {
  if (!op)
    return false;
  StringRef name = op->getName().getStringRef();
  return name == "datapath.compress" || name == "datapath.partial_product" ||
         name == "datapath.pos_partial_product";
}

FailureOr<TimingSemantics>
DatapathTimingSemanticsProvider::refine(Operation *op,
                                        const TimingDynamicContext &context) const {
  StringRef name = op->getName().getStringRef();
  if (name == "datapath.compress")
    return describeCompressor(op, context, options);
  if (name == "datapath.partial_product")
    return describePartialProduct(op, options);
  if (name == "datapath.pos_partial_product")
    return describePosPartialProduct(op, options);
  return failure();
}
