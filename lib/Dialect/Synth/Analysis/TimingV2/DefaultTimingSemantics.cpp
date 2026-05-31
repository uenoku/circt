//===- DefaultTimingSemantics.cpp - CIRCT timing semantics ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the built-in CIRCT operation semantics for TimingV2. The
// flat timing core consumes only generic TimingSemantics descriptors; concrete
// dialect dispatch stays here so users can replace this layer without changing
// graph construction, propagation, repair, or reporting.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/TimingV2/FlatTiming.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
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

static std::string getValueName(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto module = dyn_cast_or_null<hw::HWModuleOp>(
            arg.getOwner()->getParentOp())) {
      if (arg.getArgNumber() < module.getNumInputPorts())
        return module.getInputName(arg.getArgNumber()).str();
    }
    return ("arg" + Twine(arg.getArgNumber())).str();
  }

  auto *op = value.getDefiningOp();
  if (!op)
    return "value";
  if (auto name = op->getAttrOfType<StringAttr>("sv.namehint"))
    return name.getValue().str();
  if (auto name = op->getAttrOfType<StringAttr>("hw.name"))
    return name.getValue().str();
  if (auto reg = dyn_cast<seq::CompRegOp>(op))
    if (auto name = reg.getNameAttr())
      return name.getValue().str();
  if (auto reg = dyn_cast<seq::FirRegOp>(op))
    if (auto name = reg.getNameAttr())
      return name.getValue().str();
  return op->getName().stripDialect().str();
}

static std::string makeBitName(StringRef base, uint32_t bit) {
  return (base + "[" + Twine(bit) + "]").str();
}

static TimingSemanticPoint
point(Value value, uint32_t bit,
      TimingPointKind kind = TimingPointKind::ValueBit, StringRef name = {},
      Operation *owner = nullptr) {
  TimingSemanticPoint point;
  point.value = value;
  point.bit = bit;
  point.kind = kind;
  point.name = name.str();
  point.owner = owner;
  return point;
}

static void addArc(TimingSemantics &semantics, TimingSemanticPoint from,
                   TimingSemanticPoint to, Operation *op, int32_t inputIndex,
                   int32_t outputIndex, TimingArcKind kind, StringRef token,
                   std::optional<int64_t> fixedDelay = std::nullopt) {
  TimingSemanticArc arc;
  arc.from = std::move(from);
  arc.to = std::move(to);
  arc.kind = kind;
  arc.fixedDelay = fixedDelay;
  arc.op = op;
  arc.inputIndex = inputIndex;
  arc.outputIndex = outputIndex;
  arc.token = token.str();
  semantics.arcs.push_back(std::move(arc));
}

static void addSameBitArcs(TimingSemantics &semantics, Operation *op,
                           StringRef token) {
  for (auto [resultIndex, result] : llvm::enumerate(op->getResults())) {
    size_t resultWidth = getBitWidth(result);
    for (uint32_t bit = 0; bit < resultWidth; ++bit) {
      auto to = point(result, bit, TimingPointKind::ValueBit, {}, op);
      for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
        size_t operandWidth = getBitWidth(operand);
        if (operandWidth == 0)
          continue;
        uint32_t sourceBit = std::min<uint32_t>(bit, operandWidth - 1);
        addArc(semantics, point(operand, sourceBit), to, op,
               static_cast<int32_t>(operandIndex),
               static_cast<int32_t>(resultIndex), TimingArcKind::Data, token);
      }
    }
  }
}

static void
addCarryPrefixArcs(TimingSemantics &semantics, Operation *op, ValueRange inputs,
                   Value result, StringRef token,
                   std::optional<int64_t> fixedDelay = std::nullopt) {
  size_t resultWidth = getBitWidth(result);
  for (uint32_t bit = 0; bit < resultWidth; ++bit) {
    auto to = point(result, bit, TimingPointKind::ValueBit, {}, op);
    for (auto [operandIndex, operand] : llvm::enumerate(inputs)) {
      size_t operandWidth = getBitWidth(operand);
      if (operandWidth == 0)
        continue;
      uint32_t maxSourceBit = std::min<uint32_t>(bit, operandWidth - 1);
      for (uint32_t sourceBit = 0; sourceBit <= maxSourceBit; ++sourceBit)
        addArc(semantics, point(operand, sourceBit), to, op,
               static_cast<int32_t>(operandIndex), 0, TimingArcKind::Data,
               token, fixedDelay);
    }
  }
}

static int64_t getTreeDelay(size_t inputCount) {
  if (inputCount <= 1)
    return 0;
  return llvm::Log2_64_Ceil(inputCount);
}

static void addAllBitArcs(TimingSemantics &semantics, Operation *op,
                          ValueRange inputs, Value result, StringRef token,
                          int64_t delay) {
  for (uint32_t bit = 0, resultWidth = getBitWidth(result); bit < resultWidth;
       ++bit) {
    auto to = point(result, bit, TimingPointKind::ValueBit, {}, op);
    for (auto [operandIndex, operand] : llvm::enumerate(inputs)) {
      for (uint32_t sourceBit = 0, operandWidth = getBitWidth(operand);
           sourceBit < operandWidth; ++sourceBit)
        addArc(semantics, point(operand, sourceBit), to, op,
               static_cast<int32_t>(operandIndex), 0, TimingArcKind::Synthetic,
               token, delay);
    }
  }
}

static void addReductionArcs(TimingSemantics &semantics, Operation *op,
                             ValueRange inputs, Value result, StringRef token,
                             int64_t delay) {
  auto to = point(result, 0, TimingPointKind::ValueBit, {}, op);
  for (auto [operandIndex, operand] : llvm::enumerate(inputs)) {
    for (uint32_t sourceBit = 0, operandWidth = getBitWidth(operand);
         sourceBit < operandWidth; ++sourceBit)
      addArc(semantics, point(operand, sourceBit), to, op,
             static_cast<int32_t>(operandIndex), 0, TimingArcKind::Synthetic,
             token, delay);
  }
}

static void addMultiplyArcs(TimingSemantics &semantics, Operation *op,
                            ValueRange inputs, Value result) {
  for (uint32_t resultBit = 0, resultWidth = getBitWidth(result);
       resultBit < resultWidth; ++resultBit) {
    auto to = point(result, resultBit, TimingPointKind::ValueBit, {}, op);
    int64_t delay = std::max<int64_t>(
        1, getTreeDelay(resultBit + 1) +
               getTreeDelay(std::max<size_t>(1, inputs.size())));
    for (auto [operandIndex, operand] : llvm::enumerate(inputs)) {
      uint32_t operandWidth = getBitWidth(operand);
      if (operandWidth == 0)
        continue;
      uint32_t maxSourceBit = std::min<uint32_t>(resultBit, operandWidth - 1);
      for (uint32_t sourceBit = 0; sourceBit <= maxSourceBit; ++sourceBit)
        addArc(semantics, point(operand, sourceBit), to, op,
               static_cast<int32_t>(operandIndex), 0, TimingArcKind::Synthetic,
               "mul_structural", delay);
    }
  }
}

static void addShiftArcs(TimingSemantics &semantics, Operation *op,
                         StringRef token) {
  Value result = op->getResult(0);
  int64_t delay = std::max<int64_t>(1, getTreeDelay(getBitWidth(result)) + 1);
  addAllBitArcs(semantics, op, op->getOperands(), result, token, delay);
}

static void addRegisterCut(TimingSemantics &semantics, Operation *op,
                           Value input, Value result, StringRef name) {
  std::string outputName = name.empty() ? getValueName(result) : name.str();
  for (uint32_t bit = 0, e = getBitWidth(result); bit < e; ++bit)
    semantics.points.push_back(point(
        result, bit, TimingPointKind::CutStart,
        makeBitName(outputName, bit), op));

  std::string inputName = outputName + "_D";
  for (uint32_t bit = 0, e = getBitWidth(input); bit < e; ++bit) {
    addArc(semantics, point(input, bit),
           point(input, bit, TimingPointKind::CutEnd,
                 makeBitName(inputName, bit), op),
           op, 0, -1, TimingArcKind::Cut, "register_d",
           /*fixedDelay=*/0);
  }
}

FailureOr<TimingSemantics>
DefaultTimingSemanticsProvider::describe(Operation *op) const {
  TimingSemantics semantics;
  semantics.op = op;

  if (!op || op->hasTrait<OpTrait::ConstantLike>())
    return semantics;

  if (auto output = dyn_cast<hw::OutputOp>(op)) {
    auto module = output->getParentOfType<hw::HWModuleOp>();
    for (auto [index, operand] : llvm::enumerate(output.getOperands())) {
      StringRef portName =
          module && index < module.getNumOutputPorts()
              ? module.getOutputName(index)
              : StringRef("out");
      for (uint32_t bit = 0, e = getBitWidth(operand); bit < e; ++bit) {
        addArc(semantics, point(operand, bit),
               point(operand, bit, TimingPointKind::RootOutput,
                     makeBitName(portName, bit), op),
               op, static_cast<int32_t>(index), -1, TimingArcKind::Boundary,
               "root_output", /*fixedDelay=*/0);
      }
    }
    return semantics;
  }

  if (auto reg = dyn_cast<seq::CompRegOp>(op)) {
    addRegisterCut(semantics, op, reg.getInput(), reg.getResult(),
                   reg.getNameAttr() ? reg.getNameAttr().getValue()
                                     : StringRef("reg"));
    return semantics;
  }

  if (auto reg = dyn_cast<seq::FirRegOp>(op)) {
    addRegisterCut(semantics, op, reg.getNext(), reg.getResult(),
                   reg.getNameAttr() ? reg.getNameAttr().getValue()
                                     : StringRef("reg"));
    return semantics;
  }

  if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
    uint32_t resultBit = 0;
    for (Value operand : llvm::reverse(concat.getInputs())) {
      for (uint32_t operandBit = 0, e = getBitWidth(operand); operandBit < e;
           ++operandBit, ++resultBit) {
        addArc(semantics, point(operand, operandBit),
               point(concat.getResult(), resultBit, TimingPointKind::ValueBit,
                     {}, op),
               op, -1, 0, TimingArcKind::Data, "concat",
               /*fixedDelay=*/0);
      }
    }
    return semantics;
  }

  if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
    for (uint32_t bit = 0, e = getBitWidth(extract.getResult()); bit < e;
         ++bit) {
      addArc(semantics, point(extract.getInput(), extract.getLowBit() + bit),
             point(extract.getResult(), bit, TimingPointKind::ValueBit, {}, op),
             op, 0, 0, TimingArcKind::Data, "extract",
             /*fixedDelay=*/0);
    }
    return semantics;
  }

  if (auto replicate = dyn_cast<comb::ReplicateOp>(op)) {
    uint32_t inputWidth = getBitWidth(replicate.getInput());
    for (uint32_t bit = 0, e = getBitWidth(replicate.getResult()); bit < e;
         ++bit) {
      addArc(semantics,
             point(replicate.getInput(), inputWidth ? bit % inputWidth : 0),
             point(replicate.getResult(), bit, TimingPointKind::ValueBit, {},
                   op),
             op, 0, 0, TimingArcKind::Data, "replicate",
             /*fixedDelay=*/0);
    }
    return semantics;
  }

  if (auto mux = dyn_cast<comb::MuxOp>(op)) {
    for (uint32_t bit = 0, e = getBitWidth(mux.getResult()); bit < e; ++bit) {
      auto to = point(mux.getResult(), bit, TimingPointKind::ValueBit, {}, op);
      addArc(semantics, point(mux.getCond(), 0), to, op, 0, 0,
             TimingArcKind::Data, "mux_cond");
      addArc(semantics, point(mux.getTrueValue(), bit), to, op, 1, 0,
             TimingArcKind::Data, "mux_true");
      addArc(semantics, point(mux.getFalseValue(), bit), to, op, 2, 0,
             TimingArcKind::Data, "mux_false");
    }
    return semantics;
  }

  if (auto add = dyn_cast<comb::AddOp>(op)) {
    addCarryPrefixArcs(semantics, op, add.getInputs(), add.getResult(), "add");
    return semantics;
  }

  if (auto sub = dyn_cast<comb::SubOp>(op)) {
    addCarryPrefixArcs(semantics, op, op->getOperands(), sub.getResult(),
                       "sub");
    return semantics;
  }

  if (auto mul = dyn_cast<comb::MulOp>(op)) {
    addMultiplyArcs(semantics, op, mul.getInputs(), mul.getResult());
    return semantics;
  }

  if (isa<comb::DivSOp, comb::DivUOp, comb::ModSOp, comb::ModUOp>(op)) {
    auto result = op->getResult(0);
    addAllBitArcs(semantics, op, op->getOperands(), result,
                  op->getName().stripDialect(),
                  std::max<int64_t>(1, getBitWidth(result)));
    return semantics;
  }

  if (isa<comb::ShlOp, comb::ShrSOp, comb::ShrUOp>(op)) {
    addShiftArcs(semantics, op, op->getName().stripDialect());
    return semantics;
  }

  if (auto icmp = dyn_cast<comb::ICmpOp>(op)) {
    int64_t delay =
        std::max<int64_t>(1, getTreeDelay(getBitWidth(icmp.getLhs())) + 1);
    addReductionArcs(semantics, op, op->getOperands(), icmp.getResult(),
                     "icmp_structural", delay);
    return semantics;
  }

  if (auto parity = dyn_cast<comb::ParityOp>(op)) {
    addReductionArcs(semantics, op, parity.getInput(), parity.getResult(),
                     "parity", getTreeDelay(getBitWidth(parity.getInput())));
    return semantics;
  }

  if (auto truthTable = dyn_cast<comb::TruthTableOp>(op)) {
    addReductionArcs(semantics, op, truthTable.getInputs(),
                     truthTable.getResult(), "truth_table",
                     std::max<int64_t>(1, truthTable.getInputs().size()));
    return semantics;
  }

  if (auto reverse = dyn_cast<comb::ReverseOp>(op)) {
    uint32_t width = getBitWidth(reverse.getResult());
    for (uint32_t bit = 0; bit < width; ++bit)
      addArc(semantics, point(reverse.getInput(), width - 1 - bit),
             point(reverse.getResult(), bit, TimingPointKind::ValueBit, {}, op),
             op, 0, 0, TimingArcKind::Data, "reverse", /*fixedDelay=*/0);
    return semantics;
  }

  StringRef opName = op->getName().getStringRef();
  if (opName == "datapath.partial_product" ||
      opName == "datapath.pos_partial_product") {
    for (auto result : op->getResults())
      addCarryPrefixArcs(semantics, op, op->getOperands(), result,
                         "partial_product");
    return semantics;
  }

  if (opName == "datapath.compress") {
    addSameBitArcs(semantics, op, "compressor_stage");
    return semantics;
  }

  if (isa<comb::AndOp, comb::OrOp, comb::XorOp, synth::aig::AndInverterOp>(
          op)) {
    addSameBitArcs(semantics, op, op->getName().stripDialect());
    return semantics;
  }

  if (isa<hw::WireOp>(op)) {
    addSameBitArcs(semantics, op, "wire");
    return semantics;
  }

  if (opName.starts_with("seq."))
    return semantics;

  addSameBitArcs(semantics, op, "op");
  return semantics;
}
