//===- TimingDesignBuilder.cpp - Build TimingDesign from HW IR -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Timing/TimingDesignBuilder.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/SymbolTable.h"

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::timing;

TimingDesignBuilder::TimingDesignBuilder(
    mlir::MLIRContext *ctx, mlir::StringAttr topModule,
    const llvm::DenseSet<mlir::StringAttr> &libraryCells)
    : ctx(ctx), topModule(topModule), libraryCells(libraryCells) {}

uint32_t TimingDesignBuilder::freshNet() {
  uint32_t id = (uint32_t)design.nets.size();
  design.nets.push_back(TimingNet());
  design.nets.back().id = id;
  return id;
}

void TimingDesignBuilder::diagnose(
    llvm::StringRef message, const std::optional<TimingObjectKey> &object) {
  TimingDiagnostic diag;
  diag.message = mlir::StringAttr::get(ctx, message);
  if (object)
    diag.object = *object;
  diagnostics.unsupported.push_back(std::move(diag));
}

void TimingDesignBuilder::fail(llvm::StringRef message,
                               const std::optional<TimingObjectKey> &object) {
  diagnose(message, object);
  failed = true;
}

static std::optional<unsigned> typeWidth(mlir::Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (isa<seq::ClockType>(type))
    return 1;
  return std::nullopt;
}

std::optional<unsigned> TimingDesignBuilder::bitWidth(mlir::Value value) {
  return typeWidth(value.getType());
}

mlir::FailureOr<TimingDesign> TimingDesignBuilder::build(mlir::ModuleOp top) {
  design = TimingDesign();
  diagnostics = TimingDesignDiagnostics();
  failed = false;
  design.topModule = topModule;
  auto *sym = top.lookupSymbol(topModule);
  auto module = dyn_cast_or_null<hw::HWModuleOp>(sym);
  if (!module || !module.getBodyBlock()) {
    fail("top module not found or has no body");
    return mlir::failure();
  }
  // Top block arguments become primary inputs; results flow to outputs.
  auto *body = module.getBodyBlock();
  SmallVector<SmallVector<uint32_t>> argNets;
  for (auto arg : body->getArguments()) {
    auto width = bitWidth(arg);
    if (!width) {
      fail("unsupported top port type");
      return mlir::failure();
    }
    SmallVector<uint32_t> nets;
    for (unsigned i = 0; i < *width; ++i)
      nets.push_back(freshNet());
    argNets.push_back(std::move(nets));
  }
  auto resultNets = elaborateModule(module, {}, argNets, /*isTop=*/true);
  if (!resultNets)
    return mlir::failure();
  // Record primary input ports in signature order. Block arguments
  // correspond to input ports positionally.
  {
    size_t argPos = 0;
    for (const auto &port : module.getPortList()) {
      if (!port.isInput())
        continue;
      if (argPos >= argNets.size()) {
        fail("top input port without block argument");
        return mlir::failure();
      }
      for (unsigned i = 0; i < argNets[argPos].size(); ++i) {
        TimingPrimaryPort primary;
        primary.port.kind = TimingObjectKind::TopPort;
        primary.port.portName = port.name;
        primary.port.portIndex = (unsigned)port.argNum;
        primary.port.bit = i;
        primary.isInput = true;
        design.ports.push_back(std::move(primary));
      }
      ++argPos;
    }
  }
  // Record primary outputs in signature order, paired with the
  // elaborated result nets.
  {
    size_t resultPos = 0;
    for (const auto &port : module.getPortList()) {
      if (!port.isOutput())
        continue;
      if (resultPos >= resultNets->size()) {
        fail("top output port without elaborated nets");
        return mlir::failure();
      }
      for (unsigned i = 0; i < (*resultNets)[resultPos].size(); ++i) {
        TimingPrimaryPort primary;
        primary.port.kind = TimingObjectKind::TopPort;
        primary.port.portName = port.name;
        primary.port.portIndex = (unsigned)port.argNum;
        primary.port.bit = i;
        primary.isInput = false;
        design.ports.push_back(std::move(primary));
      }
      ++resultPos;
    }
  }
  diagnostics.numCells = design.cells.size();
  diagnostics.numNets = design.nets.size();
  for (const auto &net : design.nets)
    if (net.isConstant)
      ++diagnostics.numConstantNets;
  if (failed)
    return mlir::failure();
  return std::move(design);
}

std::optional<SmallVector<SmallVector<uint32_t>>>
TimingDesignBuilder::elaborateModule(hw::HWModuleOp module,
                                     ArrayRef<mlir::StringAttr> instancePath,
                                     ArrayRef<SmallVector<uint32_t>> argNets,
                                     bool isTop) {
  (void)isTop;
  if (!activeModules.insert(module.getModuleNameAttr()).second) {
    fail("recursive instantiation");
    return std::nullopt;
  }
  struct Guard {
    TimingDesignBuilder *self;
    mlir::StringAttr name;
    ~Guard() { self->activeModules.erase(name); }
  } guard{this, module.getModuleNameAttr()};

  auto *body = module.getBodyBlock();
  if (!body) {
    fail("module without body");
    return std::nullopt;
  }

  // Value to per-bit nets within this instantiation. Block arguments alias
  // the caller's nets; internal values get fresh per-instantiation nets so
  // shared modules still flatten with instance-specific loads.
  llvm::DenseMap<mlir::Value, SmallVector<uint32_t>> valueNets;
  for (auto [arg, nets] : llvm::zip(body->getArguments(), argNets))
    valueNets[arg] = nets;

  // Persistent key prefix for everything created in this instantiation.
  SmallVector<mlir::StringAttr> prefix(instancePath.begin(),
                                       instancePath.end());

  for (auto &op : body->getOperations()) {
    // Hierarchical instance: recurse into design modules, emit gates for
    // library cells, reject anything else.
    if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
      // Collect operand nets (dominance guarantees they exist).
      SmallVector<SmallVector<uint32_t>> operandNets;
      bool operandsOk = true;
      for (auto operand : inst.getOperands()) {
        auto it = valueNets.find(operand);
        if (it == valueNets.end()) {
          fail("operand without nets");
          operandsOk = false;
          break;
        }
        operandNets.push_back(it->second);
      }
      if (!operandsOk)
        return std::nullopt;
      if (libraryCells.contains(
              mlir::StringAttr::get(ctx, inst.getModuleName()))) {
        // Library gate: bind pins by port order with names for Liberty
        // binding. Inout ports are unsupported (lower first).
        auto top = inst->getParentOfType<mlir::ModuleOp>();
        auto *sym =
            top.lookupSymbol(mlir::StringAttr::get(ctx, inst.getModuleName()));
        auto cellModule = dyn_cast_or_null<hw::HWModuleOp>(sym);
        if (!cellModule) {
          fail("library cell module not found");
          return std::nullopt;
        }
        SmallVector<hw::PortInfo> inputs, outputs;
        for (const auto &port : cellModule.getPortList()) {
          if (port.isInput())
            inputs.push_back(port);
          else if (port.isOutput())
            outputs.push_back(port);
          else {
            fail("inout port unsupported for timing graph");
            return std::nullopt;
          }
        }
        if (inputs.size() != inst.getNumOperands() ||
            outputs.size() != inst.getNumResults()) {
          fail("instance port count mismatch");
          return std::nullopt;
        }
        uint32_t id = (uint32_t)design.cells.size();
        TimingCellInstance cell;
        cell.id = id;
        cell.cellType = mlir::StringAttr::get(ctx, inst.getModuleName());
        cell.instance.kind = TimingObjectKind::Instance;
        cell.instance.instancePath = prefix;
        cell.instance.instancePath.push_back(inst.getInstanceNameAttr());
        cell.instance.moduleName =
            mlir::StringAttr::get(ctx, inst.getModuleName());
        for (size_t pi = 0; pi < inputs.size(); ++pi) {
          auto width = bitWidth(inst.getOperand(pi));
          auto portWidth = typeWidth(inputs[pi].type);
          if (!width || !portWidth || *width != *portWidth) {
            fail("instance operand width mismatch");
            return std::nullopt;
          }
          auto it = valueNets.find(inst.getOperand(pi));
          if (it == valueNets.end()) {
            fail("operand without nets");
            return std::nullopt;
          }
          for (unsigned i = 0; i < *width; ++i) {
            TimingPinConnection pin;
            pin.node = TimingNodeId{id};
            pin.portName = inputs[pi].name;
            pin.portIndex = (unsigned)inputs[pi].argNum;
            pin.isInput = true;
            pin.bit = i;
            cell.pins.push_back(pin);
            design.nets[it->second[i]].loads.push_back(TimingNodeId{id});
          }
        }
        for (size_t pi = 0; pi < outputs.size(); ++pi) {
          auto result = inst.getResult(pi);
          auto width = bitWidth(result);
          auto portWidth = typeWidth(outputs[pi].type);
          if (!width || !portWidth || *width != *portWidth) {
            fail("instance result width mismatch");
            return std::nullopt;
          }
          SmallVector<uint32_t> bits;
          for (unsigned i = 0; i < *width; ++i) {
            uint32_t net = freshNet();
            TimingPinConnection pin;
            pin.node = TimingNodeId{id};
            pin.portName = outputs[pi].name;
            pin.portIndex = (unsigned)outputs[pi].argNum;
            pin.isInput = false;
            pin.bit = i;
            cell.pins.push_back(pin);
            auto &netRef = design.nets[net];
            if (netRef.driver) {
              fail("multiple drivers on net");
              return std::nullopt;
            }
            netRef.driver = TimingNodeId{id};
            bits.push_back(net);
          }
          valueNets[result] = std::move(bits);
        }
        design.cells.push_back(std::move(cell));
        continue;
      }
      // Design submodule: recurse with the extended path.
      auto top = inst->getParentOfType<mlir::ModuleOp>();
      auto *sym =
          top.lookupSymbol(mlir::StringAttr::get(ctx, inst.getModuleName()));
      auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(sym);
      if (!targetModule || !targetModule.getBodyBlock()) {
        TimingObjectKey key;
        key.kind = TimingObjectKind::Instance;
        key.instancePath = prefix;
        key.instancePath.push_back(inst.getInstanceNameAttr());
        key.moduleName = mlir::StringAttr::get(ctx, inst.getModuleName());
        fail("unmapped extern instance without timing model", key);
        return std::nullopt;
      }
      SmallVector<mlir::StringAttr> childPath = prefix;
      childPath.push_back(inst.getInstanceNameAttr());
      auto childResults =
          elaborateModule(targetModule, childPath, operandNets, false);
      if (!childResults)
        return std::nullopt;
      unsigned resultIdx = 0;
      for (auto result : inst.getResults()) {
        if (resultIdx >= childResults->size()) {
          fail("arity mismatch elaborating instance");
          return std::nullopt;
        }
        valueNets[result] = (*childResults)[resultIdx++];
      }
      continue;
    }
    // Transparent wiring: extract (bits shift by lowBit).
    if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
      auto it = valueNets.find(extract.getInput());
      auto width = bitWidth(extract.getResult());
      if (it == valueNets.end() || !width) {
        fail("extract without input nets");
        return std::nullopt;
      }
      unsigned low = extract.getLowBit();
      SmallVector<uint32_t> bits;
      for (unsigned i = 0; i < *width; ++i)
        bits.push_back(it->second[low + i]);
      valueNets[extract.getResult()] = std::move(bits);
      continue;
    }
    // Transparent wiring: concat (input 0 is MSB).
    if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
      unsigned total = 0;
      for (auto input : concat.getInputs()) {
        auto it = valueNets.find(input);
        auto width = bitWidth(input);
        if (it == valueNets.end() || !width) {
          fail("concat without input nets");
          return std::nullopt;
        }
        total += *width;
      }
      auto resultWidth = bitWidth(concat.getResult());
      if (!resultWidth || *resultWidth != total) {
        fail("concat width mismatch");
        return std::nullopt;
      }
      SmallVector<uint32_t> ordered(total, 0);
      unsigned offset = total;
      for (auto input : concat.getInputs()) {
        unsigned width = *bitWidth(input);
        offset -= width;
        const auto &inputNets = valueNets[input];
        for (unsigned i = 0; i < width; ++i)
          ordered[offset + i] = inputNets[i];
      }
      valueNets[concat.getResult()] = std::move(ordered);
      continue;
    }
    // Transparent wiring: replicate (result repeats input bits LSB-first).
    if (auto replicate = dyn_cast<comb::ReplicateOp>(op)) {
      auto it = valueNets.find(replicate.getInput());
      auto inWidth = bitWidth(replicate.getInput());
      auto outWidth = bitWidth(replicate.getResult());
      if (it == valueNets.end() || !inWidth || !outWidth ||
          *outWidth % *inWidth != 0) {
        fail("replicate without input nets or bad multiple");
        return std::nullopt;
      }
      SmallVector<uint32_t> bits;
      for (unsigned i = 0; i < *outWidth; ++i)
        bits.push_back(it->second[i % *inWidth]);
      valueNets[replicate.getResult()] = std::move(bits);
      continue;
    }
    // Transparent wiring: constants drive tied nets per bit.
    if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
      auto width = bitWidth(constant.getResult());
      if (!width) {
        fail("constant without width");
        return std::nullopt;
      }
      auto value = constant.getValue();
      SmallVector<uint32_t> bits;
      for (unsigned i = 0; i < *width; ++i) {
        uint32_t id = freshNet();
        design.nets[id].isConstant = true;
        design.nets[id].constantValue = value[i];
        bits.push_back(id);
      }
      valueNets[constant.getResult()] = std::move(bits);
      continue;
    }
    if (isa<hw::OutputOp>(op))
      continue;
    // Anything else (arithmetic, muxes, sequential ops, simulators) is
    // unsupported: fail closed, never optimistic.
    fail("unsupported operation for timing graph");
    return std::nullopt;
  }
  // Collect terminator results in order.
  SmallVector<SmallVector<uint32_t>> resultNets;
  if (auto output = dyn_cast<hw::OutputOp>(body->getTerminator())) {
    for (auto operand : output->getOperands()) {
      auto it = valueNets.find(operand);
      if (it == valueNets.end()) {
        fail("output without nets");
        return std::nullopt;
      }
      resultNets.push_back(it->second);
    }
  }
  return resultNets;
}
