//===- ConstructLEC.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_CONSTRUCTLEC
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// ConstructLEC pass
//===----------------------------------------------------------------------===//

namespace {
struct ConstructLECPass
    : public circt::impl::ConstructLECBase<ConstructLECPass> {
  using circt::impl::ConstructLECBase<ConstructLECPass>::ConstructLECBase;
  void runOnOperation() override;
  hw::HWModuleOp lookupModule(StringRef name);
  Value constructMiter(OpBuilder builder, Location loc, hw::HWModuleOp moduleA,
                       hw::HWModuleOp moduleB, bool withResult);
  LogicalResult constructSequentialLEC(OpBuilder &builder, Location loc,
                                       hw::HWModuleOp moduleA,
                                       hw::HWModuleOp moduleB, bool withResult,
                                       SmallVectorImpl<Value> *results);
};

struct NamedState {
  Operation *op;
  Value result;
  StringAttr name;
  Value clock;
  Value enable;
  Value reset;
  Value resetValue;
  Value nextValue;
  unsigned latency;
  std::optional<APInt> initialValue;
  bool isAsyncReset;
  std::optional<FlatSymbolRefAttr> arcName;
  SmallVector<Value> arcInputs;
  unsigned arcResultIndex;
};

static StringAttr getStateResultName(arc::StateOp op, unsigned index) {
  if (auto names = op->getAttrOfType<ArrayAttr>("names")) {
    if (index < names.size())
      if (auto name = dyn_cast<StringAttr>(names[index]))
        if (!name.getValue().empty())
          return name;
  }
  if (index == 0)
    if (auto name = op->getAttrOfType<StringAttr>("name"))
      if (!name.getValue().empty())
        return name;
  return {};
}

static StringAttr getRegisterName(seq::CompRegOp op) {
  if (auto name = op.getName())
    if (!name->empty())
      return StringAttr::get(op.getContext(), *name);
  return {};
}

static StringAttr getRegisterName(seq::CompRegClockEnabledOp op) {
  if (auto name = op.getName())
    if (!name->empty())
      return StringAttr::get(op.getContext(), *name);
  return {};
}

static StringAttr getRegisterName(seq::FirRegOp op) {
  if (!op.getName().empty())
    return StringAttr::get(op.getContext(), op.getName());
  return {};
}

static FailureOr<APInt> getConstInt(Value value);

static FailureOr<llvm::MapVector<StringAttr, NamedState>>
collectNamedStates(hw::HWModuleOp module) {
  llvm::MapVector<StringAttr, NamedState> states;
  unsigned unnamedRegIndex = 0;
  for (Operation &op : module.getBodyBlock()->without_terminator()) {
    if (auto state = dyn_cast<arc::StateOp>(&op)) {
      for (auto [index, result] : llvm::enumerate(state.getResults())) {
        auto name = getStateResultName(state, index);
        if (!name) {
          state.emitError()
              << "sequential arc-state mode requires every state result to "
                 "have a stable name";
          return failure();
        }
        std::optional<APInt> initialValue;
        if (!state.getInitials().empty()) {
          auto initial = getConstInt(state.getInitials()[index]);
          if (failed(initial)) {
            state.emitError()
                << "sequential arc-state mode requires state initial values "
                   "to be constant integers";
            return failure();
          }
          initialValue = *initial;
        }
        auto [it, inserted] = states.insert(
            {name, NamedState{state, result, name, state.getClock(),
                              state.getEnable(), state.getReset(), Value{},
                              Value{}, state.getLatency(), initialValue, false,
                              state.getArcAttr(),
                              SmallVector<Value>(state.getInputs()),
                              static_cast<unsigned>(index)}});
        if (!inserted) {
          state.emitError()
              << "duplicate state name '" << name.getValue() << "'";
          return failure();
        }
      }
      continue;
    }

    auto addRegister = [&](auto reg, Value nextValue, Value clock, Value enable,
                           Value reset, Value resetValue,
                           std::optional<APInt> initialValue,
                           bool isAsyncReset) -> LogicalResult {
      auto name = getRegisterName(reg);
      if (!name) {
        auto generatedName = ("reg_" + Twine(unnamedRegIndex++)).str();
        name = StringAttr::get(module.getContext(), generatedName);
      }
      auto [it, inserted] = states.insert({name, NamedState{reg,
                                                            reg.getResult(),
                                                            name,
                                                            clock,
                                                            enable,
                                                            reset,
                                                            resetValue,
                                                            nextValue,
                                                            1,
                                                            initialValue,
                                                            isAsyncReset,
                                                            std::nullopt,
                                                            {},
                                                            0}});
      if (!inserted) {
        reg.emitError() << "duplicate state name '" << name.getValue() << "'";
        return failure();
      }
      return success();
    };

    if (auto reg = dyn_cast<seq::CompRegOp>(&op)) {
      std::optional<APInt> initialValue;
      if (auto initial = reg.getInitialValue()) {
        auto constInitial =
            getConstInt(circt::seq::unwrapImmutableValue(initial));
        if (failed(constInitial)) {
          reg.emitError() << "sequential arc-state mode requires register "
                             "initial values to be constant integers";
          return failure();
        }
        initialValue = *constInitial;
      }
      if (failed(addRegister(reg, reg.getInput(), reg.getClk(), Value{},
                             reg.getReset(), reg.getResetValue(), initialValue,
                             false)))
        return failure();
      continue;
    }

    if (auto reg = dyn_cast<seq::CompRegClockEnabledOp>(&op)) {
      std::optional<APInt> initialValue;
      if (auto initial = reg.getInitialValue()) {
        auto constInitial =
            getConstInt(circt::seq::unwrapImmutableValue(initial));
        if (failed(constInitial)) {
          reg.emitError() << "sequential arc-state mode requires register "
                             "initial values to be constant integers";
          return failure();
        }
        initialValue = *constInitial;
      }
      if (failed(addRegister(reg, reg.getInput(), reg.getClk(),
                             reg.getClockEnable(), reg.getReset(),
                             reg.getResetValue(), initialValue, false)))
        return failure();
      continue;
    }

    if (auto reg = dyn_cast<seq::FirRegOp>(&op)) {
      std::optional<APInt> initialValue;
      if (auto preset = reg.getPreset())
        initialValue = *preset;
      if (failed(addRegister(reg, reg.getNext(), reg.getClk(), Value{},
                             reg.getReset(), reg.getResetValue(), initialValue,
                             reg.getIsAsync())))
        return failure();
    }
  }
  return states;
}

static LogicalResult verifyNoNestedStatefulInstances(hw::HWModuleOp module) {
  auto root = dyn_cast<ModuleOp>(module->getParentOp());
  if (!root)
    return module.emitError(
        "expected compared module to be nested in a module");

  DenseSet<Operation *> visited;
  SmallVector<hw::HWModuleOp> worklist;
  visited.insert(module);
  worklist.push_back(module);

  auto hasUnsupportedNestedState = [&](hw::HWModuleOp child,
                                       hw::InstanceOp inst) -> LogicalResult {
    WalkResult result =
        child.walk([&](Operation *op) {
          if (isa<arc::StateOp, arc::MemoryOp, arc::MemoryReadOp,
                  arc::MemoryWriteOp, arc::MemoryReadPortOp,
                  arc::MemoryWritePortOp>(op)) {
            inst.emitError()
                << "sequential arc-state mode does not yet support nested "
                   "stateful instances";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    return result.wasInterrupted() ? failure() : success();
  };

  while (!worklist.empty()) {
    auto current = worklist.pop_back_val();
    for (Operation &op : current.getBodyBlock()->without_terminator()) {
      auto inst = dyn_cast<hw::InstanceOp>(&op);
      if (!inst)
        continue;
      auto child =
          root.lookupSymbol<hw::HWModuleOp>(inst.getModuleNameAttr().getAttr());
      if (!child)
        return inst.emitError() << "failed to resolve instance target '"
                                << inst.getModuleName() << "'";
      if (!visited.insert(child).second)
        continue;
      if (failed(hasUnsupportedNestedState(child, inst)))
        return failure();
      worklist.push_back(child);
    }
  }
  return success();
}

static FailureOr<APInt> getConstInt(Value value) {
  if (auto constant = value.getDefiningOp<hw::ConstantOp>())
    return constant.getValue();
  return failure();
}

static FailureOr<unsigned> getModuleInputIndex(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return failure();
  if (!isa<hw::HWModuleOp>(arg.getOwner()->getParentOp()))
    return failure();
  return arg.getArgNumber();
}

static LogicalResult verifyStateCompatibility(NamedState lhs, NamedState rhs) {
  auto *lhsOp = lhs.op;
  if (lhs.result.getType() != rhs.result.getType())
    return lhsOp->emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched types";
  if (lhs.latency != rhs.latency)
    return lhsOp->emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched latencies";
  if (lhs.latency != 1)
    return lhsOp->emitError()
           << "state '" << lhs.name.getValue() << "' uses latency "
           << lhs.latency << "; only latency-1 states are currently supported";
  if (static_cast<bool>(lhs.clock) != static_cast<bool>(rhs.clock))
    return lhsOp->emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched clocks";
  if (auto lhsClock = lhs.clock) {
    auto lhsIndex = getModuleInputIndex(lhsClock);
    auto rhsIndex = getModuleInputIndex(rhs.clock);
    if (failed(lhsIndex) || failed(rhsIndex))
      return lhsOp->emitError()
             << "state '" << lhs.name.getValue()
             << "' must be clocked directly by a module input";
    if (*lhsIndex != *rhsIndex)
      return lhsOp->emitError() << "state '" << lhs.name.getValue()
                                << "' has mismatched clock inputs";
  }
  if (static_cast<bool>(lhs.enable) != static_cast<bool>(rhs.enable))
    return lhsOp->emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched enables";
  if (static_cast<bool>(lhs.reset) != static_cast<bool>(rhs.reset))
    return lhsOp->emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched resets";
  if (lhs.isAsyncReset != rhs.isAsyncReset)
    return lhsOp->emitError() << "state '" << lhs.name.getValue()
                              << "' has mismatched reset synchrony";
  if (lhs.initialValue != rhs.initialValue)
    return lhsOp->emitError()
           << "state '" << lhs.name.getValue()
           << "' has unsupported or mismatched initial values";
  return success();
}

static bool isValueInsideDefine(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isa_and_nonnull<arc::DefineOp>(arg.getOwner()->getParentOp());
  Operation *def = value.getDefiningOp();
  return def && isa<arc::DefineOp>(def->getParentOp());
}

namespace {
struct RegionMaterializer {
  RegionMaterializer(OpBuilder &builder, ModuleOp rootModule,
                     hw::HWModuleOp module, ValueRange regionArgs,
                     ArrayRef<unsigned> inputIndices,
                     ArrayRef<NamedState> states)
      : builder(builder), rootModule(rootModule), module(module) {
    for (auto [index, inputIndex] : llvm::enumerate(inputIndices))
      moduleInputs[module.getBodyBlock()->getArgument(inputIndex)] =
          regionArgs[index];
    for (auto [index, state] : llvm::enumerate(states))
      stateCutpoints[state.result] = regionArgs[inputIndices.size() + index];
  }

  FailureOr<Value> materialize(Value value) {
    DenseMap<Value, Value> emptyBindings;
    return materialize(value, emptyBindings);
  }

  FailureOr<Value> materializeEffectiveNextState(NamedState state) {
    FailureOr<Value> next = failure();
    if (state.arcName)
      next = materializeArcResult(*state.arcName, state.arcInputs,
                                  state.arcResultIndex);
    else
      next = materialize(state.nextValue);
    if (failed(next))
      return failure();
    Value current = stateCutpoints.lookup(state.result);
    assert(current && "state cutpoint must exist");
    if (auto enable = state.enable) {
      auto mappedEnable = materialize(enable);
      if (failed(mappedEnable))
        return failure();
      next = comb::MuxOp::create(builder, state.op->getLoc(), current.getType(),
                                 *mappedEnable, *next, current)
                 .getResult();
    }
    if (auto reset = state.reset) {
      if (state.isAsyncReset) {
        state.op->emitError("async resets are not yet supported in sequential "
                            "arc-state mode");
        return failure();
      }
      auto mappedReset = materialize(reset);
      if (failed(mappedReset))
        return failure();
      auto intType = dyn_cast<IntegerType>(current.getType());
      if (!intType) {
        state.op->emitError()
            << "reset handling currently requires integer state types";
        return failure();
      }
      auto mappedResetValue = materialize(state.resetValue);
      if (failed(mappedResetValue))
        return failure();
      next = comb::MuxOp::create(builder, state.op->getLoc(), current.getType(),
                                 *mappedReset, *mappedResetValue, *next)
                 .getResult();
    }
    return next;
  }

private:
  FailureOr<Value> materialize(Value value, DenseMap<Value, Value> &bindings) {
    if (auto it = bindings.find(value); it != bindings.end())
      return it->second;
    if (!isValueInsideDefine(value))
      if (auto it = cache.find(value); it != cache.end())
        return it->second;

    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto mapped = moduleInputs.lookup(arg);
      if (mapped)
        return mapped;
      module.emitError(
          "unsupported block argument while materializing sequential cone");
      return failure();
    }

    Operation *def = value.getDefiningOp();
    if (!def) {
      module.emitError("encountered unsupported external SSA value");
      return failure();
    }

    if (auto state = dyn_cast<arc::StateOp>(def)) {
      auto mapped = stateCutpoints.lookup(value);
      if (!mapped) {
        state.emitError() << "encountered unmatched state while materializing "
                             "sequential cone";
        return failure();
      }
      if (!isValueInsideDefine(value))
        cache[value] = mapped;
      return mapped;
    }

    if (isa<seq::CompRegOp, seq::CompRegClockEnabledOp, seq::FirRegOp>(def)) {
      auto mapped = stateCutpoints.lookup(value);
      if (!mapped) {
        def->emitError("encountered unmatched sequential state while "
                       "materializing sequential cone");
        return failure();
      }
      cache[value] = mapped;
      return mapped;
    }

    if (auto call = dyn_cast<arc::CallOp>(def))
      return materializeArcResult(call.getArcAttr(), call.getInputs(),
                                  cast<OpResult>(value).getResultNumber());

    if (isa<arc::MemoryOp, arc::MemoryReadOp, arc::MemoryWriteOp,
            arc::MemoryReadPortOp, arc::MemoryWritePortOp>(def)) {
      def->emitError("sequential arc-state mode does not yet support memories");
      return failure();
    }

    if (def->getNumRegions() != 0) {
      def->emitError("unsupported region operation in sequential cone");
      return failure();
    }

    IRMapping mapping;
    for (Value operand : def->getOperands()) {
      auto mappedOperand = materialize(operand, bindings);
      if (failed(mappedOperand))
        return failure();
      mapping.map(operand, *mappedOperand);
    }
    Operation *cloned = builder.clone(*def, mapping);
    for (auto [oldResult, newResult] :
         llvm::zip(def->getResults(), cloned->getResults()))
      if (!isValueInsideDefine(oldResult))
        cache[oldResult] = newResult;
    return cloned->getResult(cast<OpResult>(value).getResultNumber());
  }

  FailureOr<Value> materializeArcResult(FlatSymbolRefAttr arcName,
                                        ValueRange operands,
                                        unsigned resultIndex) {
    auto arc = SymbolTable::lookupNearestSymbolFrom<arc::DefineOp>(rootModule,
                                                                   arcName);
    if (!arc) {
      rootModule.emitError()
          << "failed to resolve arc symbol '" << arcName.getValue() << "'";
      return failure();
    }
    auto *term = arc.getBodyBlock().getTerminator();
    if (resultIndex >= term->getNumOperands()) {
      arc.emitError("arc result index out of range");
      return failure();
    }
    DenseMap<Value, Value> bindings;
    for (auto [arg, operand] : llvm::zip(arc.getArguments(), operands)) {
      auto mappedOperand = materialize(operand);
      if (failed(mappedOperand))
        return failure();
      bindings[arg] = *mappedOperand;
    }
    return materialize(term->getOperand(resultIndex), bindings);
  }

  OpBuilder &builder;
  ModuleOp rootModule;
  hw::HWModuleOp module;
  DenseMap<Value, Value> moduleInputs;
  DenseMap<Value, Value> stateCutpoints;
  DenseMap<Value, Value> cache;
};
} // namespace
} // namespace

static Value lookupOrCreateStringGlobal(OpBuilder &builder, ModuleOp moduleOp,
                                        StringRef str) {
  Location loc = moduleOp.getLoc();
  auto global = moduleOp.lookupSymbol<LLVM::GlobalOp>(str);
  if (!global) {
    OpBuilder b = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto arrayTy = LLVM::LLVMArrayType::get(b.getI8Type(), str.size() + 1);
    global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/true, LLVM::linkage::Linkage::Private,
        str, StringAttr::get(b.getContext(), Twine(str).concat(Twine('\00'))));
  }

  // FIXME: sanity check the fetched global: do all the attributes match what
  // we expect?

  return LLVM::AddressOfOp::create(builder, loc, global);
}

hw::HWModuleOp ConstructLECPass::lookupModule(StringRef name) {
  Operation *expectedModule = SymbolTable::lookupNearestSymbolFrom(
      getOperation(), StringAttr::get(&getContext(), name));
  if (!expectedModule || !isa<hw::HWModuleOp>(expectedModule)) {
    getOperation().emitError("module named '") << name << "' not found";
    return {};
  }
  return cast<hw::HWModuleOp>(expectedModule);
}

Value ConstructLECPass::constructMiter(OpBuilder builder, Location loc,
                                       hw::HWModuleOp moduleA,
                                       hw::HWModuleOp moduleB,
                                       bool withResult) {

  // Create the miter circuit that return equivalence result.
  auto lecOp =
      verif::LogicEquivalenceCheckingOp::create(builder, loc, withResult);

  builder.cloneRegionBefore(moduleA.getBody(), lecOp.getFirstCircuit(),
                            lecOp.getFirstCircuit().end());
  builder.cloneRegionBefore(moduleB.getBody(), lecOp.getSecondCircuit(),
                            lecOp.getSecondCircuit().end());

  moduleA->erase();
  if (moduleA != moduleB)
    moduleB->erase();

  {
    auto *term = lecOp.getFirstCircuit().front().getTerminator();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(term);
    verif::YieldOp::create(builder, loc, term->getOperands());
    term->erase();
    term = lecOp.getSecondCircuit().front().getTerminator();
    builder.setInsertionPoint(term);
    verif::YieldOp::create(builder, loc, term->getOperands());
    term->erase();
  }

  sortTopologically(&lecOp.getFirstCircuit().front());
  sortTopologically(&lecOp.getSecondCircuit().front());

  return withResult ? lecOp.getIsProven() : Value{};
}

void ConstructLECPass::runOnOperation() {
  // Create necessary function declarations and globals
  OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
  Location loc = getOperation()->getLoc();

  // Lookup the modules.
  auto moduleA = lookupModule(firstModule);
  if (!moduleA)
    return signalPassFailure();
  auto moduleB = lookupModule(secondModule);
  if (!moduleB)
    return signalPassFailure();

  if (moduleA.getModuleType() != moduleB.getModuleType()) {
    moduleA.emitError("module's IO types don't match second modules: ")
        << moduleA.getModuleType() << " vs " << moduleB.getModuleType();
    return signalPassFailure();
  }

  if (sequentialMode == lec::SequentialModeEnum::ArcState) {
    if (insertMode == lec::InsertAdditionalModeEnum::None) {
      if (failed(constructSequentialLEC(builder, loc, moduleA, moduleB,
                                        /*withResult=*/false,
                                        /*results=*/nullptr)))
        return signalPassFailure();
      return;
    }
  }

  // Only construct the miter with no additional insertions.
  if (insertMode == lec::InsertAdditionalModeEnum::None) {
    constructMiter(builder, loc, moduleA, moduleB, /*withResult*/ false);
    return;
  }

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> printfFunc;
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  // Lookup or declare printf function.
  printfFunc = LLVM::lookupOrCreateFn(builder, getOperation(), "printf", ptrTy,
                                      voidTy, true);
  if (failed(printfFunc)) {
    getOperation()->emitError("failed to lookup or create printf");
    return signalPassFailure();
  }

  // Reuse the name of the first module for the entry function, so we don't
  // have to do any uniquing and the LEC driver also already knows this name.
  FunctionType functionType = FunctionType::get(&getContext(), {}, {});
  func::FuncOp entryFunc =
      func::FuncOp::create(builder, loc, firstModule, functionType);

  if (insertMode == lec::InsertAdditionalModeEnum::Main) {
    OpBuilder::InsertionGuard guard(builder);
    auto i32Ty = builder.getI32Type();
    auto mainFunc = func::FuncOp::create(
        builder, loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    func::CallOp::create(builder, loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = LLVM::ConstantOp::create(builder, loc, i32Ty, 0);
    func::ReturnOp::create(builder, loc, constZero);
  }

  builder.createBlock(&entryFunc.getBody());

  Value areEquivalent;
  if (sequentialMode == lec::SequentialModeEnum::ArcState) {
    SmallVector<Value> results;
    if (failed(constructSequentialLEC(builder, loc, moduleA, moduleB,
                                      /*withResult=*/true, &results)))
      return signalPassFailure();
    areEquivalent =
        LLVM::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    for (Value result : results)
      areEquivalent = LLVM::AndOp::create(builder, loc, areEquivalent, result);
  } else {
    // Create the miter circuit that returns equivalence result.
    areEquivalent =
        constructMiter(builder, loc, moduleA, moduleB, /*withResult*/ true);
    assert(!!areEquivalent && "Expected LEC operation with result.");
  }

  // TODO: we should find a more elegant way of reporting the result than
  // already inserting some LLVM here
  Value eqFormatString =
      lookupOrCreateStringGlobal(builder, getOperation(), "c1 == c2\n");
  Value neqFormatString =
      lookupOrCreateStringGlobal(builder, getOperation(), "c1 != c2\n");
  Value formatString = LLVM::SelectOp::create(builder, loc, areEquivalent,
                                              eqFormatString, neqFormatString);
  LLVM::CallOp::create(builder, loc, printfFunc.value(),
                       ValueRange{formatString});

  func::ReturnOp::create(builder, loc, ValueRange{});
}

LogicalResult ConstructLECPass::constructSequentialLEC(
    OpBuilder &builder, Location loc, hw::HWModuleOp moduleA,
    hw::HWModuleOp moduleB, bool withResult, SmallVectorImpl<Value> *results) {
  if (moduleA == moduleB) {
    if (withResult && results) {
      auto proven = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      results->push_back(proven);
    }
    moduleA->erase();
    SmallVector<Operation *> defsToErase;
    for (auto def : getOperation().getOps<arc::DefineOp>())
      defsToErase.push_back(def);
    for (auto *def : defsToErase)
      def->erase();
    return success();
  }

  auto walkForUnsupportedOps = [&](hw::HWModuleOp module) -> LogicalResult {
    WalkResult result = module.walk([&](Operation *op) {
      if (isa<arc::MemoryOp, arc::MemoryReadOp, arc::MemoryWriteOp,
              arc::MemoryReadPortOp, arc::MemoryWritePortOp>(op)) {
        op->emitError(
            "sequential arc-state mode does not yet support memories");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
  };
  if (failed(walkForUnsupportedOps(moduleA)) ||
      failed(walkForUnsupportedOps(moduleB)))
    return failure();
  if (failed(verifyNoNestedStatefulInstances(moduleA)) ||
      failed(verifyNoNestedStatefulInstances(moduleB)))
    return failure();

  auto statesA = collectNamedStates(moduleA);
  auto statesB = collectNamedStates(moduleB);
  if (failed(statesA) || failed(statesB))
    return failure();
  if (statesA->size() != statesB->size())
    return moduleA.emitError("sequential arc-state mode requires the same set "
                             "of named states in both modules");

  SmallVector<NamedState> matchedA;
  SmallVector<NamedState> matchedB;
  for (auto &[name, lhs] : *statesA) {
    auto it = statesB->find(name);
    if (it == statesB->end())
      return moduleA.emitError("state '")
             << name.getValue() << "' is missing in the second module";
    if (failed(verifyStateCompatibility(lhs, it->second)))
      return failure();
    matchedA.push_back(lhs);
    matchedB.push_back(it->second);
  }
  for (auto &[name, rhs] : *statesB)
    if (!statesA->count(name))
      return moduleB.emitError("state '")
             << name.getValue() << "' is missing in the first module";

  SmallVector<unsigned> inputIndices;
  SmallVector<Type> regionArgTypes;
  for (auto [index, type] : llvm::enumerate(moduleA.getInputTypes())) {
    if (isa<seq::ClockType>(type))
      continue;
    inputIndices.push_back(index);
    regionArgTypes.push_back(type);
  }
  for (auto state : matchedA)
    regionArgTypes.push_back(state.result.getType());
  for (Type type : regionArgTypes)
    if (!type)
      return moduleA.emitError("encountered null cone argument type");
  SmallVector<Location> regionArgLocs(regionArgTypes.size(), loc);

  auto addCone = [&](auto lhsBuilderFn, auto rhsBuilderFn) -> LogicalResult {
    auto lecOp =
        verif::LogicEquivalenceCheckingOp::create(builder, loc, withResult);
    auto *lhsBlock = builder.createBlock(&lecOp.getFirstCircuit(), {},
                                         regionArgTypes, regionArgLocs);
    auto *rhsBlock = builder.createBlock(&lecOp.getSecondCircuit(), {},
                                         regionArgTypes, regionArgLocs);
    OpBuilder lhsBuilder = OpBuilder::atBlockBegin(lhsBlock);
    OpBuilder rhsBuilder = OpBuilder::atBlockBegin(rhsBlock);
    RegionMaterializer lhsMaterializer(lhsBuilder, getOperation(), moduleA,
                                       lhsBlock->getArguments(), inputIndices,
                                       matchedA);
    RegionMaterializer rhsMaterializer(rhsBuilder, getOperation(), moduleB,
                                       rhsBlock->getArguments(), inputIndices,
                                       matchedB);
    auto lhsValue = lhsBuilderFn(lhsMaterializer);
    auto rhsValue = rhsBuilderFn(rhsMaterializer);
    if (failed(lhsValue) || failed(rhsValue))
      return failure();
    if ((*lhsValue).getType() != (*rhsValue).getType())
      return moduleA.emitError("encountered mismatched cone types while "
                               "constructing sequential LEC");
    verif::YieldOp::create(lhsBuilder, loc, *lhsValue);
    verif::YieldOp::create(rhsBuilder, loc, *rhsValue);
    sortTopologically(lhsBlock);
    sortTopologically(rhsBlock);
    builder.setInsertionPointAfter(lecOp);
    if (withResult && results)
      results->push_back(lecOp.getIsProven());
    return success();
  };

  auto *outA = moduleA.getBodyBlock()->getTerminator();
  auto *outB = moduleB.getBodyBlock()->getTerminator();
  for (auto pair : llvm::zip(outA->getOperands(), outB->getOperands())) {
    Value lhsValue = std::get<0>(pair);
    Value rhsValue = std::get<1>(pair);
    if (failed(addCone(
            [&](RegionMaterializer &materializer) {
              return materializer.materialize(lhsValue);
            },
            [&](RegionMaterializer &materializer) {
              return materializer.materialize(rhsValue);
            })))
      return failure();
  }

  for (auto pair : llvm::zip(matchedA, matchedB)) {
    NamedState lhsState = std::get<0>(pair);
    NamedState rhsState = std::get<1>(pair);
    if (failed(addCone(
            [&](RegionMaterializer &materializer) {
              return materializer.materializeEffectiveNextState(lhsState);
            },
            [&](RegionMaterializer &materializer) {
              return materializer.materializeEffectiveNextState(rhsState);
            })))
      return failure();
  }

  moduleA->erase();
  if (moduleA != moduleB)
    moduleB->erase();
  SmallVector<Operation *> defsToErase;
  for (auto def : getOperation().getOps<arc::DefineOp>())
    defsToErase.push_back(def);
  for (auto *def : defsToErase)
    def->erase();
  return success();
}
