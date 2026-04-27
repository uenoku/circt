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
  arc::StateOp op;
  unsigned resultIndex;
  Value result;
  StringAttr name;
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

static FailureOr<llvm::MapVector<StringAttr, NamedState>>
collectNamedStates(hw::HWModuleOp module) {
  llvm::MapVector<StringAttr, NamedState> states;
  for (Operation &op : module.getBodyBlock()->without_terminator()) {
    auto state = dyn_cast<arc::StateOp>(&op);
    if (!state)
      continue;
    for (auto [index, result] : llvm::enumerate(state.getResults())) {
      auto name = getStateResultName(state, index);
      if (!name) {
        state.emitError()
            << "sequential arc-state mode requires every state result to "
               "have a stable name";
        return failure();
      }
      auto [it, inserted] =
          states.insert({name, NamedState{state, static_cast<unsigned>(index),
                                          result, name}});
      if (!inserted) {
        state.emitError() << "duplicate state name '" << name.getValue() << "'";
        return failure();
      }
      (void)result;
    }
  }
  return states;
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
  if (lhs.result.getType() != rhs.result.getType())
    return lhs.op.emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched types";
  if (lhs.op.getLatency() != rhs.op.getLatency())
    return lhs.op.emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched latencies";
  if (lhs.op.getLatency() != 1)
    return lhs.op.emitError()
           << "state '" << lhs.name.getValue() << "' uses latency "
           << lhs.op.getLatency()
           << "; only latency-1 states are currently supported";
  if (static_cast<bool>(lhs.op.getClock()) !=
      static_cast<bool>(rhs.op.getClock()))
    return lhs.op.emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched clocks";
  if (auto lhsClock = lhs.op.getClock()) {
    auto lhsIndex = getModuleInputIndex(lhsClock);
    auto rhsIndex = getModuleInputIndex(rhs.op.getClock());
    if (failed(lhsIndex) || failed(rhsIndex))
      return lhs.op.emitError()
             << "state '" << lhs.name.getValue()
             << "' must be clocked directly by a module input";
    if (*lhsIndex != *rhsIndex)
      return lhs.op.emitError() << "state '" << lhs.name.getValue()
                                << "' has mismatched clock inputs";
  }
  if (static_cast<bool>(lhs.op.getEnable()) !=
      static_cast<bool>(rhs.op.getEnable()))
    return lhs.op.emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched enables";
  if (static_cast<bool>(lhs.op.getReset()) !=
      static_cast<bool>(rhs.op.getReset()))
    return lhs.op.emitError()
           << "state '" << lhs.name.getValue() << "' has mismatched resets";
  if (lhs.op.getInitials().size() != rhs.op.getInitials().size())
    return lhs.op.emitError() << "state '" << lhs.name.getValue()
                              << "' has mismatched initial values";
  for (auto [leftInit, rightInit] :
       llvm::zip(lhs.op.getInitials(), rhs.op.getInitials())) {
    auto left = getConstInt(leftInit);
    auto right = getConstInt(rightInit);
    if (failed(left) || failed(right) || *left != *right)
      return lhs.op.emitError()
             << "state '" << lhs.name.getValue()
             << "' has unsupported or mismatched initial values";
  }
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
    auto next = materializeArcResult(state.op.getArcAttr(),
                                     state.op.getInputs(), state.resultIndex);
    if (failed(next))
      return failure();
    Value current = stateCutpoints.lookup(state.result);
    assert(current && "state cutpoint must exist");
    if (auto enable = state.op.getEnable()) {
      auto mappedEnable = materialize(enable);
      if (failed(mappedEnable))
        return failure();
      next = comb::MuxOp::create(builder, state.op.getLoc(), current.getType(),
                                 *mappedEnable, *next, current)
                 .getResult();
    }
    if (auto reset = state.op.getReset()) {
      auto mappedReset = materialize(reset);
      if (failed(mappedReset))
        return failure();
      auto intType = dyn_cast<IntegerType>(current.getType());
      if (!intType) {
        state.op.emitError()
            << "reset handling currently requires integer state types";
        return failure();
      }
      Value zero =
          hw::ConstantOp::create(builder, state.op.getLoc(), intType, 0);
      next = comb::MuxOp::create(builder, state.op.getLoc(), current.getType(),
                                 *mappedReset, zero, *next)
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
