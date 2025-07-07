//===- LowerComb.cpp - Lower some ops in comb -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_LOWERCOMB
#define GEN_PASS_DEF_CONSTPROP
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {
/// Lower truth tables to mux trees.
struct TruthTableToMuxTree : public OpConversionPattern<TruthTableOp> {
  using OpConversionPattern::OpConversionPattern;

private:
  /// Get a mux tree for `inputs` corresponding to the given truth table. Do
  /// this recursively by dividing the table in half for each input.
  // NOLINTNEXTLINE(misc-no-recursion)
  Value getMux(Location loc, OpBuilder &b, Value t, Value f,
               ArrayRef<bool> table, Operation::operand_range inputs) const {
    assert(table.size() == (1ull << inputs.size()));
    if (table.size() == 1)
      return table.front() ? t : f;

    size_t half = table.size() / 2;
    Value if1 =
        getMux(loc, b, t, f, table.drop_front(half), inputs.drop_front());
    Value if0 =
        getMux(loc, b, t, f, table.drop_back(half), inputs.drop_front());
    return b.create<MuxOp>(loc, inputs.front(), if1, if0, false);
  }

public:
  LogicalResult matchAndRewrite(TruthTableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    SmallVector<bool> table(
        llvm::map_range(op.getLookupTableAttr().getAsValueRange<IntegerAttr>(),
                        [](const APInt &a) { return !a.isZero(); }));
    Value t = b.create<hw::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), 1));
    Value f = b.create<hw::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), 0));

    Value tree = getMux(loc, b, t, f, table, op.getInputs());
    b.modifyOpInPlace(tree.getDefiningOp(), [&]() {
      tree.getDefiningOp()->setDialectAttrs(op->getDialectAttrs());
    });
    b.replaceOp(op, tree);
    return success();
  }
};
} // namespace

namespace {
class LowerCombPass : public impl::LowerCombBase<LowerCombPass> {
public:
  using LowerCombBase::LowerCombBase;

  void runOnOperation() override;
};
} // namespace

void LowerCombPass::runOnOperation() {
  auto module = getOperation();

  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<TruthTableOp>();

  patterns.add<TruthTableToMuxTree>(patterns.getContext());

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();
}

//===- ConstProp.cpp - Inter-module constant propagation ------------------===//
//
// Proprietary and Confidential Software of SiFive Inc. All Rights Reserved.
// See the LICENSE file for license information.
// SPDX-License-Identifier: UNLICENSED
//
//===----------------------------------------------------------------------===//
//
// This file implements the `ConstProp` pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firesim-constprop"

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Constant propagation helper
//===----------------------------------------------------------------------===//

namespace {
class ConstantPropagation {
public:
  ConstantPropagation(InstanceGraph &graph) : graph(graph) {}

  void initialize(HWModuleOp module);
  void propagate();
  std::pair<unsigned, unsigned> fold();

public:
  void enqueue(Value value, IntegerAttr attr);
  void mark(Value value, IntegerAttr attr);
  void propagate(Operation *op);

  /**
   * Returns the lattice value associated with an SSA value.
   *
   * `std::nullopt` is unknown, `IntegerAttr{}` is overdefined.
   */
  std::optional<IntegerAttr> map(Value value);

private:
  InstanceGraph &graph;
  DenseMap<Value, IntegerAttr> values;
  DenseSet<Operation *> inQueue;
  SmallVector<Operation *> overdefQueue;
  SmallVector<Operation *> valueQueue;
};
} // namespace

void ConstantPropagation::initialize(HWModuleOp module) {
  if (module.isPublic()) {
    // Mark public module inputs as overdefined.
    for (auto arg : module.getBodyBlock()->getArguments()) {
      mark(arg, {});
    }
  }

  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<ConstantOp>([&](auto cst) {
          // Constants are omitted from the mapping, but their
          // users are enqueued for propagation.
          enqueue(cst, cst.getValueAttr());
        })
        .Case<HWInstanceLike>([&](auto inst) {
          // Mark external/generated module outputs as overdefined.
          bool hasUnknownTarget = llvm::any_of(
              inst.getReferencedModuleNamesAttr(), [&](Attribute ref) {
                Operation *referencedOp =
                    graph.lookup(cast<StringAttr>(ref))->getModule();
                auto module = dyn_cast_or_null<HWModuleOp>(referencedOp);
                return !module;
              });

          if (hasUnknownTarget) {
            for (auto result : inst->getResults()) {
              mark(result, {});
            }
          }
        })
        .Case<hw::WireOp>([&](auto wire) {
          // Mark wires as overdefined since they can be targeted by force.
          mark(wire.getResult(), {});
        })
        .Default([&](auto op) {
          if (op->getNumResults() == 0)
            return;
          // Mark all non-comb ops and non-integer types as overdefined.
          bool isFoldable = hw::isCombinational(op);
          for (auto result : op->getResults()) {
            Type ty = result.getType();
            if (!hw::type_isa<IntegerType>(ty) || !isFoldable) {
              mark(result, {});
            }
          }
        });
  });
}

void ConstantPropagation::mark(Value value, IntegerAttr attr) {
  auto it = values.try_emplace(value, attr);
  if (!it.second) {
    if (it.first->second == attr)
      return;
    attr = it.first->second = IntegerAttr{};
  }
  enqueue(value, attr);
}

void ConstantPropagation::enqueue(Value value, IntegerAttr attr) {
  for (Operation *user : value.getUsers()) {
    if (inQueue.insert(user).second) {
      if (attr) {
        valueQueue.push_back(user);
      } else {
        overdefQueue.push_back(user);
      }
    }
  }
}

std::optional<IntegerAttr> ConstantPropagation::map(Value value) {
  if (auto constant = value.getDefiningOp<hw::ConstantOp>())
    return constant.getValueAttr();

  auto it = values.find(value);
  if (it == values.end())
    return std::nullopt;

  return it->second;
}

void ConstantPropagation::propagate() {
  while (!overdefQueue.empty() || !valueQueue.empty()) {
    while (!overdefQueue.empty()) {
      auto *op = overdefQueue.pop_back_val();
      inQueue.erase(op);
      propagate(op);
    }
    while (!valueQueue.empty()) {
      auto *op = valueQueue.pop_back_val();
      inQueue.erase(op);
      propagate(op);
    }
  }
}

void ConstantPropagation::propagate(Operation *op) {
  if (auto output = dyn_cast<OutputOp>(op)) {
    auto module = op->getParentOfType<HWModuleOp>();
    for (auto *node : graph[module]->uses()) {
      Operation *instLike = node->getInstance();
      if (!instLike)
        continue;

      auto inst = cast<HWInstanceLike>(instLike);
      for (auto [op, res] :
           llvm::zip(output.getOutputs(), inst->getResults())) {
        if (auto attr = map(op))
          mark(res, *attr);
      }
    }
    return;
  }

  if (auto inst = dyn_cast<HWInstanceLike>(op)) {
    for (auto ref : inst.getReferencedModuleNamesAttr()) {
      Operation *referencedOp =
          graph.lookup(cast<StringAttr>(ref))->getModule();
      auto module = dyn_cast_or_null<HWModuleOp>(referencedOp);
      if (!module)
        continue;

      Block *body = module.getBodyBlock();
      for (auto [op, arg] :
           llvm::zip(inst->getOperands(), body->getArguments())) {
        if (auto attr = map(op))
          mark(arg, *attr);
      }
    }
    return;
  }

  SmallVector<Attribute> operands;
  for (auto op : op->getOperands()) {
    auto attr = map(op);
    if (!attr)
      return;
    operands.push_back(*attr);
  }

  SmallVector<OpFoldResult, 1> results;
  if (succeeded(op->fold(operands, results)) && !results.empty()) {
    for (auto [res, value] : llvm::zip(op->getResults(), results)) {
      if (auto attr = dyn_cast<Attribute>(value)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          mark(res, intAttr);
          continue;
        }
      }
      mark(res, {});
    }
  } else {
    for (auto res : op->getResults()) {
      mark(res, {});
    }
  }
}

std::pair<unsigned, unsigned> ConstantPropagation::fold() {
  // Cache new constants in each module. Traverse the circuit to
  // populate the mapping with values to re-use.
  DenseMap<std::pair<HWModuleOp, IntegerAttr>, Value> constants;
  for (auto *node : graph) {
    Operation *moduleOp = node->getModule();
    if (!moduleOp)
      continue;
    auto module = dyn_cast<HWModuleOp>(moduleOp);
    if (!module)
      continue;
    for (Operation &op : *module.getBodyBlock()) {
      if (auto cst = dyn_cast<ConstantOp>(&op)) {
        constants.try_emplace({module, cst.getValueAttr()}, cst);
      }
    }
  }

  // Traverse the mapping from values to lattices and replace with constants.
  DenseSet<Operation *> toDelete;
  unsigned numFolded = 0;
  for (auto [value, attr] : values) {
    if (!attr)
      continue;

    ImplicitLocOpBuilder builder(value.getLoc(), value.getContext());
    builder.setInsertionPointAfterValue(value);

    HWModuleOp parent;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      parent = cast<HWModuleOp>(arg.getOwner()->getParentOp());
    } else {
      parent = value.getDefiningOp()->getParentOfType<HWModuleOp>();
    }

    auto it = constants.try_emplace({parent, attr}, Value{});
    if (it.second) {
      it.first->second = builder.create<ConstantOp>(value.getType(), attr);
    }

    value.replaceAllUsesWith(it.first->second);
    LLVM_DEBUG({
      llvm::dbgs() << "In " << parent.getModuleName() << ": Replace with "
                   << attr << ": " << value << '\n';
    });

    ++numFolded;

    if (auto *op = value.getDefiningOp()) {
      if (op->use_empty() && mlir::isMemoryEffectFree(op)) {
        toDelete.insert(op);
      }
    }
  }

  for (Operation *op : toDelete)
    op->erase();

  return {numFolded, (unsigned)toDelete.size()};
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct ConstPropPass : public impl::ConstPropBase<ConstPropPass> {
  void runOnOperation() override;
};
} // namespace

void ConstPropPass::runOnOperation() {
  ConstantPropagation prop(getAnalysis<InstanceGraph>());

  for (auto module : getOperation().getOps<HWModuleOp>())
    prop.initialize(module);

  prop.propagate();

  auto [numFolded, numErased] = prop.fold();
  markAnalysesPreserved<InstanceGraph>();
}

/// namespace comb {
/// std::unique_ptr<Pass> createConstPropPass() {
///   return std::make_unique<ConstPropPass>();
/// }
//} // namespace firesim