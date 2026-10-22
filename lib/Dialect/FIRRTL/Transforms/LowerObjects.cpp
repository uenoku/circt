//===- LowerObjects.cpp - Convert objects to SSA values -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWEROBJECTS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::firrtl;

namespace {
struct LowerObjectsPass
    : public circt::firrtl::impl::LowerObjectsBase<LowerObjectsPass> {
  void runOnOperation() override;

private:
  LogicalResult lowerObject(ObjectOp object);
};
} // namespace

static bool isConstantLike(Value value) {
  auto *def = value.getDefiningOp();
  return def && def->hasTrait<OpTrait::ConstantLike>();
}

/// Collect input assignments before changing the IR. A field may have several
/// subfield operations, so detect multiple drivers by field index rather than
/// by the SSA value of an individual subfield.
LogicalResult LowerObjectsPass::lowerObject(ObjectOp object) {
  auto type = object.getType();
  SmallVector<PropAssignOp> assignments(type.getNumElements());
  SmallVector<ObjectSubfieldOp> inputFields;
  for (auto *user : object->getUsers()) {
    auto field = dyn_cast<ObjectSubfieldOp>(user);
    if (!field || type.getElement(field.getIndex()).direction == Direction::Out)
      continue;
    inputFields.push_back(field);
    for (auto &use : field->getUses()) {
      auto assign = dyn_cast<PropAssignOp>(use.getOwner());
      if (!assign || use.getOperandNumber() != 0)
        return field.emitOpError("input port may only be used as an assignment "
                                 "destination");
      auto &previous = assignments[field.getIndex()];
      if (previous) {
        auto diag = assign.emitError("input port ")
                    << type.getElement(field.getIndex()).name
                    << " cannot be driven by multiple operations";
        diag.attachNote(previous.getLoc()) << "other driver is here";
        return failure();
      }
      previous = assign;
    }
  }

  SmallVector<Value> inputs;
  for (auto [index, element] : llvm::enumerate(type.getElements())) {
    if (element.direction == Direction::Out)
      continue;
    auto assign = assignments[index];
    if (!assign)
      return object.emitError("uninitialized input port ") << element.name;
    inputs.push_back(assign.getSrc());
  }

  // Classes cannot use wires to break object dependencies. Leave their objects
  // with non-constant inputs in declaration form.
  bool constantInputs = llvm::all_of(inputs, isConstantLike);
  if (!constantInputs && !isa<FModuleOp>(object->getParentOp()))
    return success();

  OpBuilder builder(object);
  WireOp wire;
  if (!constantInputs) {
    // Replace the declaration with one wire for the object itself, not its
    // fields. Create the object at block end, where all its inputs are
    // available. Earlier output accesses and cyclic references use the wire
    // instead.
    wire = WireOp::create(builder, object.getLoc(), type,
                          (object.getName() + "_wire").str());
    builder.setInsertionPointToEnd(object->getBlock());
    ++numObjectWiresCreated;
  }
  auto create = ObjectCreateOp::create(builder, object.getLoc(), type,
                                       object.getNameAttr(), inputs);
  // Preserve any discardable attributes attached to the declaration.
  create->setDiscardableAttrs(object->getDiscardableAttrDictionary());
  Value replacement = create.getResult();
  if (wire) {
    PropAssignOp::create(builder, object.getLoc(), wire.getResult(),
                         create.getResult());
    replacement = wire.getResult();
  }
  numInputAssignmentsRemoved +=
      llvm::count_if(assignments, [](PropAssignOp assign) {
        return static_cast<bool>(assign);
      });
  numInputFieldsRemoved += inputFields.size();
  for (auto assign : assignments)
    if (assign)
      assign.erase();
  for (auto field : inputFields)
    field.erase();
  object.replaceAllUsesWith(replacement);
  object.erase();
  ++numObjectsConverted;
  return success();
}

static uint64_t countOperations(Operation *root) {
  uint64_t count = 0;
  root->walk([&](Operation *) { ++count; });
  return count;
}

void LowerObjectsPass::runOnOperation() {
  // Only modules and classes have bodies containing object declarations.
  auto module = getOperation();
  auto *block =
      TypeSwitch<Operation *, Block *>(module)
          .Case<FModuleOp, ClassOp>([](auto op) { return op.getBodyBlock(); })
          .Default([](auto) { return nullptr; });
  if (!block)
    return;

  // Collect first since lowering erases the declarations and some of their
  // users. Do not touch blocks without objects: hoisting their constants would
  // change unrelated operation ordering.
  SmallVector<ObjectOp> objects;
  for (auto object : block->getOps<ObjectOp>())
    objects.push_back(object);
  if (objects.empty())
    return;

  const auto operationsBefore = countOperations(module.getOperation());
  numOperationsBefore += operationsBefore;

  // Make constant inputs available at every object declaration without
  // reordering dependent operations. Preserve the relative order of constants.
  auto insertionPoint = block->begin();
  for (auto &op : llvm::make_early_inc_range(*block)) {
    if (!op.hasTrait<OpTrait::ConstantLike>())
      continue;
    if (op.getIterator() == insertionPoint)
      ++insertionPoint;
    else
      op.moveBefore(block, insertionPoint);
  }

  for (auto object : objects)
    if (failed(lowerObject(object)))
      return signalPassFailure();

  const auto operationsAfter = countOperations(module.getOperation());
  numOperationsAfter += operationsAfter;
  numOperationsRemoved += operationsBefore - operationsAfter;
}
