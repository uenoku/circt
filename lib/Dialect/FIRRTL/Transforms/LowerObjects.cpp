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
#include "llvm/ADT/SetVector.h"

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
};
} // namespace

/// Collect input assignments before changing the IR. A field may have several
/// subfield operations, so detect multiple drivers by field index rather than
/// by the SSA value of an individual subfield.
static LogicalResult lowerObject(ObjectOp object, bool &converted,
                                 size_t &numAssignmentsRemoved,
                                 size_t &numInputFieldsRemoved) {
  converted = false;
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

  // Leave objects with non-constant inputs in declaration form. Their input
  // assignments may introduce dependencies that cannot be ordered in SSA.
  if (!llvm::all_of(inputs, [](Value input) {
        auto *def = input.getDefiningOp();
        return def && def->hasTrait<OpTrait::ConstantLike>();
      }))
    return success();

  OpBuilder builder(object);
  auto create = ObjectCreateOp::create(builder, object.getLoc(), type,
                                       object.getNameAttr(), inputs);
  // Preserve any discardable attributes attached to the declaration.
  create->setDiscardableAttrs(object->getDiscardableAttrDictionary());
  numAssignmentsRemoved += llvm::count_if(assignments, [](PropAssignOp assign) {
    return static_cast<bool>(assign);
  });
  numInputFieldsRemoved += inputFields.size();
  for (auto assign : assignments)
    if (assign)
      assign.erase();
  for (auto field : inputFields)
    field.erase();
  object.replaceAllUsesWith(create.getResult());
  object.erase();
  converted = true;
  return success();
}

static uint64_t countOperations(Operation *root) {
  uint64_t count = 0;
  root->walk([&](Operation *) { ++count; });
  return count;
}

void LowerObjectsPass::runOnOperation() {
  // This pass is scheduled under nestAny(), so it runs on each module and
  // class independently. Ignore other operations nested directly in the
  // circuit, such as external module declarations.
  auto *container = getOperation();
  if (!isa<FModuleOp, ClassOp>(container) || container->getRegion(0).empty())
    return;
  auto *block = &container->getRegion(0).front();

  // Collect first since lowering erases the declarations and some of their
  // users. Do not touch blocks without objects: hoisting their constants would
  // change unrelated operation ordering.
  SmallVector<ObjectOp> objects;
  for (auto object : block->getOps<ObjectOp>())
    objects.push_back(object);
  if (objects.empty())
    return;

  const auto operationsBefore = countOperations(container);
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

  for (auto object : objects) {
    bool converted;
    size_t assignmentsRemoved = 0, inputFieldsRemoved = 0;
    if (failed(lowerObject(object, converted, assignmentsRemoved,
                           inputFieldsRemoved)))
      return signalPassFailure();
    if (!converted)
      continue;
    ++numObjectsConverted;
    numInputAssignmentsRemoved += assignmentsRemoved;
    numInputFieldsRemoved += inputFieldsRemoved;
  }

  const auto operationsAfter = countOperations(container);
  numOperationsAfter += operationsAfter;
  numOperationsRemoved += operationsBefore - operationsAfter;
}
