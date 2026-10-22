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

  // Replace the declaration with one wire for the object itself, not its
  // fields. Create the object at block end, where all its inputs are available.
  // Earlier output accesses and cyclic references use the wire instead.
  OpBuilder builder(object);
  auto wire = WireOp::create(builder, object.getLoc(), type,
                             (object.getName() + "_wire").str());
  builder.setInsertionPointToEnd(object->getBlock());
  ++numObjectWiresCreated;
  auto create = ObjectCreateOp::create(builder, object.getLoc(), type,
                                       object.getNameAttr(), inputs);
  // Preserve any discardable attributes attached to the declaration.
  create->setDiscardableAttrs(object->getDiscardableAttrDictionary());
  PropAssignOp::create(builder, object.getLoc(), wire.getResult(),
                       create.getResult());
  for (auto assign : assignments)
    if (assign) {
      assign.erase();
      ++numInputAssignmentsRemoved;
      ++numRemoved;
    }
  for (auto field : inputFields) {
    field.erase();
    ++numInputFieldsRemoved;
    ++numRemoved;
  }
  object.replaceAllUsesWith(wire.getResult());
  object.erase();
  ++numObjectsConverted;
  ++numRemoved;
  return success();
}

void LowerObjectsPass::runOnOperation() {
  auto result = getOperation()->walk([&](Block *block) {
    // Advance to the next object before erasing this one and its users.
    for (auto object : llvm::make_early_inc_range(block->getOps<ObjectOp>()))
      if (failed(lowerObject(object)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    signalPassFailure();
}
