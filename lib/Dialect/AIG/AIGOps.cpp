//===- LoopScheduleOps.cpp - LoopSchedule CIRCT Operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the AIG ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace circt;
using namespace circt::aig;

#define GET_OP_CLASSES
#include "circt/Dialect/AIG/AIG.cpp.inc"

void AIGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/AIG/AIG.cpp.inc"
      >();
}

OpFoldResult AndInverterOp::fold(FoldAdaptor adaptor) {
  // if (getLhs() == getRhs()) {
  //   if (hasNoInvertedInputs())
  //     return getLhs();

  //   if (getInvertLhs() != getInvertRhs())
  //     return IntegerAttr::get(
  //         getType(), APInt::getZero(getType().getIntOrFloatBitWidth()));
  // }

  return {};
}

LogicalResult AndInverterOp::canonicalize(AndInverterOp op, PatternRewriter &rewriter) {
  // if (!op.getInvertLhs() && op.getInvertRhs()) {
  //   // Always invert the lhs.
  //   rewriter.replaceOpWithNewOp<AndInverterOp>(op, op.getRhs(), op.getLhs(), true,
  //                                      false);
  //   return success();
  // }
  return failure();
}

mlir::ParseResult AndInverterOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<bool> inverts;
  auto loc = parser.getCurrentLocation();

  while (true) {
    if (succeeded(parser.parseOptionalKeyword("not"))) {
      inverts.push_back(true);
    } else {
      inverts.push_back(false);
    }
    operands.push_back(OpAsmParser::UnresolvedOperand());

    if (parser.parseOperand(operands.back()))
      return failure();
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  mlir::Type resultRawType{};
  llvm::ArrayRef<mlir::Type> resultTypes(&resultRawType, 1);

  {
    mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return mlir::failure();
    resultRawType = type;
  }

  result.addTypes(resultTypes);
  result.addAttribute("inverted",
                      parser.getBuilder().getDenseBoolArrayAttr(inverts));
  if (parser.resolveOperands(operands, resultTypes[0], loc, result.operands))
    return mlir::failure();
  return mlir::success();
}

void AndInverterOp::print(mlir::OpAsmPrinter &odsPrinter) {
  odsPrinter << ' ';
  llvm::interleaveComma(llvm::zip(getInverted(), getInputs()), odsPrinter,
                        [&](auto &&pair) {
                          auto [invert, input] = pair;
                          if (invert) {
                            odsPrinter << "not";
                          }
                          odsPrinter << ' ';
                          odsPrinter << input;
                        });
  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("inverted");
  odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  odsPrinter << ' ' << ":";
  odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = llvm::dyn_cast<mlir::Type>(type))
      odsPrinter.printStrippedAttrOrType(validType);
    else
      odsPrinter << type;
  }
}

APInt AndInverterOp::evaluate(ArrayRef<APInt> inputs) {
  assert(inputs.size() == getNumOperands() &&
         "Expected as many inputs as operands");
  assert(inputs.size() != 0 && "Expected non-empty input list");
  APInt result = APInt::getAllOnes(inputs.front().getBitWidth());
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (isInverted(idx))
      result &= ~input;
    else
      result &= input;
  }
  return result;
}

#include "circt/Dialect/AIG/AIGDialect.cpp.inc"
