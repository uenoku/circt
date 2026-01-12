//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "circt/Support/NPNClass.h"
#include "circt/Support/Naming.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace circt;
using namespace circt::synth::mig;
using namespace circt::synth::aig;

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.cpp.inc"

LogicalResult MajorityInverterOp::verify() {
  if (getNumOperands() % 2 != 1)
    return emitOpError("requires an odd number of operands");

  return success();
}

llvm::APInt MajorityInverterOp::evaluate(ArrayRef<APInt> inputs) {
  assert(inputs.size() == getNumOperands() &&
         "Number of inputs must match number of operands");

  if (inputs.size() == 3) {
    auto a = (isInverted(0) ? ~inputs[0] : inputs[0]);
    auto b = (isInverted(1) ? ~inputs[1] : inputs[1]);
    auto c = (isInverted(2) ? ~inputs[2] : inputs[2]);
    return (a & b) | (a & c) | (b & c);
  }

  // General case for odd number of inputs != 3
  auto width = inputs[0].getBitWidth();
  APInt result(width, 0);

  for (size_t bit = 0; bit < width; ++bit) {
    size_t count = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      // Count the number of 1s, considering inversion.
      if (isInverted(i) ^ inputs[i][bit])
        count++;
    }

    if (count > inputs.size() / 2)
      result.setBit(bit);
  }

  return result;
}

OpFoldResult MajorityInverterOp::fold(FoldAdaptor adaptor) {
  // TODO: Implement maj(x, 1, 1) = 1, maj(x, 0, 0) = 0

  SmallVector<APInt, 3> inputValues;
  for (auto input : adaptor.getInputs()) {
    auto attr = llvm::dyn_cast_or_null<IntegerAttr>(input);
    if (!attr)
      return {};
    inputValues.push_back(attr.getValue());
  }

  auto result = evaluate(inputValues);
  return IntegerAttr::get(getType(), result);
}

LogicalResult MajorityInverterOp::canonicalize(MajorityInverterOp op,
                                               PatternRewriter &rewriter) {
  if (op.getNumOperands() == 1) {
    if (op.getInverted()[0])
      return failure();
    rewriter.replaceOp(op, op.getOperand(0));
    return success();
  }

  // For now, only support 3 operands.
  if (op.getNumOperands() != 3)
    return failure();

  // Return if the idx-th operand is a constant (inverted if necessary),
  // otherwise return std::nullopt.
  auto getConstant = [&](unsigned index) -> std::optional<llvm::APInt> {
    APInt value;
    if (mlir::matchPattern(op.getInputs()[index], mlir::m_ConstantInt(&value)))
      return op.isInverted(index) ? ~value : value;
    return std::nullopt;
  };

  // Replace the op with the idx-th operand (inverted if necessary).
  auto replaceWithIndex = [&](int index) {
    bool inverted = op.isInverted(index);
    if (inverted)
      rewriter.replaceOpWithNewOp<MajorityInverterOp>(
          op, op.getType(), op.getOperand(index), true);
    else
      rewriter.replaceOp(op, op.getOperand(index));
    return success();
  };

  // Pattern match following cases:
  // maj_inv(x, x, y) -> x
  // maj_inv(x, y, not y) -> x
  for (int i = 0; i < 2; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      int k = 3 - (i + j);
      assert(k >= 0 && k < 3);
      // If we have two identical operands, we can fold.
      if (op.getOperand(i) == op.getOperand(j)) {
        // If they are inverted differently, we can fold to the third.
        if (op.isInverted(i) != op.isInverted(j))
          return replaceWithIndex(k);
        return replaceWithIndex(i);
      }

      // If i and j are constant.
      if (auto c1 = getConstant(i)) {
        if (auto c2 = getConstant(j)) {
          // If both constants are equal, we can fold.
          if (*c1 == *c2) {
            rewriter.replaceOpWithNewOp<hw::ConstantOp>(
                op, op.getType(), mlir::IntegerAttr::get(op.getType(), *c1));
            return success();
          }
          // If constants are complementary, we can fold.
          if (*c1 == ~*c2)
            return replaceWithIndex(k);
        }
      }
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AIG Operations
//===----------------------------------------------------------------------===//

OpFoldResult AndInverterOp::fold(FoldAdaptor adaptor) {
  if (getNumOperands() == 1 && !isInverted(0))
    return getOperand(0);

  auto inputs = adaptor.getInputs();
  if (inputs.size() == 2)
    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(inputs[1])) {
      auto value = intAttr.getValue();
      if (isInverted(1))
        value = ~value;
      if (value.isZero())
        return IntegerAttr::get(
            IntegerType::get(getContext(), value.getBitWidth()), value);
      if (value.isAllOnes()) {
        if (isInverted(0))
          return {};

        return getOperand(0);
      }
    }
  return {};
}

LogicalResult AndInverterOp::canonicalize(AndInverterOp op,
                                          PatternRewriter &rewriter) {
  SmallDenseMap<Value, bool> seen;
  SmallVector<Value> uniqueValues;
  SmallVector<bool> uniqueInverts;

  APInt constValue =
      APInt::getAllOnes(op.getResult().getType().getIntOrFloatBitWidth());

  bool invertedConstFound = false;
  bool flippedFound = false;

  for (auto [value, inverted] : llvm::zip(op.getInputs(), op.getInverted())) {
    bool newInverted = inverted;
    if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
      if (inverted) {
        constValue &= ~constOp.getValue();
        invertedConstFound = true;
      } else {
        constValue &= constOp.getValue();
      }
      continue;
    }

    if (auto andInverterOp = value.getDefiningOp<synth::aig::AndInverterOp>()) {
      if (andInverterOp.getInputs().size() == 1 &&
          andInverterOp.isInverted(0)) {
        value = andInverterOp.getOperand(0);
        newInverted = andInverterOp.isInverted(0) ^ inverted;
        flippedFound = true;
      }
    }

    auto it = seen.find(value);
    if (it == seen.end()) {
      seen.insert({value, newInverted});
      uniqueValues.push_back(value);
      uniqueInverts.push_back(newInverted);
    } else if (it->second != newInverted) {
      // replace with const 0
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(
          op, APInt::getZero(value.getType().getIntOrFloatBitWidth()));
      return success();
    }
  }

  // If the constant is zero, we can just replace with zero.
  if (constValue.isZero()) {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, constValue);
    return success();
  }

  // No change.
  if ((uniqueValues.size() == op.getInputs().size() && !flippedFound) ||
      (!constValue.isAllOnes() && !invertedConstFound &&
       uniqueValues.size() + 1 == op.getInputs().size()))
    return failure();

  if (!constValue.isAllOnes()) {
    auto constOp = hw::ConstantOp::create(rewriter, op.getLoc(), constValue);
    uniqueInverts.push_back(false);
    uniqueValues.push_back(constOp);
  }

  // It means the input is reduced to all ones.
  if (uniqueValues.empty()) {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, constValue);
    return success();
  }

  // build new op with reduced input values
  replaceOpWithNewOpAndCopyNamehint<synth::aig::AndInverterOp>(
      rewriter, op, uniqueValues, uniqueInverts);
  return success();
}

APInt AndInverterOp::evaluate(ArrayRef<APInt> inputs) {
  assert(inputs.size() == getNumOperands() &&
         "Expected as many inputs as operands");
  assert(!inputs.empty() && "Expected non-empty input list");
  APInt result = APInt::getAllOnes(inputs.front().getBitWidth());
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (isInverted(idx))
      result &= ~input;
    else
      result &= input;
  }
  return result;
}

static Value lowerVariadicAndInverterOp(AndInverterOp op, OperandRange operands,
                                        ArrayRef<bool> inverts,
                                        PatternRewriter &rewriter) {
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    if (inverts[0])
      return AndInverterOp::create(rewriter, op.getLoc(), operands[0], true);
    else
      return operands[0];
  case 2:
    return AndInverterOp::create(rewriter, op.getLoc(), operands[0],
                                 operands[1], inverts[0], inverts[1]);
  default:
    auto firstHalf = operands.size() / 2;
    auto lhs =
        lowerVariadicAndInverterOp(op, operands.take_front(firstHalf),
                                   inverts.take_front(firstHalf), rewriter);
    auto rhs =
        lowerVariadicAndInverterOp(op, operands.drop_front(firstHalf),
                                   inverts.drop_front(firstHalf), rewriter);
    return AndInverterOp::create(rewriter, op.getLoc(), lhs, rhs);
  }
  return Value();
}

LogicalResult circt::synth::AndInverterVariadicOpConversion::matchAndRewrite(
    AndInverterOp op, PatternRewriter &rewriter) const {
  if (op.getInputs().size() <= 2)
    return failure();
  // TODO: This is a naive implementation that creates a balanced binary tree.
  //       We can improve by analyzing the dataflow and creating a tree that
  //       improves the critical path or area.
  rewriter.replaceOp(op, lowerVariadicAndInverterOp(
                             op, op.getOperands(), op.getInverted(), rewriter));
  return success();
}

LogicalResult circt::synth::topologicallySortGraphRegionBlocks(
    mlir::Operation *op,
    llvm::function_ref<bool(mlir::Value, mlir::Operation *)> isOperandReady) {
  // Sort the operations topologically
  auto walkResult = op->walk([&](Region *region) {
    auto regionKindOp =
        dyn_cast<mlir::RegionKindInterface>(region->getParentOp());
    if (!regionKindOp ||
        regionKindOp.hasSSADominance(region->getRegionNumber()))
      return WalkResult::advance();

    // Graph region.
    for (auto &block : *region) {
      if (!mlir::sortTopologically(&block, isOperandReady))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// ISOP (Irredundant Sum-of-Products) Implementation
//===----------------------------------------------------------------------===//

using llvm::APInt;
using llvm::SmallVector;

/// Precomputed masks for variables in truth tables up to 6 variables (64 bits).
/// Masks[var][0] = mask where var=0 (negative literal)
/// Masks[var][1] = mask where var=1 (positive literal)
static constexpr uint64_t kVarMasks[6][2] = {
    {0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL}, // var 0: alternating bits
    {0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL}, // var 1: pairs of bits
    {0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL}, // var 2: nibbles
    {0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL}, // var 3: bytes
    {0x0000FFFF0000FFFFULL, 0xFFFF0000FFFF0000ULL}, // var 4: half-words
    {0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL}, // var 5: words
};

/// Create a mask for a variable in the truth table.
/// For positive=true: mask has 1s where var=1 in the truth table encoding
/// For positive=false: mask has 1s where var=0 in the truth table encoding
static APInt createVarMask(unsigned numVars, unsigned var, bool positive) {
  uint32_t numBits = 1u << numVars;

  // Use precomputed table for small cases (up to 6 variables = 64 bits)
  if (numVars <= 6) {
    assert(var < 6);
    uint64_t maskValue = kVarMasks[var][positive ? 1 : 0];
    // Mask off bits beyond numBits
    if (numBits < 64)
      maskValue &= (1ULL << numBits) - 1;
    return APInt(numBits, maskValue);
  }

  // For larger cases, build mask by setting bits in blocks
  APInt mask(numBits, 0);
  uint32_t shift = 1u << var;

  for (uint32_t i = 0; i < numBits; i += 2 * shift) {
    if (positive) {
      // Set upper half of each block
      for (uint32_t j = 0; j < shift && (i + shift + j) < numBits; ++j)
        mask.setBit(i + shift + j);
    } else {
      // Set lower half of each block
      for (uint32_t j = 0; j < shift && (i + j) < numBits; ++j)
        mask.setBit(i + j);
    }
  }

  return mask;
}

/// Compute cofactor of a Boolean function.
static std::pair<APInt, APInt>
computeCofactors(const APInt &f, unsigned numVars, unsigned var) {
  uint32_t numBits = 1u << numVars;
  uint32_t shift = 1u << var;

  // Create mask that selects bits for each cofactor
  APInt blockMask = APInt::getLowBitsSet(numBits, shift);

  // Build masks for both cofactors in one pass
  APInt mask0(numBits, 0); // Selects bits where var=0
  APInt mask1(numBits, 0); // Selects bits where var=1

  for (uint32_t i = 0; i < numBits; i += 2 * shift) {
    mask0 |= blockMask.shl(i);         // Lower half of each block
    mask1 |= blockMask.shl(i + shift); // Upper half of each block
  }

  // Extract bits for each cofactor
  APInt selected0 = f & mask0;
  APInt selected1 = f & mask1;

  // Duplicate to fill entire truth table
  APInt cof0 = selected0 | selected0.shl(shift);  // Copy lower to upper
  APInt cof1 = selected1 | selected1.lshr(shift); // Copy upper to lower

  return {cof0, cof1};
}

/// Check if a variable is in the support of the function.
static bool variableInSupport(const APInt &f, unsigned numVars, unsigned var) {
  auto [f0, f1] = computeCofactors(f, numVars, var);
  return f0 != f1;
}

/// Minato-Morreale ISOP algorithm.
static APInt isopRec(const APInt &tt, const APInt &dc, unsigned numVars,
                     unsigned varIndex, synth::SOPForm &result) {
  // Invariant: tt must be a subset of dc (all ON-set bits are in care set)
  assert((tt & ~dc).isZero() && "tt must be subset of dc");

  // Base case: nothing to cover
  if (tt.isZero())
    return tt;

  // Base case: all don't-cares, add empty cube
  if (dc.isAllOnes()) {
    result.cubes.emplace_back(numVars);
    return dc;
  }

  assert(varIndex > 0 && "No more variables to process");

  // Find the highest variable that actually appears in tt or dc
  int var = varIndex - 1;
  for (; var >= 0; --var)
    if (variableInSupport(tt, numVars, var) ||
        variableInSupport(dc, numVars, var))
      break;

  // If no variable found, add empty cube if needed
  assert(var >= 0 && "No variable found in tt or dc");

  // Compute cofactors with respect to the splitting variable
  auto [negativeCofactor, positiveCofactor] =
      computeCofactors(tt, numVars, var);
  auto [negativeDC, positiveDC] = computeCofactors(dc, numVars, var);

  // Recurse on minterms unique to negative cofactor (will get !var literal)
  size_t negativeBegin = result.cubes.size();
  APInt negativeCover =
      isopRec(negativeCofactor & ~positiveDC, negativeDC, numVars, var, result);
  size_t negativeEnd = result.cubes.size();

  // Recurse on minterms unique to positive cofactor (will get var literal)
  APInt positiveCover =
      isopRec(positiveCofactor & ~negativeDC, positiveDC, numVars, var, result);
  size_t positiveEnd = result.cubes.size();

  // Recurse on shared minterms (will get no literal for this variable)
  APInt sharedCover = isopRec((negativeCofactor & ~negativeCover) |
                                  (positiveCofactor & ~positiveCover),
                              negativeDC & positiveDC, numVars, var, result);

  // Create masks for the variable to restrict covers to their domains
  APInt negativeMask =
      createVarMask(numVars, var, false); // Minterms where var=0
  APInt positiveMask =
      createVarMask(numVars, var, true); // Minterms where var=1

  // Combine results: restrict each cover to its domain
  APInt totalCover = sharedCover | (negativeCover & negativeMask) |
                     (positiveCover & positiveMask);

  // Add negative literal to cubes from first recursion
  APInt mask(numVars, 1 << var);
  for (size_t i = negativeBegin; i < negativeEnd; ++i) {
    result.cubes[i].mask |= mask;
    result.cubes[i].inverted |= mask;
  }

  // Add positive literal to cubes from second recursion
  for (size_t i = negativeEnd; i < positiveEnd; ++i) {
    result.cubes[i].mask |= mask;
    // inverted bit remains 0 for positive literal var
  }

  // Verify invariants
  assert((tt & ~totalCover).isZero() && "result must cover tt");
  assert((totalCover & ~dc).isZero() && "result must be subset of dc");

  return totalCover;
}

void synth::SOPForm::dump(llvm::raw_ostream &os) const {
  os << "SOPForm: " << numVars << " vars, " << cubes.size() << " cubes\n";
  for (const auto &cube : cubes) {
    os << "  (";
    for (unsigned i = 0; i < numVars; ++i) {
      if (cube.mask[i]) {
        os << (cube.inverted[i] ? "!" : "");
        os << "x" << i << " ";
      }
    }
    os << ")\n";
  }
}

APInt synth::SOPForm::computeTruthTable() const {
  APInt tt(1 << numVars, 0);
  for (const auto &cube : cubes) {
    APInt cubeTT = ~APInt(1 << numVars, 0);
    for (unsigned i = 0; i < numVars; ++i) {
      if (cube.mask[i]) {
        cubeTT &= createVarMask(numVars, i, !cube.inverted[i]);
      }
    }
    tt |= cubeTT;
  }
  return tt;
}

bool synth::SOPForm::isIrredundant() {
  APInt tt = computeTruthTable();
  for (auto &cube : cubes) {
    auto temporary = cube;
    // Remove one literal from the cube
    for (unsigned i = 0; i < numVars; ++i) {
      if (temporary.mask[i]) {
        cube.mask.setBitVal(i, 0);
        cube.inverted.setBitVal(i, 0);
        if (tt == computeTruthTable())
          return false;
        cube = temporary;
      }
    }
  }

  return true;
}

synth::SOPForm
synth::extractSOPFromTruthTable(const circt::BinaryTruthTable &tt) {
  synth::SOPForm sop(tt.numInputs);

  if (tt.numInputs == 0 || tt.table.isZero())
    return sop;

  // Call the ISOP algorithm
  // dc = tt means all ON-set bits are also don't-cares (no OFF-set constraints)
  (void)isopRec(tt.table, tt.table, tt.numInputs, tt.numInputs, sop);

// Verify the result is correct
#ifdef DEBUG
  APInt result = sop.computeTruthTable();
  (void)result;
  assert(result == tt.table && "ISOP does not match original truth table!");
#endif

  return sop;
}
