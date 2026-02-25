//===- VerifyNativeSCA.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <optional>

#define DEBUG_TYPE "verify-native-sca"

using namespace circt;
using namespace mlir;

namespace circt {
#define GEN_PASS_DEF_VERIFYNATIVESCA
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

namespace {

using Monomial = llvm::BitVector;

static APInt makeSignedConst(int64_t value) {
  return APInt(64, static_cast<uint64_t>(value), true);
}

static APInt normalizeSigned(APInt value) {
  unsigned width = std::max(2u, value.getSignificantBits() + 1);
  return value.sextOrTrunc(width);
}

static APInt addSigned(const APInt &a, const APInt &b) {
  unsigned width = std::max(a.getBitWidth(), b.getBitWidth()) + 1;
  return normalizeSigned(a.sext(width) + b.sext(width));
}

static APInt mulSigned(const APInt &a, const APInt &b) {
  unsigned width = a.getBitWidth() + b.getBitWidth();
  return normalizeSigned(a.sext(width) * b.sext(width));
}

static APInt negSigned(const APInt &a) {
  return normalizeSigned((-a.sext(a.getBitWidth() + 1)));
}

static bool isDivisibleSigned(const APInt &num, const APInt &den) {
  if (den.isZero())
    return false;
  unsigned width = std::max(num.getBitWidth(), den.getBitWidth()) + 1;
  return num.sext(width).srem(den.sext(width)).isZero();
}

static APInt exactDivSigned(const APInt &num, const APInt &den) {
  unsigned width = std::max(num.getBitWidth(), den.getBitWidth()) + 1;
  return normalizeSigned(num.sext(width).sdiv(den.sext(width)));
}

static bool monomialDivides(const Monomial &a, const Monomial &b) {
  for (int bit = a.find_first(); bit >= 0; bit = a.find_next(bit))
    if (!b.test(bit))
      return false;
  return true;
}

static Monomial monomialMul(const Monomial &a, const Monomial &b) {
  Monomial result = a;
  result |= b;
  return result;
}

static Monomial monomialDiv(const Monomial &num, const Monomial &den) {
  Monomial result = num;
  for (int bit = den.find_first(); bit >= 0; bit = den.find_next(bit))
    result.reset(bit);
  return result;
}

static int compareMonomial(const Monomial &a, const Monomial &b) {
  assert(a.size() == b.size() && "mismatched monomial widths");
  for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i) {
    bool ai = a.test(i);
    bool bi = b.test(i);
    if (ai != bi)
      return ai ? 1 : -1;
  }
  if (a.count() != b.count())
    return a.count() > b.count() ? 1 : -1;
  return 0;
}

struct Polynomial {
  explicit Polynomial(unsigned numVars) : numVars(numVars) {}

  static Polynomial getZero(unsigned numVars) { return Polynomial(numVars); }

  static Polynomial getConstant(unsigned numVars, int64_t c) {
    Polynomial poly(numVars);
    if (c != 0)
      poly.addTerm(Monomial(numVars, false), makeSignedConst(c));
    return poly;
  }

  static Polynomial getVariable(unsigned numVars, unsigned var) {
    Polynomial poly(numVars);
    Monomial mono(numVars, false);
    mono.set(var);
    poly.addTerm(mono, makeSignedConst(1));
    return poly;
  }

  void addTerm(const Monomial &mono, const APInt &coeff) {
    if (coeff.isZero())
      return;
    auto it = terms.find(mono);
    if (it == terms.end()) {
      terms.try_emplace(mono, normalizeSigned(coeff));
      return;
    }
    APInt combined = addSigned(it->second, coeff);
    if (combined.isZero())
      terms.erase(it);
    else
      it->second = combined;
  }

  void addScaled(const Polynomial &other, const APInt &coeff,
                 const Monomial &mono) {
    if (coeff.isZero())
      return;
    for (const auto &kv : other.terms) {
      Monomial newMono = monomialMul(kv.first, mono);
      APInt newCoeff = mulSigned(kv.second, coeff);
      addTerm(newMono, newCoeff);
    }
  }

  void add(const Polynomial &other) {
    for (const auto &kv : other.terms)
      addTerm(kv.first, kv.second);
  }

  void sub(const Polynomial &other) {
    for (const auto &kv : other.terms)
      addTerm(kv.first, negSigned(kv.second));
  }

  Polynomial mul(const Polynomial &other) const {
    Polynomial result(numVars);
    for (const auto &lhs : terms)
      for (const auto &rhs : other.terms)
        result.addTerm(monomialMul(lhs.first, rhs.first),
                       mulSigned(lhs.second, rhs.second));
    return result;
  }

  bool isZero() const { return terms.empty(); }
  size_t size() const { return terms.size(); }

  unsigned maxDegree() const {
    unsigned degree = 0;
    for (const auto &kv : terms)
      degree = std::max<unsigned>(degree, kv.first.count());
    return degree;
  }

  std::optional<std::pair<Monomial, APInt>> getLeadingTerm() const {
    if (terms.empty())
      return std::nullopt;
    auto it = terms.begin();
    Monomial best = it->first;
    APInt coeff = it->second;
    ++it;
    for (; it != terms.end(); ++it) {
      if (compareMonomial(it->first, best) > 0) {
        best = it->first;
        coeff = it->second;
      }
    }
    return std::make_pair(best, coeff);
  }

  unsigned numVars;
  llvm::DenseMap<Monomial, APInt> terms;
};

struct BitExpr {
  enum Kind { Const0, Const1, Var } kind;
  unsigned var = 0;

  static BitExpr getConst0() { return {Const0, 0}; }
  static BitExpr getConst1() { return {Const1, 0}; }
  static BitExpr getVar(unsigned v) { return {Var, v}; }
};

static Polynomial bitToPoly(const BitExpr &bit, unsigned numVars) {
  switch (bit.kind) {
  case BitExpr::Const0:
    return Polynomial::getZero(numVars);
  case BitExpr::Const1:
    return Polynomial::getConstant(numVars, 1);
  case BitExpr::Var:
    return Polynomial::getVariable(numVars, bit.var);
  }
  llvm_unreachable("unknown bit kind");
}

static APInt powerOfTwoCoeff(unsigned bit) {
  return APInt::getOneBitSet(bit + 2, bit);
}

static Polynomial wordToPoly(ArrayRef<BitExpr> bits, unsigned numVars) {
  Polynomial result(numVars);
  for (unsigned i = 0; i < bits.size(); ++i) {
    auto bitPoly = bitToPoly(bits[i], numVars);
    result.addScaled(bitPoly, powerOfTwoCoeff(i), Monomial(numVars, false));
  }
  return result;
}

struct Reducer {
  Polynomial poly;
  Monomial leadMono;
  APInt leadCoeff;
  bool valid = false;

  explicit Reducer(unsigned numVars)
      : poly(Polynomial::getZero(numVars)), leadMono(numVars, false),
        leadCoeff(makeSignedConst(0)) {}
};

struct RewriteBudget {
  uint64_t maxTerms;
  uint32_t maxDegree;
  uint64_t maxSteps;
};

struct RewriteStats {
  uint64_t steps = 0;
  bool exceeded = false;
};

static bool updateLeading(Reducer &reducer) {
  auto lead = reducer.poly.getLeadingTerm();
  if (!lead)
    return false;
  reducer.leadMono = lead->first;
  reducer.leadCoeff = lead->second;
  reducer.valid = !reducer.leadCoeff.isZero() && reducer.leadMono.any();
  return reducer.valid;
}

static bool reduceOnce(Polynomial &target, const Reducer &reducer,
                       RewriteStats &stats) {
  if (!reducer.valid)
    return false;

  std::optional<std::pair<Monomial, APInt>> best;
  for (const auto &kv : target.terms) {
    if (!monomialDivides(reducer.leadMono, kv.first))
      continue;
    if (!isDivisibleSigned(kv.second, reducer.leadCoeff))
      continue;
    if (!best || compareMonomial(kv.first, best->first) > 0)
      best = std::make_pair(kv.first, kv.second);
  }
  if (!best)
    return false;

  const Monomial &targetMono = best->first;
  const APInt &targetCoeff = best->second;
  APInt factorCoeff = exactDivSigned(targetCoeff, reducer.leadCoeff);
  Monomial factorMono = monomialDiv(targetMono, reducer.leadMono);
  LLVM_DEBUG(llvm::dbgs() << "[native-sca] reduce step " << stats.steps
                          << " terms=" << target.size() << '\n');
  target.addScaled(reducer.poly, negSigned(factorCoeff), factorMono);
  ++stats.steps;
  return true;
}

class ModuleEncoder {
public:
  ModuleEncoder(StringRef prefix, unsigned &nextVarId,
                SmallVectorImpl<std::string> &varNames,
                SmallVectorImpl<Reducer> &reducers, unsigned numVars)
      : prefix(prefix), nextVarId(nextVarId), varNames(varNames),
        reducers(reducers), numVars(numVars) {}

  static unsigned getBitWidth(Value value) {
    return hw::getBitWidth(value.getType());
  }

  void assignBits(Value value, ArrayRef<BitExpr> bits) {
    bitMap[value] = SmallVector<BitExpr>(bits.begin(), bits.end());
  }

  ArrayRef<BitExpr> getBits(Value value) const {
    auto it = bitMap.find(value);
    assert(it != bitMap.end() && "value is not encoded yet");
    return it->second;
  }

  LogicalResult processModule(hw::HWModuleOp module) {
    scanProbes(module);
    for (Operation &op : *module.getBodyBlock()) {
      if (failed(processOp(&op)))
        return failure();
    }
    emitProbeReducers();
    for (auto &reducer : guidedReducers)
      reducers.push_back(std::move(reducer));
    return success();
  }

  SmallVector<BitExpr> getOutputBits(hw::HWModuleOp module) const {
    auto outputOp = cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    SmallVector<BitExpr> result;
    for (Value value : outputOp.getOperands()) {
      auto bits = getBits(value);
      result.append(bits.begin(), bits.end());
    }
    return result;
  }

private:
  struct ProbeGroup {
    std::map<unsigned, SmallVector<BitExpr>> inputs;
    std::map<unsigned, SmallVector<BitExpr>> outputs;
  };

  StringRef prefix;
  unsigned &nextVarId;
  SmallVectorImpl<std::string> &varNames;
  SmallVectorImpl<Reducer> &reducers;
  unsigned numVars;
  llvm::DenseMap<Value, SmallVector<BitExpr>> bitMap;
  llvm::DenseMap<int64_t, ProbeGroup> compressProbeGroups;
  llvm::DenseMap<Value, int64_t> compressOutputOrigin;
  SmallVector<Reducer> guidedReducers;

  unsigned freshVar() {
    unsigned id = nextVarId++;
    if (id >= varNames.size())
      varNames.resize(id + 1);
    varNames[id] = llvm::formatv("{0}{1}", prefix, id).str();
    return id;
  }

  SmallVector<BitExpr> assignFreshBits(Value value) {
    unsigned width = getBitWidth(value);
    SmallVector<BitExpr> bits;
    bits.reserve(width);
    for (unsigned i = 0; i < width; ++i)
      bits.push_back(BitExpr::getVar(freshVar()));
    bitMap[value] = bits;
    return bits;
  }

  void addReducer(Polynomial poly) {
    addReducer(std::move(poly), /*guided=*/false);
  }

  void addReducer(Polynomial poly, bool guided) {
    Reducer reducer(numVars);
    reducer.poly = std::move(poly);
    updateLeading(reducer);
    if (guided)
      guidedReducers.push_back(std::move(reducer));
    else
      reducers.push_back(std::move(reducer));
  }

  void scanProbes(hw::HWModuleOp module) {
    for (Operation &op : *module.getBodyBlock()) {
      auto probe = dyn_cast<verif::PolyProbeOp>(op);
      if (!probe)
        continue;
      if (probe.getOrigin() != "compress" || probe.getRole() != "output")
        continue;
      auto originIDAttr = probe.getOriginIdAttr();
      if (!originIDAttr)
        continue;
      compressOutputOrigin[probe.getValue()] = originIDAttr.getInt();
    }
  }

  Polynomial xorPoly(const Polynomial &lhs, const Polynomial &rhs) const {
    Polynomial result = lhs;
    result.add(rhs);
    Polynomial prod = lhs.mul(rhs);
    result.addScaled(prod, makeSignedConst(-2), Monomial(numVars, false));
    return result;
  }

  Polynomial majorityPoly(const Polynomial &a, const Polynomial &b,
                          const Polynomial &c) const {
    Polynomial result = a.mul(b);
    result.add(a.mul(c));
    result.add(b.mul(c));
    Polynomial abc = a.mul(b).mul(c);
    result.addScaled(abc, makeSignedConst(-2), Monomial(numVars, false));
    return result;
  }

  BitExpr bindGateToPoly(Polynomial rhs) {
    unsigned gate = freshVar();
    Polynomial lhs = Polynomial::getVariable(numVars, gate);
    lhs.sub(rhs);
    addReducer(std::move(lhs));
    return BitExpr::getVar(gate);
  }

  BitExpr emitXor(BitExpr a, BitExpr b) {
    return bindGateToPoly(
        xorPoly(bitToPoly(a, numVars), bitToPoly(b, numVars)));
  }

  BitExpr emitMajority(BitExpr a, BitExpr b, BitExpr c) {
    return bindGateToPoly(majorityPoly(
        bitToPoly(a, numVars), bitToPoly(b, numVars), bitToPoly(c, numVars)));
  }

  LogicalResult lowerCompress(datapath::CompressOp compressOp) {
    unsigned width = getBitWidth(compressOp.getResults().front());
    unsigned targetRows = compressOp.getNumResults();
    if (targetRows < 2)
      return compressOp.emitOpError("requires at least 2 results");

    SmallVector<SmallVector<BitExpr>> columns(width);
    for (Value input : compressOp.getInputs()) {
      auto bits = getBits(input);
      for (unsigned i = 0; i < width; ++i)
        columns[i].push_back(bits[i]);
    }

    for (unsigned bit = 0; bit < width; ++bit) {
      while (columns[bit].size() > targetRows) {
        if (columns[bit].size() < 3)
          break;
        BitExpr a = columns[bit].pop_back_val();
        BitExpr b = columns[bit].pop_back_val();
        BitExpr c = columns[bit].pop_back_val();
        BitExpr sum = emitXor(emitXor(a, b), c);
        BitExpr carry = emitMajority(a, b, c);
        columns[bit].push_back(sum);
        if (bit + 1 < width)
          columns[bit + 1].push_back(carry);
      }
    }

    for (unsigned row = 0; row < targetRows; ++row) {
      SmallVector<BitExpr> rowBits;
      rowBits.reserve(width);
      for (unsigned bit = 0; bit < width; ++bit) {
        if (row < columns[bit].size())
          rowBits.push_back(columns[bit][row]);
        else
          rowBits.push_back(BitExpr::getConst0());
      }
      bitMap[compressOp.getResult(row)] = std::move(rowBits);
    }
    return success();
  }

  Polynomial orPoly(const Polynomial &lhs, const Polynomial &rhs) const {
    Polynomial result = lhs;
    result.add(rhs);
    Polynomial prod = lhs.mul(rhs);
    result.sub(prod);
    return result;
  }

  LogicalResult processOp(Operation *op) {
    if (!isa<verif::PolyProbeOp, hw::OutputOp>(op) &&
        !op->getResults().empty()) {
      bool allCompressProbeOutputs = true;
      for (Value result : op->getResults()) {
        if (!compressOutputOrigin.count(result)) {
          allCompressProbeOutputs = false;
          break;
        }
      }
      if (allCompressProbeOutputs) {
        for (Value result : op->getResults())
          assignFreshBits(result);
        LLVM_DEBUG(llvm::dbgs() << "[native-sca] abstracted hinted op "
                                << op->getName() << '\n');
        return success();
      }
    }

    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<comb::AndOp>([&](comb::AndOp andOp) {
          unsigned width = getBitWidth(andOp.getResult());
          SmallVector<BitExpr> resultBits;
          resultBits.reserve(width);
          for (unsigned i = 0; i < width; ++i) {
            unsigned gate = freshVar();
            resultBits.push_back(BitExpr::getVar(gate));
            Polynomial rhs = Polynomial::getConstant(numVars, 1);
            for (Value input : andOp.getInputs())
              rhs = rhs.mul(bitToPoly(getBits(input)[i], numVars));
            Polynomial lhs = Polynomial::getVariable(numVars, gate);
            lhs.sub(rhs);
            addReducer(std::move(lhs));
          }
          bitMap[andOp.getResult()] = std::move(resultBits);
          return success();
        })
        .Case<comb::XorOp>([&](comb::XorOp xorOp) {
          unsigned width = getBitWidth(xorOp.getResult());
          auto inputs = xorOp.getInputs();
          SmallVector<BitExpr> current(getBits(inputs[0]).begin(),
                                       getBits(inputs[0]).end());
          for (unsigned idx = 1; idx < inputs.size(); ++idx) {
            auto next = getBits(inputs[idx]);
            SmallVector<BitExpr> fresh;
            fresh.reserve(width);
            for (unsigned i = 0; i < width; ++i) {
              unsigned gate = freshVar();
              fresh.push_back(BitExpr::getVar(gate));
              auto lhs = Polynomial::getVariable(numVars, gate);
              auto rhs = xorPoly(bitToPoly(current[i], numVars),
                                 bitToPoly(next[i], numVars));
              lhs.sub(rhs);
              addReducer(std::move(lhs));
            }
            current = std::move(fresh);
          }
          bitMap[xorOp.getResult()] = std::move(current);
          return success();
        })
        .Case<comb::OrOp>([&](comb::OrOp orOp) {
          unsigned width = getBitWidth(orOp.getResult());
          auto inputs = orOp.getInputs();
          SmallVector<BitExpr> current(getBits(inputs[0]).begin(),
                                       getBits(inputs[0]).end());
          for (unsigned idx = 1; idx < inputs.size(); ++idx) {
            auto next = getBits(inputs[idx]);
            SmallVector<BitExpr> fresh;
            fresh.reserve(width);
            for (unsigned i = 0; i < width; ++i) {
              unsigned gate = freshVar();
              fresh.push_back(BitExpr::getVar(gate));
              auto lhs = Polynomial::getVariable(numVars, gate);
              auto rhs = orPoly(bitToPoly(current[i], numVars),
                                bitToPoly(next[i], numVars));
              lhs.sub(rhs);
              addReducer(std::move(lhs));
            }
            current = std::move(fresh);
          }
          bitMap[orOp.getResult()] = std::move(current);
          return success();
        })
        .Case<comb::ExtractOp>([&](comb::ExtractOp extractOp) {
          auto inputBits = getBits(extractOp.getInput());
          unsigned lowBit = extractOp.getLowBit();
          unsigned width = getBitWidth(extractOp.getResult());
          SmallVector<BitExpr> result;
          result.reserve(width);
          for (unsigned i = 0; i < width; ++i)
            result.push_back(inputBits[lowBit + i]);
          bitMap[extractOp.getResult()] = std::move(result);
          return success();
        })
        .Case<comb::ConcatOp>([&](comb::ConcatOp concatOp) {
          SmallVector<BitExpr> result;
          for (int i = concatOp.getNumOperands() - 1; i >= 0; --i) {
            auto bits = getBits(concatOp.getOperand(i));
            result.append(bits.begin(), bits.end());
          }
          bitMap[concatOp.getResult()] = std::move(result);
          return success();
        })
        .Case<comb::ReplicateOp>([&](comb::ReplicateOp replicateOp) {
          auto inputBits = getBits(replicateOp.getInput());
          unsigned inputWidth = inputBits.size();
          unsigned resultWidth = getBitWidth(replicateOp.getResult());
          SmallVector<BitExpr> result;
          result.reserve(resultWidth);
          for (unsigned i = 0; i < resultWidth; ++i)
            result.push_back(inputBits[i % inputWidth]);
          bitMap[replicateOp.getResult()] = std::move(result);
          return success();
        })
        .Case<hw::ConstantOp>([&](hw::ConstantOp constantOp) {
          APInt value = constantOp.getValue();
          SmallVector<BitExpr> bits;
          bits.reserve(value.getBitWidth());
          for (unsigned i = 0; i < value.getBitWidth(); ++i)
            bits.push_back(value[i] ? BitExpr::getConst1()
                                    : BitExpr::getConst0());
          bitMap[constantOp.getResult()] = std::move(bits);
          return success();
        })
        .Case<comb::AddOp>([&](comb::AddOp addOp) {
          auto inputs = addOp.getInputs();
          if (inputs.empty()) {
            addOp.emitOpError("requires at least one operand");
            return failure();
          }

          bool guidedByProbe = !inputs.empty();
          std::optional<int64_t> originID;
          for (Value input : inputs) {
            auto it = compressOutputOrigin.find(input);
            if (it == compressOutputOrigin.end()) {
              guidedByProbe = false;
              break;
            }
            if (!originID)
              originID = it->second;
            else if (*originID != it->second) {
              guidedByProbe = false;
              break;
            }
          }

          if (guidedByProbe) {
            assignFreshBits(addOp.getResult());
            unsigned width = getBitWidth(addOp.getResult());
            Polynomial lhs = wordToPoly(getBits(addOp.getResult()), numVars);

            unsigned carryBits =
                inputs.size() > 1 ? llvm::Log2_64_Ceil(inputs.size()) : 0;
            for (unsigned i = 0; i < carryBits; ++i) {
              unsigned carryVar = freshVar();
              lhs.addScaled(Polynomial::getVariable(numVars, carryVar),
                            powerOfTwoCoeff(width + i),
                            Monomial(numVars, false));
            }

            Polynomial rhs = Polynomial::getZero(numVars);
            for (Value input : inputs)
              rhs.add(wordToPoly(getBits(input), numVars));

            lhs.sub(rhs);
            addReducer(std::move(lhs), /*guided=*/true);
            LLVM_DEBUG(llvm::dbgs() << "[native-sca] guided add reducer for "
                                    << addOp.getResult() << '\n');
            return success();
          }

          unsigned width = getBitWidth(addOp.getResult());
          SmallVector<BitExpr> current(getBits(inputs[0]).begin(),
                                       getBits(inputs[0]).end());

          for (unsigned inputIdx = 1; inputIdx < inputs.size(); ++inputIdx) {
            auto nextBits = getBits(inputs[inputIdx]);
            SmallVector<BitExpr> sums;
            sums.reserve(width);
            BitExpr carry = BitExpr::getConst0();

            for (unsigned i = 0; i < width; ++i) {
              BitExpr sum = emitXor(emitXor(current[i], nextBits[i]), carry);
              BitExpr nextCarry = emitMajority(current[i], nextBits[i], carry);
              sums.push_back(sum);
              carry = nextCarry;
            }
            current = std::move(sums);
          }

          bitMap[addOp.getResult()] = std::move(current);
          return success();
        })
        .Case<comb::MulOp>([&](comb::MulOp mulOp) {
          assignFreshBits(mulOp.getResult());
          Polynomial lhs = wordToPoly(getBits(mulOp.getResult()), numVars);
          auto inputs = mulOp.getInputs();
          Polynomial rhs = wordToPoly(getBits(inputs[0]), numVars)
                               .mul(wordToPoly(getBits(inputs[1]), numVars));
          lhs.sub(rhs);
          addReducer(std::move(lhs));
          return success();
        })
        .Case<datapath::CompressOp>([&](datapath::CompressOp compressOp) {
          for (Value result : compressOp.getResults())
            assignFreshBits(result);
          Polynomial lhs = Polynomial::getZero(numVars);
          for (Value result : compressOp.getResults())
            lhs.add(wordToPoly(getBits(result), numVars));
          Polynomial rhs = Polynomial::getZero(numVars);
          for (Value input : compressOp.getInputs())
            rhs.add(wordToPoly(getBits(input), numVars));
          lhs.sub(rhs);
          addReducer(std::move(lhs), /*guided=*/true);
          return success();
        })
        .Case<datapath::PartialProductOp>(
            [&](datapath::PartialProductOp partialProductOp) {
              for (Value result : partialProductOp.getResults())
                assignFreshBits(result);
              Polynomial lhs = Polynomial::getZero(numVars);
              for (Value result : partialProductOp.getResults())
                lhs.add(wordToPoly(getBits(result), numVars));
              Polynomial rhs =
                  wordToPoly(getBits(partialProductOp.getLhs()), numVars)
                      .mul(wordToPoly(getBits(partialProductOp.getRhs()),
                                      numVars));
              lhs.sub(rhs);
              addReducer(std::move(lhs));
              return success();
            })
        .Case<datapath::PosPartialProductOp>(
            [&](datapath::PosPartialProductOp op) {
              for (Value result : op.getResults())
                assignFreshBits(result);
              Polynomial lhs = Polynomial::getZero(numVars);
              for (Value result : op.getResults())
                lhs.add(wordToPoly(getBits(result), numVars));

              Polynomial addends =
                  wordToPoly(getBits(op.getAddend0()), numVars);
              addends.add(wordToPoly(getBits(op.getAddend1()), numVars));
              Polynomial rhs = addends.mul(
                  wordToPoly(getBits(op.getMultiplicand()), numVars));
              lhs.sub(rhs);
              addReducer(std::move(lhs));
              return success();
            })
        .Case<hw::OutputOp>([](hw::OutputOp) { return success(); })
        .Case<verif::PolyProbeOp>([&](verif::PolyProbeOp probeOp) {
          if (probeOp.getOrigin() != "compress")
            return success();

          auto originIDAttr = probeOp.getOriginIdAttr();
          if (!originIDAttr)
            return success();

          auto value = probeOp.getValue();
          auto valueIt = bitMap.find(value);
          if (valueIt == bitMap.end()) {
            probeOp.emitOpError("probed value is not encoded yet");
            return failure();
          }

          auto &group = compressProbeGroups[originIDAttr.getInt()];
          if (probeOp.getRole() == "input") {
            if (auto inputIdx = probeOp.getInputIdxAttr())
              group.inputs[inputIdx.getInt()] = valueIt->second;
            return success();
          }
          if (probeOp.getRole() == "output") {
            if (auto resultIdx = probeOp.getResultIdxAttr())
              group.outputs[resultIdx.getInt()] = valueIt->second;
            return success();
          }
          return success();
        })
        .Default([&](Operation *unsupported) {
          return unsupported->emitError(
                     "unsupported operation for native SCA verification: ")
                 << unsupported->getName();
        });
  }

  void emitProbeReducers() {
    for (auto &kv : compressProbeGroups) {
      auto &group = kv.second;
      if (group.inputs.empty() || group.outputs.empty())
        continue;

      Polynomial lhs = Polynomial::getZero(numVars);
      for (auto &out : group.outputs)
        lhs.add(wordToPoly(out.second, numVars));

      Polynomial rhs = Polynomial::getZero(numVars);
      for (auto &in : group.inputs)
        rhs.add(wordToPoly(in.second, numVars));

      lhs.sub(rhs);
      addReducer(std::move(lhs), /*guided=*/true);
      LLVM_DEBUG(llvm::dbgs() << "[native-sca] added compress probe reducer id="
                              << kv.first << '\n');
    }
  }
};

static void appendInputBits(hw::HWModuleOp module, unsigned &nextVarId,
                            SmallVectorImpl<std::string> &varNames,
                            SmallVectorImpl<SmallVector<BitExpr>> &portBits) {
  for (auto port : module.getHWModuleType().getPorts()) {
    if (port.dir != hw::ModulePort::Direction::Input)
      continue;
    unsigned width = hw::getBitWidth(port.type);
    SmallVector<BitExpr> bits;
    bits.reserve(width);
    for (unsigned i = 0; i < width; ++i) {
      unsigned id = nextVarId++;
      if (id >= varNames.size())
        varNames.resize(id + 1);
      varNames[id] = llvm::formatv("{0}{1}", port.name, i).str();
      bits.push_back(BitExpr::getVar(id));
    }
    portBits.push_back(std::move(bits));
  }
}

static LogicalResult assignModuleInputs(hw::HWModuleOp module,
                                        ModuleEncoder &enc,
                                        ArrayRef<SmallVector<BitExpr>> inputs) {
  unsigned argIndex = 0;
  unsigned inputIndex = 0;
  for (auto port : module.getHWModuleType().getPorts()) {
    if (port.dir != hw::ModulePort::Direction::Input)
      continue;
    if (inputIndex >= inputs.size())
      return module.emitOpError("internal input assignment mismatch");
    enc.assignBits(module.getBody().getArgument(argIndex), inputs[inputIndex]);
    ++inputIndex;
    ++argIndex;
  }
  return success();
}

struct VerifyNativeSCAPass
    : public circt::impl::VerifyNativeSCABase<VerifyNativeSCAPass> {
  using circt::impl::VerifyNativeSCABase<
      VerifyNativeSCAPass>::VerifyNativeSCABase;

  void runOnOperation() override {
    auto module = getOperation();
    if (firstModule.empty() || secondModule.empty()) {
      module.emitError("both --first-module and --second-module are required");
      return signalPassFailure();
    }

    auto first = module.lookupSymbol<hw::HWModuleOp>(firstModule);
    if (!first) {
      module.emitError("hw.module named '") << firstModule << "' not found";
      return signalPassFailure();
    }
    auto second = module.lookupSymbol<hw::HWModuleOp>(secondModule);
    if (!second) {
      module.emitError("hw.module named '") << secondModule << "' not found";
      return signalPassFailure();
    }

    if (first.getModuleType() != second.getModuleType()) {
      first.emitError("module IO types do not match second module: ")
          << first.getModuleType() << " vs " << second.getModuleType();
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[native-sca] compare " << firstModule << " vs "
                            << secondModule << '\n');

    unsigned numVars = 1;
    unsigned nextVarId = 0;
    SmallVector<std::string> varNames;
    varNames.resize(1);
    SmallVector<SmallVector<BitExpr>> inputBits;
    appendInputBits(first, nextVarId, varNames, inputBits);

    unsigned estimatedVarCount = nextVarId;
    auto countFreshVars = [&](hw::HWModuleOp hwModule) {
      for (Operation &op : *hwModule.getBodyBlock()) {
        llvm::TypeSwitch<Operation *>(&op)
            .Case<comb::AndOp>([&](comb::AndOp andOp) {
              estimatedVarCount += hw::getBitWidth(andOp.getResult().getType());
            })
            .Case<comb::XorOp>([&](comb::XorOp xorOp) {
              unsigned width = hw::getBitWidth(xorOp.getResult().getType());
              estimatedVarCount += width * (xorOp.getInputs().size() - 1);
            })
            .Case<comb::OrOp>([&](comb::OrOp orOp) {
              unsigned width = hw::getBitWidth(orOp.getResult().getType());
              estimatedVarCount += width * (orOp.getInputs().size() - 1);
            })
            .Case<comb::AddOp>([&](comb::AddOp addOp) {
              unsigned width = hw::getBitWidth(addOp.getResult().getType());
              estimatedVarCount +=
                  2 * width *
                  (std::max<size_t>(1, addOp.getInputs().size()) - 1);
            })
            .Case<comb::MulOp>([&](comb::MulOp mulOp) {
              estimatedVarCount += hw::getBitWidth(mulOp.getResult().getType());
            })
            .Case<datapath::CompressOp>([&](datapath::CompressOp compressOp) {
              unsigned width =
                  hw::getBitWidth(compressOp.getResults().front().getType());
              estimatedVarCount +=
                  4 * width *
                  std::max<size_t>(1, compressOp.getInputs().size());
            })
            .Case<datapath::PartialProductOp>(
                [&](datapath::PartialProductOp partialProductOp) {
                  for (Value result : partialProductOp.getResults())
                    estimatedVarCount += hw::getBitWidth(result.getType());
                })
            .Case<datapath::PosPartialProductOp>(
                [&](datapath::PosPartialProductOp posPartialProductOp) {
                  for (Value result : posPartialProductOp.getResults())
                    estimatedVarCount += hw::getBitWidth(result.getType());
                })
            .Default([](Operation *) {});
      }
    };
    countFreshVars(first);
    countFreshVars(second);
    if (estimatedVarCount == 0)
      estimatedVarCount = 1;
    numVars = estimatedVarCount;

    varNames.resize(numVars);
    nextVarId = 0;
    inputBits.clear();
    appendInputBits(first, nextVarId, varNames, inputBits);

    SmallVector<Reducer> reducers;
    ModuleEncoder firstEncoder("s", nextVarId, varNames, reducers, numVars);
    if (failed(assignModuleInputs(first, firstEncoder, inputBits)) ||
        failed(firstEncoder.processModule(first))) {
      signalPassFailure();
      return;
    }

    ModuleEncoder secondEncoder("i", nextVarId, varNames, reducers, numVars);
    if (failed(assignModuleInputs(second, secondEncoder, inputBits)) ||
        failed(secondEncoder.processModule(second))) {
      signalPassFailure();
      return;
    }

    auto firstOut = firstEncoder.getOutputBits(first);
    auto secondOut = secondEncoder.getOutputBits(second);
    if (firstOut.size() != secondOut.size()) {
      second.emitError("output width mismatch between compared modules");
      return signalPassFailure();
    }

    Polynomial residual = wordToPoly(secondOut, numVars);
    residual.sub(wordToPoly(firstOut, numVars));
    LLVM_DEBUG(llvm::dbgs() << "[native-sca] reducers=" << reducers.size()
                            << " residual-terms=" << residual.size() << '\n');

    RewriteBudget budget{maxTerms, maxDegree, maxSteps};
    RewriteStats stats;

    bool changed = true;
    while (changed) {
      changed = false;
      for (auto it = reducers.rbegin(); it != reducers.rend(); ++it) {
        while (reduceOnce(residual, *it, stats)) {
          changed = true;
          if (residual.size() > budget.maxTerms ||
              residual.maxDegree() > budget.maxDegree ||
              stats.steps > budget.maxSteps) {
            stats.exceeded = true;
            break;
          }
        }
        if (stats.exceeded)
          break;
      }
      if (stats.exceeded)
        break;
    }

    if (stats.exceeded) {
      llvm::errs() << "c1 ? c2\n";
      auto diag = second.emitOpError(
          "inconclusive: native SCA rewrite budget exceeded");
      diag << " (terms=" << residual.size()
           << ", max-degree=" << residual.maxDegree()
           << ", steps=" << stats.steps << ")";
      return signalPassFailure();
    }

    if (!residual.isZero()) {
      llvm::errs() << "c1 != c2\n";
      LLVM_DEBUG(llvm::dbgs()
                 << "[native-sca] final residual terms=" << residual.size()
                 << " degree=" << residual.maxDegree()
                 << " steps=" << stats.steps << '\n');
      auto diag =
          second.emitOpError("native SCA check failed: non-zero residual");
      diag << " (terms=" << residual.size()
           << ", max-degree=" << residual.maxDegree()
           << ", steps=" << stats.steps << ")";
      return signalPassFailure();
    }

    llvm::outs() << "c1 == c2\n";
  }
};

} // namespace
