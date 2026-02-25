//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the export of comb-level IR as a Singular CAS script
// for verifying equivalence of two hw.modules via Gröbner basis reduction.
//
// Given a spec module and an implementation module with matching port
// signatures, this pass encodes both as polynomials and verifies that
// output_impl - output_spec = 0 modulo the gate ideal + Boolean constraints.
//
// Reference: Lv et al., "Scalable and Efficient Verification of Arithmetic
// Circuits using Algebraic Geometry" (2013).
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportSCAProof.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace mlir;

namespace {

static std::string powerOfTwo(unsigned bit) {
  llvm::APInt value(/*numBits=*/bit + 1, 1);
  value = value.shl(bit);
  llvm::SmallString<32> text;
  value.toStringUnsigned(text);
  return std::string(text);
}

/// Maps each SSA Value to a vector of variable names, one per bit.
using BitVarMap = llvm::DenseMap<Value, SmallVector<std::string>>;

/// Encodes a single hw.module's gates as polynomials. Multiple ModuleEncoders
/// can share input variable names (for the shared primary inputs) while using
/// distinct internal gate variable names.
class ModuleEncoder {
public:
  ModuleEncoder(StringRef prefix, unsigned &nextVarId)
      : prefix(prefix), nextVarId(nextVarId) {}

  /// Get the bit width of a value.
  static unsigned getBitWidth(Value v) { return hw::getBitWidth(v.getType()); }

  /// Assign bit-level variables to a value (for inputs).
  void assignBitVars(Value v, ArrayRef<std::string> vars) {
    bitVars[v] = SmallVector<std::string>(vars.begin(), vars.end());
  }

  /// Get bit variables for a value.
  ArrayRef<std::string> getBitVarsFor(Value v) {
    auto it = bitVars.find(v);
    assert(it != bitVars.end() && "Value not yet assigned bit variables");
    return it->second;
  }

  /// Get output bit variables (after processing).
  SmallVector<std::string> getOutputBitVars(hw::HWModuleOp module) {
    auto outputOp = cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    SmallVector<std::string> result;
    for (auto operand : outputOp.getOperands()) {
      auto vars = getBitVarsFor(operand);
      for (auto &v : vars)
        result.push_back(v);
    }
    return result;
  }

  /// Process all operations in the module body.
  LogicalResult processModule(hw::HWModuleOp module) {
    for (auto &op : *module.getBodyBlock()) {
      if (failed(processOp(&op)))
        return failure();
    }
    return success();
  }

  /// Collected data accessible after processing.
  SmallVector<std::string> allVars;
  SmallVector<std::string> gatePolynomials;
  SmallVector<std::string> booleanVars;

private:
  StringRef prefix;
  unsigned &nextVarId;
  BitVarMap bitVars;

  std::string freshVar() {
    return llvm::formatv("{0}{1}", prefix, nextVarId++).str();
  }

  /// Assign fresh bit variables to a value and register them.
  void assignFreshBitVars(Value v) {
    unsigned width = getBitWidth(v);
    auto &vars = bitVars[v];
    vars.reserve(width);
    for (unsigned i = 0; i < width; ++i) {
      std::string var = freshVar();
      vars.push_back(var);
      allVars.push_back(var);
      booleanVars.push_back(var);
    }
  }

  /// Build a word-level polynomial string from bit variables:
  /// bits[0] + 2*bits[1] + 4*bits[2] + ...
  static std::string wordPoly(ArrayRef<std::string> bits) {
    std::string result;
    llvm::raw_string_ostream ss(result);
    bool first = true;
    for (unsigned i = 0; i < bits.size(); ++i) {
      if (bits[i] == "0")
        continue;
      if (!first)
        ss << "+";
      if (i == 0)
        ss << bits[i];
      else
        ss << llvm::formatv("{0}*{1}", powerOfTwo(i), bits[i]);
      first = false;
    }
    if (first)
      ss << "0";
    return result;
  }

  LogicalResult processOp(Operation *op);
};

} // namespace

LogicalResult ModuleEncoder::processOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<comb::AndOp>([&](auto andOp) {
        unsigned width = getBitWidth(andOp.getResult());
        auto &resultVars = bitVars[andOp.getResult()];
        resultVars.reserve(width);
        for (unsigned i = 0; i < width; ++i) {
          std::string gateVar = freshVar();
          resultVars.push_back(gateVar);
          allVars.push_back(gateVar);
          booleanVars.push_back(gateVar);
          std::string expr;
          for (auto [idx, input] : llvm::enumerate(andOp.getInputs())) {
            auto inputVars = getBitVarsFor(input);
            if (idx == 0)
              expr = inputVars[i];
            else
              expr = llvm::formatv("{0}*{1}", expr, inputVars[i]).str();
          }
          gatePolynomials.push_back(
              llvm::formatv("{0}-({1})", gateVar, expr).str());
        }
        return success();
      })
      .Case<comb::XorOp>([&](auto xorOp) {
        unsigned width = getBitWidth(xorOp.getResult());
        auto inputs = xorOp.getInputs();
        SmallVector<std::string> currentVars(getBitVarsFor(inputs[0]).begin(),
                                             getBitVarsFor(inputs[0]).end());
        for (unsigned inputIdx = 1; inputIdx < inputs.size(); ++inputIdx) {
          auto nextVars = getBitVarsFor(inputs[inputIdx]);
          SmallVector<std::string> newVars;
          for (unsigned i = 0; i < width; ++i) {
            std::string gateVar = freshVar();
            newVars.push_back(gateVar);
            allVars.push_back(gateVar);
            booleanVars.push_back(gateVar);
            gatePolynomials.push_back(llvm::formatv("{0}-({1}+{2}-2*{1}*{2})",
                                                    gateVar, currentVars[i],
                                                    nextVars[i])
                                          .str());
          }
          currentVars = std::move(newVars);
        }
        bitVars[xorOp.getResult()] = std::move(currentVars);
        return success();
      })
      .Case<comb::OrOp>([&](auto orOp) {
        unsigned width = getBitWidth(orOp.getResult());
        auto inputs = orOp.getInputs();
        SmallVector<std::string> currentVars(getBitVarsFor(inputs[0]).begin(),
                                             getBitVarsFor(inputs[0]).end());
        for (unsigned inputIdx = 1; inputIdx < inputs.size(); ++inputIdx) {
          auto nextVars = getBitVarsFor(inputs[inputIdx]);
          SmallVector<std::string> newVars;
          for (unsigned i = 0; i < width; ++i) {
            std::string gateVar = freshVar();
            newVars.push_back(gateVar);
            allVars.push_back(gateVar);
            booleanVars.push_back(gateVar);
            gatePolynomials.push_back(llvm::formatv("{0}-({1}+{2}-{1}*{2})",
                                                    gateVar, currentVars[i],
                                                    nextVars[i])
                                          .str());
          }
          currentVars = std::move(newVars);
        }
        bitVars[orOp.getResult()] = std::move(currentVars);
        return success();
      })
      .Case<comb::ExtractOp>([&](auto extractOp) {
        auto inputVars = getBitVarsFor(extractOp.getInput());
        unsigned lowBit = extractOp.getLowBit();
        unsigned width = getBitWidth(extractOp.getResult());
        auto &resultVars = bitVars[extractOp.getResult()];
        resultVars.reserve(width);
        for (unsigned i = 0; i < width; ++i)
          resultVars.push_back(inputVars[lowBit + i]);
        return success();
      })
      .Case<comb::ConcatOp>([&](auto concatOp) {
        auto &resultVars = bitVars[concatOp.getResult()];
        for (int i = concatOp.getNumOperands() - 1; i >= 0; --i) {
          auto inputVars = getBitVarsFor(concatOp.getOperand(i));
          for (const auto &v : inputVars)
            resultVars.push_back(v);
        }
        return success();
      })
      .Case<comb::ReplicateOp>([&](auto repOp) {
        auto inputVars = getBitVarsFor(repOp.getInput());
        unsigned resultWidth = getBitWidth(repOp.getResult());
        unsigned inputWidth = inputVars.size();
        auto &resultVars = bitVars[repOp.getResult()];
        resultVars.reserve(resultWidth);
        for (unsigned i = 0; i < resultWidth; ++i)
          resultVars.push_back(inputVars[i % inputWidth]);
        return success();
      })
      .Case<hw::ConstantOp>([&](auto constOp) {
        APInt val = constOp.getValue();
        unsigned width = val.getBitWidth();
        auto &resultVars = bitVars[constOp.getResult()];
        resultVars.reserve(width);
        for (unsigned i = 0; i < width; ++i)
          resultVars.push_back(val[i] ? "1" : "0");
        return success();
      })
      .Case<comb::AddOp>([&](auto addOp) {
        // Word-level add: sum of output bits == sum of input words (mod 2^w).
        assignFreshBitVars(addOp.getResult());
        std::string outPoly = wordPoly(getBitVarsFor(addOp.getResult()));
        std::string inPoly;
        {
          llvm::raw_string_ostream ss(inPoly);
          for (auto [i, input] : llvm::enumerate(addOp.getInputs())) {
            if (i > 0)
              ss << "+";
            ss << "(" << wordPoly(getBitVarsFor(input)) << ")";
          }
        }
        gatePolynomials.push_back(
            llvm::formatv("({0})-({1})", outPoly, inPoly).str());
        return success();
      })
      .Case<datapath::CompressOp>([&](auto compressOp) {
        // compress: sum of outputs == sum of inputs (word-level).
        // Assign fresh bit vars to each result.
        for (auto result : compressOp.getResults())
          assignFreshBitVars(result);

        // Build constraint: sum_outputs - sum_inputs = 0
        std::string lhs;
        {
          llvm::raw_string_ostream ss(lhs);
          for (auto [i, result] : llvm::enumerate(compressOp.getResults())) {
            if (i > 0)
              ss << "+";
            ss << "(" << wordPoly(getBitVarsFor(result)) << ")";
          }
        }
        std::string rhs;
        {
          llvm::raw_string_ostream ss(rhs);
          for (auto [i, input] : llvm::enumerate(compressOp.getInputs())) {
            if (i > 0)
              ss << "+";
            ss << "(" << wordPoly(getBitVarsFor(input)) << ")";
          }
        }
        gatePolynomials.push_back(llvm::formatv("({0})-({1})", lhs, rhs).str());
        return success();
      })
      .Case<datapath::PartialProductOp>([&](auto ppOp) {
        // partial_product: sum of outputs == lhs * rhs (word-level).
        for (auto result : ppOp.getResults())
          assignFreshBitVars(result);

        std::string lhs;
        {
          llvm::raw_string_ostream ss(lhs);
          for (auto [i, result] : llvm::enumerate(ppOp.getResults())) {
            if (i > 0)
              ss << "+";
            ss << "(" << wordPoly(getBitVarsFor(result)) << ")";
          }
        }
        std::string rhs =
            llvm::formatv("({0})*({1})", wordPoly(getBitVarsFor(ppOp.getLhs())),
                          wordPoly(getBitVarsFor(ppOp.getRhs())));
        gatePolynomials.push_back(llvm::formatv("({0})-({1})", lhs, rhs).str());
        return success();
      })
      .Case<datapath::PosPartialProductOp>([&](auto pppOp) {
        // pos_partial_product: sum of outputs == (addend0 + addend1) *
        // multiplicand
        for (auto result : pppOp.getResults())
          assignFreshBitVars(result);

        std::string lhs;
        {
          llvm::raw_string_ostream ss(lhs);
          for (auto [i, result] : llvm::enumerate(pppOp.getResults())) {
            if (i > 0)
              ss << "+";
            ss << "(" << wordPoly(getBitVarsFor(result)) << ")";
          }
        }
        std::string rhs = llvm::formatv(
            "(({0})+({1}))*({2})", wordPoly(getBitVarsFor(pppOp.getAddend0())),
            wordPoly(getBitVarsFor(pppOp.getAddend1())),
            wordPoly(getBitVarsFor(pppOp.getMultiplicand())));
        gatePolynomials.push_back(llvm::formatv("({0})-({1})", lhs, rhs).str());
        return success();
      })
      .Case<hw::OutputOp>([&](auto) { return success(); })
      .Default([&](Operation *op) {
        return op->emitError("unsupported operation for SCA proof export: ")
               << op->getName();
      });
}

namespace {

class SCAProofExporter {
public:
  SCAProofExporter(hw::HWModuleOp specModule, hw::HWModuleOp implModule,
                   llvm::raw_ostream &os)
      : specModule(specModule), implModule(implModule), os(os) {}

  LogicalResult run();

private:
  hw::HWModuleOp specModule;
  hw::HWModuleOp implModule;
  llvm::raw_ostream &os;

  /// Build shared input variable names from a module's input ports.
  SmallVector<std::pair<StringRef, SmallVector<std::string>>>
  buildInputVarNames(hw::HWModuleOp module);

  /// Assign shared input variables to a module's block arguments.
  void
  assignInputs(ModuleEncoder &encoder, hw::HWModuleOp module,
               const SmallVector<std::pair<StringRef, SmallVector<std::string>>>
                   &inputVars);

  /// Emit a word-level polynomial from bit variables.
  static std::string emitWordPoly(ArrayRef<std::string> bits);
};

} // namespace

SmallVector<std::pair<StringRef, SmallVector<std::string>>>
SCAProofExporter::buildInputVarNames(hw::HWModuleOp module) {
  SmallVector<std::pair<StringRef, SmallVector<std::string>>> result;
  auto moduleType = module.getHWModuleType();
  for (auto [idx, port] : llvm::enumerate(moduleType.getPorts())) {
    if (port.dir != hw::ModulePort::Direction::Input)
      continue;
    unsigned width = hw::getBitWidth(port.type);
    SmallVector<std::string> vars;
    for (unsigned i = 0; i < width; ++i)
      vars.push_back(llvm::formatv("{0}{1}", port.name, i).str());
    result.push_back({port.name, std::move(vars)});
  }
  return result;
}

void SCAProofExporter::assignInputs(
    ModuleEncoder &encoder, hw::HWModuleOp module,
    const SmallVector<std::pair<StringRef, SmallVector<std::string>>>
        &inputVars) {
  unsigned argIdx = 0;
  for (auto &[name, vars] : inputVars) {
    auto blockArg = module.getBody().getArgument(argIdx++);
    encoder.assignBitVars(blockArg, vars);
  }
}

std::string SCAProofExporter::emitWordPoly(ArrayRef<std::string> bits) {
  std::string result;
  llvm::raw_string_ostream ss(result);
  bool first = true;
  for (unsigned i = 0; i < bits.size(); ++i) {
    if (bits[i] == "0")
      continue;
    if (!first)
      ss << "+";
    if (i == 0)
      ss << bits[i];
    else
      ss << llvm::formatv("{0}*{1}", powerOfTwo(i), bits[i]);
    first = false;
  }
  if (first)
    ss << "0";
  return result;
}

LogicalResult SCAProofExporter::run() {
  // Verify port signatures match.
  auto specType = specModule.getHWModuleType();
  auto implType = implModule.getHWModuleType();
  if (specType.getPorts().size() != implType.getPorts().size())
    return implModule.emitError(
        "spec and impl modules must have the same port signature");

  for (auto [specPort, implPort] :
       llvm::zip(specType.getPorts(), implType.getPorts())) {
    if (specPort.dir != implPort.dir || specPort.type != implPort.type)
      return implModule.emitError("port type mismatch between spec and impl");
  }

  // Build shared input variable names from the spec module's ports.
  auto inputVars = buildInputVarNames(specModule);

  // Encode spec module.
  unsigned nextVarId = 0;
  ModuleEncoder specEncoder("s", nextVarId);
  assignInputs(specEncoder, specModule, inputVars);
  if (failed(specEncoder.processModule(specModule)))
    return failure();

  // Encode impl module.
  ModuleEncoder implEncoder("i", nextVarId);
  assignInputs(implEncoder, implModule, inputVars);
  if (failed(implEncoder.processModule(implModule)))
    return failure();

  // Get output bit vars for both.
  auto specOutVars = specEncoder.getOutputBitVars(specModule);
  auto implOutVars = implEncoder.getOutputBitVars(implModule);

  if (specOutVars.size() != implOutVars.size())
    return implModule.emitError("output width mismatch between spec and impl");

  // Collect all boolean vars (inputs + both modules' gates).
  SmallVector<std::string> allBooleanVars;
  for (auto &[name, vars] : inputVars)
    for (auto &v : vars)
      allBooleanVars.push_back(v);
  allBooleanVars.append(specEncoder.booleanVars);
  allBooleanVars.append(implEncoder.booleanVars);

  // Build variable ordering: impl outputs, impl gates (reverse), spec outputs,
  // spec gates (reverse), then shared inputs.
  SmallVector<std::string> orderedVars;

  for (auto &v : implOutVars)
    orderedVars.push_back(v);
  for (auto it = implEncoder.allVars.rbegin(); it != implEncoder.allVars.rend();
       ++it)
    orderedVars.push_back(*it);
  for (auto &v : specOutVars)
    orderedVars.push_back(v);
  for (auto it = specEncoder.allVars.rbegin(); it != specEncoder.allVars.rend();
       ++it)
    orderedVars.push_back(*it);
  for (auto &[name, vars] : inputVars)
    for (auto &v : vars)
      orderedVars.push_back(v);

  // Deduplicate.
  llvm::DenseSet<StringRef> seen;
  SmallVector<std::string> uniqueVars;
  for (auto &v : orderedVars) {
    if (v == "0" || v == "1")
      continue;
    if (seen.insert(StringRef(v)).second)
      uniqueVars.push_back(v);
  }

  // Emit Singular script.
  os << "// SCA equivalence proof generated by CIRCT\n";
  os << "// Spec: " << specModule.getName() << "\n";
  os << "// Impl: " << implModule.getName() << "\n";
  os << "// Verify using Singular (https://www.singular.uni-kl.de/)\n";
  os << "// Run: Singular < this_file.singular\n\n";

  // Ring declaration.
  os << "ring R = 0, (";
  for (auto [i, v] : llvm::enumerate(uniqueVars)) {
    if (i > 0)
      os << ",";
    os << v;
  }
  os << "), lp;\n\n";

  // Gate polynomials from both modules.
  os << "// Gate polynomials (spec + impl)\n";
  os << "ideal J =\n";
  SmallVector<std::string> allPolys;
  allPolys.append(specEncoder.gatePolynomials);
  allPolys.append(implEncoder.gatePolynomials);
  for (auto [i, poly] : llvm::enumerate(allPolys)) {
    os << "  " << poly;
    if (i + 1 < allPolys.size())
      os << ",";
    os << "\n";
  }
  if (allPolys.empty())
    os << "  0\n";
  os << ";\n\n";

  // Boolean constraints.
  os << "// Boolean constraints (x^2 - x = 0)\n";
  os << "ideal B =\n";
  llvm::DenseSet<StringRef> emittedBool;
  bool first = true;
  for (auto &v : allBooleanVars) {
    if (v == "0" || v == "1")
      continue;
    if (!emittedBool.insert(StringRef(v)).second)
      continue;
    if (!first)
      os << ",\n";
    os << "  " << v << "^2-" << v;
    first = false;
  }
  if (first)
    os << "  0";
  os << "\n;\n\n";

  // Spec polynomial: output_impl - output_spec.
  os << "// Specification: impl_output == spec_output\n";
  std::string specPoly = emitWordPoly(specOutVars);
  std::string implPoly = emitWordPoly(implOutVars);
  os << "poly spec = (" << implPoly << ") - (" << specPoly << ");\n\n";

  // Verify.
  os << "ideal I = J + B;\n";
  os << "// Result of 0 means the circuits are equivalent\n";
  os << "reduce(spec, std(I));\n";
  os << "quit;\n";

  return success();
}

LogicalResult circt::exportSCAProof(hw::HWModuleOp specModule,
                                    hw::HWModuleOp implModule,
                                    llvm::raw_ostream &os) {
  SCAProofExporter exporter(specModule, implModule, os);
  return exporter.run();
}

void circt::registerExportSCAProofTranslation() {
  static mlir::TranslateFromMLIRRegistration toSCAProof(
      "export-sca-proof",
      "Export two hw.modules as Singular script for SCA equivalence checking",
      [](mlir::ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        SmallVector<hw::HWModuleOp> hwModules(module.getOps<hw::HWModuleOp>());
        if (hwModules.size() != 2)
          return module.emitError(
                     "expected exactly 2 hw.modules (spec and impl), got ")
                 << hwModules.size();
        // First module is spec, second is impl.
        return exportSCAProof(hwModules[0], hwModules[1], os);
      },
      [](DialectRegistry &registry) {
        registry.insert<hw::HWDialect, comb::CombDialect,
                        datapath::DatapathDialect>();
      });
}
