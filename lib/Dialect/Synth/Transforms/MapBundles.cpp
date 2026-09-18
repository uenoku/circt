//===- MapBundles.cpp - Map bundles to multi-output cells --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Consumes `synth.create_bundle` ops: for each bundle, derives the joint
// function of its operands over their shared support by fresh cone
// simulation (sound: every walked value is live, so nothing read can
// dangle), and matches it against multi-output library cells under
// input/output permutation (positive phases only; the discrete
// phase-aware flow covers the rest). On a match the bundle becomes one
// cell instance (cheapest area wins); otherwise it is erased and the
// discrete logic maps as before. Nothing downstream ever observes a bundle.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthOpInterfaces.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_MAPBUNDLES
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-map-bundles"

namespace {

// A multi-output library cell candidate.
struct MultiOutputCell {
  hw::HWModuleOp module;
  StringAttr name;
  // Per-output truth tables over input-port order.
  SmallVector<uint64_t, 2> tables;
  unsigned numInputs = 0;
  double area = std::numeric_limits<double>::infinity();
};

struct MapBundlesPass
    : public circt::synth::impl::MapBundlesBase<MapBundlesPass> {
  using MapBundlesBase<MapBundlesPass>::MapBundlesBase;

  /// True when the target holds no state: no clock-marked input pins and
  /// no stateful ops in its body. (Nested instance bodies are the
  /// callee's own responsibility at their discovery; v1 scope.)
  static bool isCombinationalCell(hw::HWModuleOp module) {
    for (const auto &port : module.getPortList()) {
      if (!port.isInput() || !port.attrs)
        continue;
      if (auto pin = dyn_cast_or_null<DictionaryAttr>(
              port.attrs.get("synth.liberty.pin")))
        if (auto clock = dyn_cast_or_null<StringAttr>(pin.get("clock")))
          if (clock.getValue() == "true")
            return false;
    }
    bool hasState = false;
    if (auto *body = module.getBodyBlock()) {
      body->walk([&](Operation *op) {
        if (isa<synth::FFOp, synth::LatchOp, synth::ClockGateOp, seq::FirRegOp,
                seq::CompRegOp, seq::CompRegClockEnabledOp, seq::FirMemOp>(op))
          hasState = true;
      });
    }
    return !hasState;
  }

  void runOnOperation() override {
    ModuleOp top = getOperation();
    SmallVector<MultiOutputCell> cells = collectCells(top);
    bool changed = false;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isa<hw::HWModuleExternOp>(module) ||
          module->hasAttr("synth.liberty.cell") ||
          module->hasAttr("synth.dont_use") ||
          module->hasAttr("synth.mapping_cost"))
        continue;
      changed |= mapInModule(module, cells);
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

  /// Combinational cells with exactly two all-i1 outputs and a small
  /// all-i1 input count. Approval is a liberty marker or a mapping_cost
  /// attribute (DeriveMappingCost only prices single-output cells, so
  /// mapping_cost alone would exclude every multi-output cell), minus
  /// explicit dont_use. Area (mapping_cost preferred) drives
  /// cheapest-wins selection.
  SmallVector<MultiOutputCell> collectCells(ModuleOp top) {
    SmallVector<MultiOutputCell> cells;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isa<hw::HWModuleExternOp>(module))
        continue;
      if (module->hasAttr("synth.dont_use"))
        continue;
      bool approved = module->hasAttr("synth.liberty.cell") ||
                      module->hasAttr("synth.mapping_cost");
      if (!approved)
        continue;
      double area = std::numeric_limits<double>::infinity();
      if (auto cost =
              module->getAttrOfType<MappingCostAttr>("synth.mapping_cost"))
        area = cost.getArea().getValue().convertToDouble();
      else if (auto cellDict =
                   module->getAttrOfType<DictionaryAttr>("synth.liberty.cell"))
        if (auto areaAttr = dyn_cast_or_null<FloatAttr>(cellDict.get("area")))
          area = areaAttr.getValue().convertToDouble();
      unsigned numInputs = 0, numOutputs = 0;
      bool allSingleBit = true;
      for (const auto &port : module.getPortList()) {
        if (port.isInput()) {
          ++numInputs;
          allSingleBit &= port.type.isInteger(1);
        } else if (port.isOutput()) {
          ++numOutputs;
          allSingleBit &= port.type.isInteger(1);
        } else {
          allSingleBit = false;
        }
      }
      if (!allSingleBit || numOutputs != 2 || numInputs < 2 || numInputs > 3)
        continue;
      // Combinational only: no stateful bodies, no clock pins (flops have
      // two outputs and would otherwise match buffer/inverter pairs).
      if (!isCombinationalCell(module))
        continue;
      auto *body = module.getBodyBlock();
      if (!body)
        continue;
      auto *terminator = body->getTerminator();
      auto outputOp = dyn_cast_or_null<hw::OutputOp>(terminator);
      if (!outputOp || outputOp->getNumOperands() != 2)
        continue;
      MultiOutputCell cell;
      cell.module = module;
      cell.name = module.getModuleNameAttr();
      cell.numInputs = numInputs;
      bool ok = true;
      for (Value output : outputOp->getOperands()) {
        auto table = getTruthTable(ValueRange{output}, body);
        if (failed(table) || table->numInputs != (int64_t)numInputs ||
            table->numOutputs != 1 || table->table.getBitWidth() > 64) {
          ok = false;
          break;
        }
        cell.tables.push_back(table->table.getZExtValue());
      }
      if (!ok || cell.tables.size() != 2)
        continue;
      cell.area = area;
      cells.push_back(std::move(cell));
    }
    numCandidates += cells.size();
    // Deterministic selection order (cheapest first, name tiebreak).
    llvm::sort(cells, [](const MultiOutputCell &a, const MultiOutputCell &b) {
      if (a.area != b.area)
        return a.area < b.area;
      return a.name.getValue() < b.name.getValue();
    });
    return cells;
  }

  bool mapInModule(hw::HWModuleOp module, ArrayRef<MultiOutputCell> cells) {
    // Collect bundles first; rewriting mutates the block.
    SmallVector<synth::CreateBundleOp> bundles;
    for (auto &block : module->getRegion(0))
      for (auto &op : block)
        if (auto bundle = dyn_cast<synth::CreateBundleOp>(op))
          bundles.push_back(bundle);
    if (bundles.empty())
      return false;
    bool changed = false;
    for (auto bundle : bundles) {
      if (mapOneBundle(bundle, cells))
        changed = true;
    }
    return changed;
  }

  /// Joint tables of the bundle operands over canonical (sorted) shared
  /// support, derived by fresh cone simulation. Soundness: every walked
  /// value is live (roots are used by the bundle being processed; every
  /// step follows operands of live ops), so nothing read can dangle.
  /// Returns nullopt when the cones leave the simulatable domain.
  struct JointTables {
    SmallVector<Value, 4> leaves;
    uint64_t tables[2];
  };

  // Cone-walking state (fresh per bundle).
  struct ConeWalker {
    static constexpr unsigned kMaxSupport = 8;
    static constexpr unsigned kMaxNodes = 8192;
    SmallVector<Value> support;
    DenseSet<Value> supportSet;
    DenseMap<Value, APInt> constVals;
    SmallVector<Operation *> postorder;
    DenseSet<Value> doneValues;
    DenseSet<Operation *> activeOps;
    bool ok = true;

    void addLeaf(Value v) {
      if (supportSet.insert(v).second) {
        support.push_back(v);
        if (support.size() > kMaxSupport)
          ok = false;
      }
    }

    void walk(Value v) {
      if (!ok || doneValues.contains(v))
        return;
      if (!v.getType().isInteger(1)) {
        ok = false;
        return;
      }
      Operation *defOp = v.getDefiningOp();
      if (!defOp) {
        // Block argument: opaque input.
        addLeaf(v);
        doneValues.insert(v);
        return;
      }
      if (auto constant = dyn_cast<hw::ConstantOp>(defOp)) {
        constVals[v] = constant.getValue().isZero() ? APInt::getZero(1)
                                                    : APInt::getAllOnes(1);
        doneValues.insert(v);
        return;
      }
      // Nested bundles, state, and unknown ops end the walk (conservative).
      if (isa<synth::CreateBundleOp>(defOp)) {
        ok = false;
        return;
      }
      if (isa<hw::InstanceOp>(defOp)) {
        // Cell/flop outputs are stable values: opaque inputs.
        addLeaf(v);
        doneValues.insert(v);
        return;
      }
      // Single-bit projections of wider values are likewise opaque.
      if (isa<comb::ExtractOp>(defOp)) {
        addLeaf(v);
        doneValues.insert(v);
        return;
      }
      bool simulatable =
          isa<synth::ChoiceOp, comb::AndOp, comb::OrOp, comb::XorOp>(defOp) ||
          isa<BooleanLogicOpInterface>(defOp);
      if (!simulatable || defOp->getNumResults() != 1) {
        ok = false;
        return;
      }
      if (!activeOps.insert(defOp).second) {
        ok = false; // Combinational cycle.
        return;
      }
      for (Value operand : defOp->getOperands())
        walk(operand);
      activeOps.erase(defOp);
      if (!ok)
        return;
      if (postorder.size() > kMaxNodes) {
        ok = false;
        return;
      }
      postorder.push_back(defOp);
      doneValues.insert(v);
    }
  };

  std::optional<JointTables> deriveJointTables(synth::CreateBundleOp bundle) {
    if (bundle.getNumOperands() != 2)
      return std::nullopt;
    ConeWalker walker;
    for (Value root : bundle.getOperands())
      walker.walk(root);
    if (!walker.ok || walker.support.size() < 2 || walker.support.size() > 3)
      return std::nullopt;
    unsigned n = walker.support.size();
    DenseMap<Value, APInt> eval;
    for (unsigned i = 0; i < n; ++i)
      eval[walker.support[i]] = circt::createVarMask(n, i, true);
    for (const auto &entry : walker.constVals)
      eval[entry.first] = entry.second.isZero() ? APInt::getZero(1u << n)
                                                : APInt::getAllOnes(1u << n);
    for (Operation *op : walker.postorder) {
      if (auto choiceOp = dyn_cast<synth::ChoiceOp>(op)) {
        if (choiceOp.getInputs().empty() ||
            !eval.contains(choiceOp.getInputs().front()))
          return std::nullopt;
        eval[choiceOp.getResult()] = eval[choiceOp.getInputs().front()];
      } else if (auto logicOp = dyn_cast<BooleanLogicOpInterface>(op)) {
        for (auto value : logicOp.getInputs())
          if (!eval.contains(value))
            return std::nullopt;
        eval[logicOp.getResult()] =
            logicOp.evaluateBooleanLogic([&](unsigned i) -> const APInt & {
              auto it = eval.find(logicOp.getInput(i));
              assert(it != eval.end() && "checked above");
              return it->second;
            });
      } else if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
        auto it = eval.find(xorOp.getOperand(0));
        if (it == eval.end())
          return std::nullopt;
        APInt result = it->second;
        for (unsigned i = 1; i < xorOp.getNumOperands(); ++i) {
          it = eval.find(xorOp.getOperand(i));
          if (it == eval.end())
            return std::nullopt;
          result ^= it->second;
        }
        eval[xorOp.getResult()] = result;
      } else if (auto andOp = dyn_cast<comb::AndOp>(op)) {
        APInt result = APInt::getAllOnes(1u << n);
        for (auto operand : andOp.getOperands()) {
          auto it = eval.find(operand);
          if (it == eval.end())
            return std::nullopt;
          result &= it->second;
        }
        eval[andOp.getResult()] = result;
      } else if (auto orOp = dyn_cast<comb::OrOp>(op)) {
        APInt result = APInt::getZero(1u << n);
        for (auto operand : orOp.getOperands()) {
          auto it = eval.find(operand);
          if (it == eval.end())
            return std::nullopt;
          result |= it->second;
        }
        eval[orOp.getResult()] = result;
      } else {
        return std::nullopt;
      }
    }
    // Discovery (DFS pre-) order is deterministic for fixed IR, unlike
    // pointer order, so tables stay in walk order: no reordering needed.
    // Matching tries every input permutation anyway.
    JointTables joint;
    joint.leaves.append(walker.support.begin(), walker.support.end());
    for (unsigned o = 0; o < 2; ++o) {
      auto it = eval.find(bundle.getOperand(o));
      if (it == eval.end())
        return std::nullopt;
      joint.tables[o] = it->second.getZExtValue();
    }
    return joint;
  }

  /// Try every input permutation and both output orders against each cell.
  bool mapOneBundle(synth::CreateBundleOp bundle,
                    ArrayRef<MultiOutputCell> cells) {
    auto joint = deriveJointTables(bundle);
    if (!joint) {
      eraseBundle(bundle);
      return true;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "bundle joint n=" << joint->leaves.size() << " T0=";
      llvm::dbgs().write_hex(joint->tables[0]);
      llvm::dbgs() << " T1=";
      llvm::dbgs().write_hex(joint->tables[1]);
      llvm::dbgs() << "\n";
    });
    unsigned n = joint->leaves.size();
    SmallVector<unsigned, 4> inPerm;
    for (unsigned i = 0; i < n; ++i)
      inPerm.push_back(i);
    do {
      for (unsigned swap = 0; swap < 2; ++swap) {
        for (const auto &cell : cells) {
          if (cell.numInputs != n)
            continue;
          bool match = true;
          for (unsigned row = 0, rows = 1u << n; row < rows && match; ++row) {
            unsigned cellRow = 0;
            for (unsigned j = 0; j < n; ++j)
              if ((row >> j) & 1u)
                cellRow |= 1u << inPerm[j];
            for (unsigned o = 0; o < 2; ++o) {
              unsigned cellOut = swap ? 1u - o : o;
              if (((joint->tables[o] >> row) & 1u) !=
                  ((cell.tables[cellOut] >> cellRow) & 1u)) {
                match = false;
                break;
              }
            }
          }
          if (match) {
            rewriteBundle(bundle, cell, inPerm, swap, *joint);
            return true;
          }
        }
      }
    } while (std::next_permutation(inPerm.begin(), inPerm.end()));
    eraseBundle(bundle);
    return true;
  }

  void rewriteBundle(synth::CreateBundleOp bundle, const MultiOutputCell &cell,
                     ArrayRef<unsigned> inPerm, unsigned swap,
                     const JointTables &joint) {
    OpBuilder builder(bundle->getBlock(), bundle->getIterator());
    SmallVector<Value> inputs;
    for (unsigned i = 0; i < joint.leaves.size(); ++i) {
      // Port i is driven by the canonical leaf mapped to it: invert perm.
      for (unsigned j = 0; j < joint.leaves.size(); ++j)
        if (inPerm[j] == i) {
          inputs.push_back(joint.leaves[j]);
          break;
        }
    }
    auto inst = hw::InstanceOp::create(builder, bundle.getLoc(), cell.module,
                                       "mapped_mo", inputs);
    // Redirect EVERY use of both roots (bundle outputs and original
    // consumers) to the cell outputs: the discrete roots always die, and
    // shared producers survive via the use_empty gate in eraseDeadCone.
    // This is what makes joint mapping profitable by construction.
    Value sumRoot = bundle.getOperand(0), carryRoot = bundle.getOperand(1);
    SmallVector<Operation *, 2> rootOps{sumRoot.getDefiningOp(),
                                        carryRoot.getDefiningOp()};
    Value cellSum = inst.getResult(swap ? 1 : 0);
    Value cellCarry = inst.getResult(swap ? 0 : 1);
    bundle.getResult(0).replaceAllUsesWith(cellSum);
    bundle.getResult(1).replaceAllUsesWith(cellCarry);
    sumRoot.replaceAllUsesWith(cellSum);
    carryRoot.replaceAllUsesWith(cellCarry);
    bundle.erase();
    for (Operation *rootOp : rootOps)
      eraseDeadCone(rootOp);
    ++numMapped;
  }

  /// Erase newly-dead combinational producers, cascading. Only pure logic
  /// ops are eligible; cells, state, and shared values are never touched
  /// (use_empty gates everything). Safe: every walked value is live by
  /// construction (roots are used by the bundle being processed).
  void eraseDeadCone(Operation *root) {
    SmallVector<Operation *> worklist;
    DenseSet<Operation *> seen;
    if (root)
      worklist.push_back(root);
    for (unsigned i = 0; i < worklist.size(); ++i) {
      Operation *op = worklist[i];
      if (!seen.insert(op).second || !op->use_empty())
        continue;
      if (!isa<synth::aig::AndInverterOp, synth::XorInverterOp,
               synth::MuxInverterOp, synth::DotOp, synth::MajorityOp,
               synth::OneHotOp, synth::GambleOp, comb::AndOp, comb::OrOp,
               comb::XorOp, hw::ConstantOp>(op))
        continue;
      for (Value operand : op->getOperands())
        if (auto *defOp = operand.getDefiningOp())
          worklist.push_back(defOp);
      op->erase();
    }
  }

  void eraseBundle(synth::CreateBundleOp bundle) {
    for (unsigned i = 0, e = bundle.getNumOperands(); i < e; ++i)
      bundle.getResult(i).replaceAllUsesWith(bundle.getOperand(i));
    bundle.erase();
    ++numErased;
  }
};

} // namespace
