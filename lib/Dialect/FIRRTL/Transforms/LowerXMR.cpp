//===- LowerXMR.cpp - FIRRTL Lower to XMR -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FIRRTL XMR Lowering.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HierPathCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-lower-xmr"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERXMR
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using hw::InnerRefAttr;

/// The LowerXMRPass will replace every RefResolveOp with an XMR encoded within
/// a verbatim expr op. This also removes every RefType port from the modules
/// and corresponding instances. This is a dataflow analysis over a very
/// constrained RefType. Domain of the dataflow analysis is the set of all
/// RefSendOps. It computes an interprocedural reaching definitions (of
/// RefSendOp) analysis. Essentially every RefType value must be mapped to one
/// and only one RefSendOp. The analysis propagates the dataflow from every
/// RefSendOp to every value of RefType across modules. The RefResolveOp is the
/// final leaf into which the dataflow must reach.
///
/// Since there can be multiple readers, multiple RefResolveOps can be reachable
/// from a single RefSendOp. To support multiply instantiated modules and
/// multiple readers, it is essential to track the path to the RefSendOp, other
/// than just the RefSendOp. For example, if there exists a wire `xmr_wire` in
/// module `Foo`, the algorithm needs to support generating Top.Bar.Foo.xmr_wire
/// and Top.Foo.xmr_wire and Top.Zoo.Foo.xmr_wire for different instance paths
/// that exist in the circuit.

namespace {
struct XMRNode {
  using NextNodeOnPath = std::optional<size_t>;
  using SymOrIndexOp = PointerUnion<Attribute, Operation *>;
  SymOrIndexOp info;
  NextNodeOnPath next;
};
[[maybe_unused]] llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                               const XMRNode &node) {
  os << "node(";
  if (auto attr = dyn_cast<Attribute>(node.info))
    os << "path=" << attr;
  else {
    auto subOp = cast<RefSubOp>(cast<Operation *>(node.info));
    os << "index=" << subOp.getIndex() << " (-> " << subOp.getType() << ")";
  }
  os << ", next=" << node.next << ")";
  return os;
}

/// Track information about operations being created in a module.  This is used
/// to generate more compact code and reuse operations where possible.
class ModuleState {

public:
  ModuleState(FModuleOp &moduleOp) : body(moduleOp.getBodyBlock()) {}

  /// Return the existing XMRRefOp for this type, symbol, and suffix for this
  /// module.  Otherwise, create a new one.  The first XMRRefOp will be created
  /// at the beginning of the module.  Subsequent XMRRefOps will be created
  /// immediately following the first one.
  Value getOrCreateXMRRefOp(Type type, FlatSymbolRefAttr symbol,
                            StringAttr suffix, ImplicitLocOpBuilder &builder) {
    // Return the saved XMRRefOp.
    auto it = xmrRefCache.find({type, symbol, suffix});
    if (it != xmrRefCache.end())
      return it->getSecond();

    // Create a new XMRRefOp.
    OpBuilder::InsertionGuard guard(builder);
    if (xmrRefPoint.isSet())
      builder.restoreInsertionPoint(xmrRefPoint);
    else
      builder.setInsertionPointToStart(body);

    Value xmr = XMRRefOp::create(builder, type, symbol, suffix);
    xmrRefCache.insert({{type, symbol, suffix}, xmr});

    xmrRefPoint = builder.saveInsertionPoint();
    return xmr;
  };

private:
  /// The module's body.  This is used to set the insertion point for the first
  /// created operation.
  Block *body;

  /// Map used to know if we created this XMRRefOp before.
  DenseMap<std::tuple<Type, SymbolRefAttr, StringAttr>, Value> xmrRefCache;

  /// The saved insertion point for XMRRefOps.
  OpBuilder::InsertPoint xmrRefPoint;
};
} // end anonymous namespace

class LowerXMRPass : public circt::firrtl::impl::LowerXMRBase<LowerXMRPass> {

  void runOnOperation() override {
    // Populate a CircuitNamespace that can be used to generate unique
    // circuit-level symbols.
    CircuitNamespace ns(getOperation());
    circuitNamespace = &ns;

    hw::HierPathCache pc(
        &ns, OpBuilder::InsertPoint(getOperation().getBodyBlock(),
                                    getOperation().getBodyBlock()->begin()));
    hierPathCache = &pc;

    llvm::EquivalenceClasses<Value> eq;
    dataFlowClasses = &eq;

    InstanceGraph &instanceGraph = getAnalysis<InstanceGraph>();
    SymbolTable &symTable = getAnalysis<SymbolTable>();
    SmallVector<RefResolveOp> resolveOps;
    SmallVector<RefSubOp> indexingOps;
    SmallVector<Operation *> forceAndReleaseOps;
    // The dataflow function, that propagates the reachable RefSendOp across
    // RefType Ops.
    auto transferFunc = [&](Operation *op) -> LogicalResult {
      return TypeSwitch<Operation *, LogicalResult>(op)
          .Case<RefSendOp>([&](RefSendOp send) {
            // Get a reference to the actual signal to which the XMR will be
            // generated.
            Value xmrDef = send.getBase();
            if (isZeroWidth(send.getType().getType())) {
              markForRemoval(send);
              return success();
            }

            if (auto verbExpr = xmrDef.getDefiningOp<VerbatimExprOp>())
              if (verbExpr.getSymbolsAttr().empty() && verbExpr->hasOneUse()) {
                // This represents the internal path into a module. For
                // generating the correct XMR, no node can be created in this
                // module. Create a null InnerRef and ensure the hierarchical
                // path ends at the parent that instantiates this module.
                auto inRef = InnerRefAttr();
                auto ind = addReachingSendsEntry(send.getResult(), inRef);
                xmrPathSuffix[ind] = verbExpr.getText();
                markForRemoval(verbExpr);
                markForRemoval(send);
                return success();
              }
            // Get an InnerRefAttr to the value being sent.
            auto *xmrDefOp = xmrDef.getDefiningOp();

            // Add the symbol directly if the operation targets a specific
            // result. This ensures that operations like InstanceOp and MemOp,
            // which have inner symbols that target the operation itself (not a
            // specific result), still get nodes created to distinguish which
            // result is being referenced.
            if (auto innerSymOp =
                    dyn_cast_or_null<hw::InnerSymbolOpInterface>(xmrDefOp))
              if (innerSymOp.getTargetResultIndex()) {
                addReachingSendsEntry(send.getResult(), getInnerRefTo(xmrDef));
                markForRemoval(send);
                return success();
              }

            // The operation cannot support an inner symbol, or it has
            // multiple results and doesn't target a specific result, so
            // create a node and replace all uses of the original value with
            // the node (except the node itself).
            ImplicitLocOpBuilder b(xmrDef.getLoc(), &getContext());
            b.setInsertionPointAfterValue(xmrDef);
            SmallString<32> opName;
            auto nameKind = NameKindEnum::DroppableName;

            if (auto [name, rootKnown] = getFieldName(
                    getFieldRefFromValue(xmrDef, /*lookThroughCasts=*/true),
                    /*nameSafe=*/true);
                rootKnown) {
              opName = name + "_probe";
              nameKind = NameKindEnum::InterestingName;
            } else if (xmrDefOp) {
              // Inspect "name" directly for ops that aren't named by above.
              // (e.g., firrtl.constant)
              if (auto name = xmrDefOp->getAttrOfType<StringAttr>("name")) {
                (Twine(name.strref()) + "_probe").toVector(opName);
                nameKind = NameKindEnum::InterestingName;
              }
            }
            auto node = NodeOp::create(b, xmrDef, opName, nameKind);
            auto newValue = node.getResult();
            // Replace all uses except the node itself and except when the value
            // is the destination of a connect (operand 0). We need to preserve
            // connect destinations to maintain proper flow semantics.
            xmrDef.replaceUsesWithIf(newValue, [&](OpOperand &operand) {
              if (operand.getOwner() == node.getOperation())
                return false;
              if (isa<FConnectLike>(operand.getOwner()) &&
                  operand.getOperandNumber() == 0)
                return false;
              return true;
            });
            xmrDef = newValue;

            // Create a new entry for this RefSendOp. The path is currently
            // local.
            addReachingSendsEntry(send.getResult(), getInnerRefTo(xmrDef));
            markForRemoval(send);
            return success();
          })
          .Case<RWProbeOp>([&](RWProbeOp rwprobe) {
            if (!isZeroWidth(rwprobe.getType().getType()))
              addReachingSendsEntry(rwprobe.getResult(), rwprobe.getTarget());
            markForRemoval(rwprobe);
            return success();
          })
          .Case<MemOp>([&](MemOp mem) {
            // MemOp can produce debug ports of RefType. Each debug port
            // represents the RefType for the corresponding register of the
            // memory. Since the memory is not yet generated the register name
            // is assumed to be "Memory". Note that MemOp creates RefType
            // without a RefSend.
            for (const auto &res : llvm::enumerate(mem.getResults()))
              if (isa<RefType>(mem.getResult(res.index()).getType())) {
                auto inRef = getInnerRefTo(mem);
                auto ind = addReachingSendsEntry(res.value(), inRef);
                xmrPathSuffix[ind] = "Memory";
                // Just node that all the debug ports of memory must be removed.
                // So this does not record the port index.
                refPortsToRemoveMap[mem].resize(1);
              }
            return success();
          })
          .Case<InstanceOp>(
              [&](auto inst) { return handleInstanceOp(inst, instanceGraph); })
          .Case<InstanceChoiceOp>([&](auto inst) {
            return handleInstanceChoiceOp(inst, instanceGraph, symTable);
          })
          .Case<FConnectLike>([&](FConnectLike connect) {
            // Ignore BaseType.
            if (!isa<RefType>(connect.getSrc().getType()))
              return success();
            markForRemoval(connect);
            if (isZeroWidth(
                    type_cast<RefType>(connect.getSrc().getType()).getType()))
              return success();
            // Merge the dataflow classes of destination into the source of the
            // Connect. This handles two cases:
            // 1. If the dataflow at the source is known, then the
            // destination is also inferred. By merging the dataflow class of
            // destination with source, every value reachable from the
            // destination automatically infers a reaching RefSend.
            // 2. If dataflow at source is unkown, then just record that both
            // source and destination will have the same dataflow information.
            // Later in the pass when the reaching RefSend is inferred at the
            // leader of the dataflowClass, then we automatically infer the
            // dataflow at this connect and every value reachable from the
            // destination.
            dataFlowClasses->unionSets(connect.getSrc(), connect.getDest());
            return success();
          })
          .Case<RefSubOp>([&](RefSubOp op) -> LogicalResult {
            markForRemoval(op);
            if (isZeroWidth(op.getType().getType()))
              return success();

            // Enqueue for processing after visiting other operations.
            indexingOps.push_back(op);
            return success();
          })
          .Case<RefResolveOp>([&](RefResolveOp resolve) {
            // Merge dataflow, under the same conditions as above for Connect.
            // 1. If dataflow at the resolve.getRef is known, propagate that to
            // the result. This is true for downward scoped XMRs, that is,
            // RefSendOp must be visited before the corresponding RefResolveOp
            // is visited.
            // 2. Else, just record that both result and ref should have the
            // same reaching RefSend. This condition is true for upward scoped
            // XMRs. That is, RefResolveOp can be visited before the
            // corresponding RefSendOp is recorded.

            markForRemoval(resolve);
            if (!isZeroWidth(resolve.getType()))
              dataFlowClasses->unionSets(resolve.getRef(), resolve.getResult());
            resolveOps.push_back(resolve);
            return success();
          })
          .Case<RefCastOp>([&](RefCastOp op) {
            markForRemoval(op);
            if (!isZeroWidth(op.getType().getType()))
              dataFlowClasses->unionSets(op.getInput(), op.getResult());
            return success();
          })
          .Case<Forceable>([&](Forceable op) {
            // Handle declarations containing refs as "data".
            if (type_isa<RefType>(op.getDataRaw().getType())) {
              markForRemoval(op);
              return success();
            }

            // Otherwise, if forceable track the rwprobe result.
            if (!op.isForceable() || op.getDataRef().use_empty() ||
                isZeroWidth(op.getDataType()))
              return success();

            addReachingSendsEntry(op.getDataRef(), getInnerRefTo(op));
            return success();
          })
          .Case<RefForceOp, RefForceInitialOp, RefReleaseOp,
                RefReleaseInitialOp>([&](auto op) {
            forceAndReleaseOps.push_back(op);
            return success();
          })
          .Default([&](auto) { return success(); });
    };

    SmallVector<FModuleOp> publicModules;

    // Traverse the modules in post order.
    auto result = instanceGraph.walkPostOrder([&](auto &node) -> LogicalResult {
      auto module = dyn_cast<FModuleOp>(*node.getModule());
      if (!module)
        return success();
      LLVM_DEBUG(llvm::dbgs()
                 << "Traversing module:" << module.getModuleNameAttr() << "\n");

      moduleStates.insert({module, ModuleState(module)});

      if (module.isPublic())
        publicModules.push_back(module);

      auto result = module.walk([&](Operation *op) {
        if (transferFunc(op).failed())
          return WalkResult::interrupt();
        return WalkResult::advance();
      });

      if (result.wasInterrupted())
        return failure();

      // Since we walk operations pre-order and not along dataflow edges,
      // ref.sub may not be resolvable when we encounter them (they're not
      // just unification). This can happen when refs go through an output
      // port or input instance result and back into the design. Handle these
      // by walking them, resolving what we can, until all are handled or
      // nothing can be resolved.
      while (!indexingOps.empty()) {
        // Grab the set of unresolved ref.sub's.
        decltype(indexingOps) worklist;
        worklist.swap(indexingOps);

        for (auto op : worklist) {
          auto inputEntry =
              getRemoteRefSend(op.getInput(), /*errorIfNotFound=*/false);
          // If we can't resolve, add back and move on.
          if (!inputEntry)
            indexingOps.push_back(op);
          else
            addReachingSendsEntry(op.getResult(), op.getOperation(),
                                  inputEntry);
        }
        // If nothing was resolved, give up.
        if (worklist.size() == indexingOps.size()) {
          auto op = worklist.front();
          getRemoteRefSend(op.getInput());
          op.emitError(
                "indexing through probe of unknown origin (input probe?)")
              .attachNote(op.getInput().getLoc())
              .append("indexing through this reference");
          return failure();
        }
      }

      // Record all the RefType ports to be removed later.
      size_t numPorts = module.getNumPorts();
      for (size_t portNum = 0; portNum < numPorts; ++portNum)
        if (isa<RefType>(module.getPortType(portNum)))
          setPortToRemove(module, portNum, numPorts);

      return success();
    });
    if (failed(result))
      return signalPassFailure();

    LLVM_DEBUG({
      for (const auto &I :
           *dataFlowClasses) { // Iterate over all of the equivalence sets.
        if (!I->isLeader())
          continue; // Ignore non-leader sets.
        // Print members in this set.
        llvm::interleave(dataFlowClasses->members(*I), llvm::dbgs(), "\n");
        llvm::dbgs() << "\n dataflow at leader::" << I->getData() << "\n =>";
        auto iter = dataflowAt.find(I->getData());
        if (iter != dataflowAt.end()) {
          for (auto init = refSendPathList[iter->getSecond()]; init.next;
               init = refSendPathList[*init.next])
            llvm::dbgs() << "\n " << init;
        }
        llvm::dbgs() << "\n Done\n"; // Finish set.
      }
    });
    for (auto refResolve : resolveOps)
      if (handleRefResolve(refResolve).failed())
        return signalPassFailure();
    for (auto *op : forceAndReleaseOps)
      if (failed(handleForceReleaseOp(op)))
        return signalPassFailure();
    for (auto module : publicModules) {
      if (failed(handlePublicModuleRefPorts(module)))
        return signalPassFailure();
    }
    garbageCollect();

    // Clean up
    moduleNamespaces.clear();
    visitedModules.clear();
    dataflowAt.clear();
    refSendPathList.clear();
    dataFlowClasses = nullptr;
    refPortsToRemoveMap.clear();
    opsToRemove.clear();
    xmrPathSuffix.clear();
    circuitNamespace = nullptr;
    hierPathCache = nullptr;
  }

  /// Generate the ABI ref_<module> prefix string into `prefix`.
  void getRefABIPrefix(FModuleLike mod, SmallVectorImpl<char> &prefix) {
    auto modName = mod.getModuleName();
    if (auto ext = dyn_cast<FExtModuleOp>(*mod))
      modName = ext.getExtModuleName();
    (Twine("ref_") + modName).toVector(prefix);
  }

  /// Get full macro name as StringAttr for the specified ref port.
  /// Uses existing 'prefix', optionally preprends the backtick character.
  StringAttr getRefABIMacroForPort(FModuleLike mod, size_t portIndex,
                                   const Twine &prefix, bool backTick = false) {
    return StringAttr::get(&getContext(), Twine(backTick ? "`" : "") + prefix +
                                              "_" + mod.getPortName(portIndex));
  }

  LogicalResult resolveReferencePath(mlir::TypedValue<RefType> refVal,
                                     ImplicitLocOpBuilder builder,
                                     mlir::FlatSymbolRefAttr &ref,
                                     SmallString<128> &stringLeaf) {
    assert(stringLeaf.empty());

    auto remoteOpPath = getRemoteRefSend(refVal);
    if (!remoteOpPath)
      return failure();
    SmallVector<Attribute> refSendPath;
    SmallVector<RefSubOp> indexing;
    size_t lastIndex;
    while (remoteOpPath) {
      lastIndex = *remoteOpPath;
      auto entr = refSendPathList[*remoteOpPath];
      if (entr.info)
        TypeSwitch<XMRNode::SymOrIndexOp>(entr.info)
            .Case<Attribute>([&](auto attr) {
              // If the path is a singular verbatim expression, the attribute of
              // the send path list entry will be null.
              if (attr)
                refSendPath.push_back(attr);
            })
            .Case<Operation *>(
                [&](auto *op) { indexing.push_back(cast<RefSubOp>(op)); });
      remoteOpPath = entr.next;
    }
    auto iter = xmrPathSuffix.find(lastIndex);

    // If this xmr has a suffix string (internal path into a module, that is not
    // yet generated).
    if (iter != xmrPathSuffix.end()) {
      if (!refSendPath.empty())
        stringLeaf.append(".");
      stringLeaf.append(iter->getSecond());
    }

    assert(!(refSendPath.empty() && stringLeaf.empty()) &&
           "nothing to index through");

    // All indexing done as the ref is plumbed around indexes through
    // the target/referent, not the current point of the path which
    // describes how to access the referent we're indexing through.
    // Above we gathered all indexing operations, so now append them
    // to the path (after any relevant `xmrPathSuffix`) to reach
    // the target element.
    // Generating these strings here (especially if ref is sent
    // out from a different design) is fragile but should get this
    // working well enough while sorting out how to do this better.
    // Some discussion of this can be found here:
    // https://github.com/llvm/circt/pull/5551#discussion_r1258908834
    for (auto subOp : llvm::reverse(indexing)) {
      TypeSwitch<FIRRTLBaseType>(subOp.getInput().getType().getType())
          .Case<FVectorType, OpenVectorType>([&](auto vecType) {
            (Twine("[") + Twine(subOp.getIndex()) + "]").toVector(stringLeaf);
          })
          .Case<BundleType, OpenBundleType>([&](auto bundleType) {
            auto fieldName = bundleType.getElementName(subOp.getIndex());
            stringLeaf.append({".", fieldName});
          });
    }

    if (!refSendPath.empty())
      // Compute the HierPathOp that stores the path.
      ref = FlatSymbolRefAttr::get(
          hierPathCache
              ->getOrCreatePath(builder.getArrayAttr(refSendPath),
                                builder.getLoc())
              .getSymNameAttr());

    return success();
  }

  LogicalResult resolveReference(mlir::TypedValue<RefType> refVal,
                                 ImplicitLocOpBuilder &builder,
                                 FlatSymbolRefAttr &ref, StringAttr &xmrAttr) {
    auto remoteOpPath = getRemoteRefSend(refVal);
    if (!remoteOpPath)
      return failure();

    SmallString<128> xmrString;
    if (failed(resolveReferencePath(refVal, builder, ref, xmrString)))
      return failure();
    xmrAttr =
        xmrString.empty() ? StringAttr{} : builder.getStringAttr(xmrString);

    return success();
  }

  // Replace the Force/Release's ref argument with a resolved XMRRef.
  LogicalResult handleForceReleaseOp(Operation *op) {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<RefForceOp, RefForceInitialOp, RefReleaseOp, RefReleaseInitialOp>(
            [&](auto op) {
              // Drop if zero-width target.
              auto destType = op.getDest().getType();
              if (isZeroWidth(destType.getType())) {
                op.erase();
                return success();
              }

              ImplicitLocOpBuilder builder(op.getLoc(), op);
              FlatSymbolRefAttr ref;
              StringAttr str;
              if (failed(resolveReference(op.getDest(), builder, ref, str)))
                return failure();

              if (!ref) {
                // No hierpath (e.g., InstanceChoiceOp): force/release not
                // supported yet.
                op.emitError(
                    "force/release operations on instance choice ref ports are "
                    "not yet supported");
                return failure();
              }

              Value xmr =
                  moduleStates.find(op->template getParentOfType<FModuleOp>())
                      ->getSecond()
                      .getOrCreateXMRRefOp(destType, ref, str, builder);
              op.getDestMutable().assign(xmr);
              return success();
            })
        .Default([](auto *op) {
          return op->emitError("unexpected operation kind");
        });
  }

  // Replace the RefResolveOp with verbatim op representing the XMR.
  LogicalResult handleRefResolve(RefResolveOp resolve) {
    auto resWidth = getBitWidth(resolve.getType());
    if (resWidth.has_value() && *resWidth == 0) {
      // Donot emit 0 width XMRs, replace it with constant 0.
      ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
      auto zeroUintType = UIntType::get(builder.getContext(), 0);
      auto zeroC = builder.createOrFold<BitCastOp>(
          resolve.getType(), ConstantOp::create(builder, zeroUintType,
                                                getIntZerosAttr(zeroUintType)));
      resolve.getResult().replaceAllUsesWith(zeroC);
      return success();
    }

    FlatSymbolRefAttr ref;
    StringAttr str;
    ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
    if (failed(resolveReference(resolve.getRef(), builder, ref, str)))
      return failure();

    Value result;
    if (ref) {
      // Standard case: hierpath with optional suffix
      result = XMRDerefOp::create(builder, resolve.getType(), ref, str);
    } else {
      // No hierpath (e.g., InstanceChoiceOp): the suffix is the complete XMR
      // Use VerbatimWireOp to emit the macro directly
      result = VerbatimWireOp::create(builder, resolve.getType(),
                                      str ? str.getValue() : "");
    }
    resolve.getResult().replaceAllUsesWith(result);
    return success();
  }

  void setPortToRemove(Operation *op, size_t index, size_t numPorts) {
    if (refPortsToRemoveMap[op].size() < numPorts)
      refPortsToRemoveMap[op].resize(numPorts);
    refPortsToRemoveMap[op].set(index);
  }

  // Propagate the reachable RefSendOp across modules.
  LogicalResult handleInstanceOp(InstanceOp inst,
                                 InstanceGraph &instanceGraph) {
    Operation *mod = inst.getReferencedModule(instanceGraph);
    if (auto extRefMod = dyn_cast<FExtModuleOp>(mod)) {
      auto numPorts = inst.getNumResults();
      SmallString<128> circuitRefPrefix;

      /// Get the resolution string for this ref-type port.
      auto getPath = [&](size_t portNo) {
        // Otherwise, we're using the ref ABI.  Generate the prefix string
        // and return the macro for the specified port.
        if (circuitRefPrefix.empty())
          getRefABIPrefix(extRefMod, circuitRefPrefix);

        return getRefABIMacroForPort(extRefMod, portNo, circuitRefPrefix, true);
      };

      for (const auto &res : llvm::enumerate(inst.getResults())) {
        if (!isa<RefType>(inst.getResult(res.index()).getType()))
          continue;

        auto inRef = getInnerRefTo(inst);
        auto ind = addReachingSendsEntry(res.value(), inRef);

        xmrPathSuffix[ind] = getPath(res.index());
        // The instance result and module port must be marked for removal.
        setPortToRemove(inst, res.index(), numPorts);
        setPortToRemove(extRefMod, res.index(), numPorts);
      }
      return success();
    }
    auto refMod = dyn_cast<FModuleOp>(mod);
    bool multiplyInstantiated = !visitedModules.insert(refMod).second;
    for (size_t portNum = 0, numPorts = inst.getNumResults();
         portNum < numPorts; ++portNum) {
      auto instanceResult = inst.getResult(portNum);
      if (!isa<RefType>(instanceResult.getType()))
        continue;
      if (!refMod)
        return inst.emitOpError("cannot lower ext modules with RefType ports");
      // Reference ports must be removed.
      setPortToRemove(inst, portNum, numPorts);
      // Drop the dead-instance-ports.
      if (instanceResult.use_empty() ||
          isZeroWidth(type_cast<RefType>(instanceResult.getType()).getType()))
        continue;
      auto refModuleArg = refMod.getArgument(portNum);
      if (inst.getPortDirection(portNum) == Direction::Out) {
        // For output instance ports, the dataflow is into this module.
        // Get the remote RefSendOp, that flows through the module ports.
        // If dataflow at remote module argument does not exist, error out.
        auto remoteOpPath = getRemoteRefSend(refModuleArg);
        if (!remoteOpPath)
          return failure();
        // Get the path to reaching refSend at the referenced module argument.
        // Now append this instance to the path to the reaching refSend.
        addReachingSendsEntry(instanceResult, getInnerRefTo(inst),
                              remoteOpPath);
      } else {
        // For input instance ports, the dataflow is into the referenced module.
        // Input RefType port implies, generating an upward scoped XMR.
        // No need to add the instance context, since downward reference must be
        // through single instantiated modules.
        if (multiplyInstantiated)
          return refMod.emitOpError(
                     "multiply instantiated module with input RefType port '")
                 << refMod.getPortName(portNum) << "'";
        dataFlowClasses->unionSets(
            dataFlowClasses->getOrInsertLeaderValue(refModuleArg),
            dataFlowClasses->getOrInsertLeaderValue(instanceResult));
      }
    }
    return success();
  }

  /// Resolve the XMR path for a target module's ref port.
  /// Returns the HierPath symbol and suffix string for the path.
  LogicalResult resolveModuleRefPortPath(FModuleOp targetMod, size_t portNum,
                                         ImplicitLocOpBuilder &builder,
                                         FlatSymbolRefAttr &hierPathRef,
                                         SmallString<128> &suffix) {
    auto refModuleArg = targetMod.getArgument(portNum);
    auto remoteOpPath =
        getRemoteRefSend(refModuleArg, /*errorIfNotFound=*/false);
    if (!remoteOpPath)
      return failure();

    // Collect InnerRefAttrs and indexing operations from the path.
    SmallVector<Attribute> refSendPath;
    SmallVector<RefSubOp> indexing;
    size_t lastIndex = *remoteOpPath;

    while (remoteOpPath) {
      lastIndex = *remoteOpPath;
      auto entr = refSendPathList[*remoteOpPath];
      if (entr.info) {
        if (auto attr = dyn_cast<Attribute>(entr.info))
          refSendPath.push_back(attr);
        else if (auto *op = dyn_cast<Operation *>(entr.info))
          indexing.push_back(cast<RefSubOp>(op));
      }
      remoteOpPath = entr.next;
    }

    // Create HierPath from the collected InnerRefAttrs.
    if (!refSendPath.empty()) {
      auto hierPath = hierPathCache->getOrCreatePath(
          builder.getArrayAttr(refSendPath), builder.getLoc());
      hierPathRef = FlatSymbolRefAttr::get(hierPath.getSymNameAttr());
    }

    // Append suffix from xmrPathSuffix map (e.g., memory/extmodule paths).
    if (auto iter = xmrPathSuffix.find(lastIndex); iter != xmrPathSuffix.end())
      suffix.append(iter->getSecond());

    // Append indexing operations to the suffix.
    for (auto subOp : llvm::reverse(indexing)) {
      TypeSwitch<FIRRTLBaseType>(subOp.getInput().getType().getType())
          .Case<FVectorType, OpenVectorType>([&](auto) {
            (Twine("[") + Twine(subOp.getIndex()) + "]").toVector(suffix);
          })
          .Case<BundleType, OpenBundleType>([&](auto bundleType) {
            suffix.append({".", bundleType.getElementName(subOp.getIndex())});
          });
    }

    return success();
  }

  /// Handle XMR lowering for InstanceChoiceOp.
  /// Generates macro-based XMR paths with ifdef guards for each target choice.
  ///
  /// Note: This function generates emit::FileOp directly rather than deferring
  /// to handlePublicModuleRefPorts because:
  /// 1. Instance choice ref ports require complex ifdef-guarded macro
  /// definitions
  /// 2. The path resolution logic needs access to the instance choice's target
  ///    choices and option information
  /// 3. Generating files here keeps all instance choice XMR logic localized
  ///
  /// Example output:
  ///   `ifdef __option__Platform_FPGA
  ///     `define ref_Top_inst_probe `__target_Platform_Top_inst.inner.r
  ///   `elsif __option__Platform_ASIC
  ///     `define ref_Top_inst_probe `__target_Platform_Top_inst.middle.deep.r
  ///   `else
  ///     `define ref_Top_inst_probe `__target_Platform_Top_inst.r
  ///   `endif
  LogicalResult handleInstanceChoiceOp(InstanceChoiceOp inst,
                                       InstanceGraph &instanceGraph,
                                       SymbolTable &symTable) {
    auto parentModule = inst->getParentOfType<FModuleOp>();

    // This should have been set by the PopulateInstanceChoiceSymbols pass.
    auto instanceMacro = inst.getInstanceMacroAttr();
    if (!instanceMacro)
      return inst.emitOpError("missing instanceMacro attribute - ensure "
                              "PopulateInstanceChoiceSymbols pass has run");

    auto optionName = inst.getOptionNameAttr();
    auto numPorts = inst.getNumResults();
    auto *body = getOperation().getBodyBlock();
    auto declBuilder = ImplicitLocOpBuilder::atBlockBegin(inst.getLoc(), body);

    // Get all target choices (case -> module mappings).
    auto targetChoices = inst.getTargetChoices();
    auto defaultTarget = inst.getDefaultTargetAttr();

    // Path information for a module's ref port.
    struct PathInfo {
      FlatSymbolRefAttr hierPath;
      std::string suffix;
    };
    DenseMap<std::pair<StringRef, size_t>, PathInfo> pathCache;

    // Get or compute XMR path for a module's ref port.
    auto getModuleXMRPath = [&](FlatSymbolRefAttr moduleRef,
                                size_t portNum) -> std::optional<PathInfo> {
      auto key = std::make_pair(moduleRef.getValue(), portNum);
      if (auto it = pathCache.find(key); it != pathCache.end())
        return it->second;

      auto *node = instanceGraph.lookup(moduleRef.getAttr());
      if (!node)
        return std::nullopt;
      auto targetMod = dyn_cast<FModuleOp>(*node->getModule());
      if (!targetMod)
        return std::nullopt;

      FlatSymbolRefAttr hierPathRef;
      SmallString<128> suffix;
      if (failed(resolveModuleRefPortPath(targetMod, portNum, declBuilder,
                                          hierPathRef, suffix)))
        return std::nullopt;

      PathInfo pathInfo{hierPathRef, suffix.str().str()};
      pathCache[key] = pathInfo;
      return pathInfo;
    };

    // Process ref ports and create macros.
    SmallVector<std::tuple<StringAttr, size_t>> refPorts;
    for (size_t portNum = 0; portNum < numPorts; ++portNum) {
      auto instanceResult = inst.getResult(portNum);
      if (!isa<RefType>(instanceResult.getType()))
        continue;

      setPortToRemove(inst, portNum, numPorts);

      // Skip dead or zero-width ports.
      if (instanceResult.use_empty() ||
          isZeroWidth(type_cast<RefType>(instanceResult.getType()).getType()))
        continue;

      // Generate macro name: ref_<parent>_<instance>_<port>
      SmallString<128> macroName;
      llvm::raw_svector_ostream(macroName)
          << "ref_" << parentModule.getName() << "_" << inst.getInstanceName()
          << "_" << inst.getPortName(portNum);

      auto macroNameAttr = StringAttr::get(&getContext(), macroName);
      sv::MacroDeclOp::create(declBuilder, macroNameAttr, ArrayAttr(),
                              StringAttr());
      refPorts.emplace_back(macroNameAttr, portNum);

      // Register the macro as the XMR path for this port.
      SmallString<128> macroPath("`");
      macroPath.append(macroName);
      auto ind = addReachingSendsEntry(instanceResult, InnerRefAttr());
      xmrPathSuffix[ind] = macroPath;
    }

    // Only generate header file for public modules with ref ports.
    if (refPorts.empty() || !parentModule.isPublic())
      return success();

    // Generate ref_<module>.sv file with macro definitions.
    SmallString<128> fileName("ref_");
    fileName.append(parentModule.getName());
    fileName.append(".sv");

    auto fileBuilder = ImplicitLocOpBuilder(inst.getLoc(), parentModule);
    emit::FileOp::create(fileBuilder, fileName, [&] {
      for (auto [macroNameAttr, portNum] : refPorts) {
        SmallVector<Attribute> symbols;
        DenseMap<Attribute, size_t> symbolIndices;

        // Add instance macro to symbols array (always at index 0).
        symbols.push_back(instanceMacro);

        // Build macro value with symbol substitution.
        // {{0}} = instance macro, {{1+}} = HierPaths
        auto buildMacroValue = [&](const PathInfo &pathInfo) -> std::string {
          SmallString<128> value("{{0}}");

          if (pathInfo.hierPath) {
            auto [it, inserted] =
                symbolIndices.try_emplace(pathInfo.hierPath, symbols.size());
            if (inserted)
              symbols.push_back(pathInfo.hierPath);
            value.append(".{{");
            value.append(std::to_string(it->second));
            value.append("}}");
          }

          if (!pathInfo.suffix.empty()) {
            if (!pathInfo.hierPath)
              value.append(".");
            value.append(pathInfo.suffix);
          }
          return value.str().str();
        };

        // Build option macro names for ifdef chain.
        SmallVector<StringRef> macroNames;
        SmallVector<std::string> macroNameStorage;
        for (auto [caseRef, moduleRef] : targetChoices) {
          SmallString<64> optionMacro("__option__");
          optionMacro.append(optionName.getValue());
          optionMacro.append("_");
          optionMacro.append(caseRef.getLeafReference().getValue());
          macroNameStorage.push_back(optionMacro.str().str());
          macroNames.push_back(macroNameStorage.back());
        }

        // Create nested ifdef structure for each target choice.
        auto createMacroDef = [&](std::optional<PathInfo> pathInfo) {
          std::string macroValue =
              pathInfo ? buildMacroValue(*pathInfo) : "{{0}}";
          sv::MacroDefOp::create(fileBuilder, macroNameAttr,
                                 fileBuilder.getStringAttr(macroValue),
                                 fileBuilder.getArrayAttr(symbols));
        };

        sv::createNestedIfDefs(
            macroNames,
            [&](StringRef macro, std::function<void()> thenCtor,
                std::function<void()> elseCtor) {
              sv::IfDefOp::create(fileBuilder, macro, std::move(thenCtor),
                                  std::move(elseCtor));
            },
            [&](size_t index) {
              createMacroDef(
                  getModuleXMRPath(targetChoices[index].second, portNum));
            },
            [&]() {
              createMacroDef(getModuleXMRPath(defaultTarget, portNum));
            });
      }
    });

    return success();
  }

  LogicalResult handlePublicModuleRefPorts(FModuleOp module) {
    auto *body = getOperation().getBodyBlock();

    // Find all the output reference ports.
    SmallString<128> circuitRefPrefix;
    SmallVector<std::tuple<StringAttr, StringAttr, ArrayAttr>> ports;
    auto declBuilder =
        ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), body);
    for (size_t portIndex = 0, numPorts = module.getNumPorts();
         portIndex != numPorts; ++portIndex) {
      auto refType = type_dyn_cast<RefType>(module.getPortType(portIndex));
      if (!refType || isZeroWidth(refType.getType()) ||
          module.getPortDirection(portIndex) != Direction::Out)
        continue;
      auto portValue =
          cast<mlir::TypedValue<RefType>>(module.getArgument(portIndex));
      mlir::FlatSymbolRefAttr ref;
      SmallString<128> stringLeaf;
      if (failed(resolveReferencePath(portValue, declBuilder, ref, stringLeaf)))
        return failure();

      SmallString<128> formatString;
      if (ref)
        formatString += "{{0}}";
      formatString += stringLeaf;

      // Insert a macro with the format:
      // ref_<module-name>_<ref-name> <path>
      if (circuitRefPrefix.empty())
        getRefABIPrefix(module, circuitRefPrefix);
      auto macroName =
          getRefABIMacroForPort(module, portIndex, circuitRefPrefix);
      sv::MacroDeclOp::create(declBuilder, macroName, ArrayAttr(),
                              StringAttr());
      ports.emplace_back(macroName, declBuilder.getStringAttr(formatString),
                         ref ? declBuilder.getArrayAttr({ref}) : ArrayAttr{});
    }

    // Create a file only if the module has at least one ref port.
    if (ports.empty())
      return success();

    // The macros will be exported to a `ref_<module-name>.sv` file.
    // In the IR, the file is inserted before the module.
    auto fileBuilder = ImplicitLocOpBuilder(module.getLoc(), module);
    emit::FileOp::create(fileBuilder, circuitRefPrefix + ".sv", [&] {
      for (auto [macroName, formatString, symbols] : ports) {
        sv::MacroDefOp::create(fileBuilder, FlatSymbolRefAttr::get(macroName),
                               formatString, symbols);
      }
    });

    return success();
  }

  /// Get the cached namespace for a module.
  hw::InnerSymbolNamespace &getModuleNamespace(FModuleLike module) {
    return moduleNamespaces.try_emplace(module, module).first->second;
  }

  InnerRefAttr getInnerRefTo(Value val) {
    if (auto arg = dyn_cast<BlockArgument>(val))
      return ::getInnerRefTo(
          cast<FModuleLike>(arg.getParentBlock()->getParentOp()),
          arg.getArgNumber(),
          [&](FModuleLike mod) -> hw::InnerSymbolNamespace & {
            return getModuleNamespace(mod);
          });
    return getInnerRefTo(val.getDefiningOp());
  }

  InnerRefAttr getInnerRefTo(Operation *op) {
    return ::getInnerRefTo(op,
                           [&](FModuleLike mod) -> hw::InnerSymbolNamespace & {
                             return getModuleNamespace(mod);
                           });
  }

  void markForRemoval(Operation *op) { opsToRemove.push_back(op); }

  std::optional<size_t> getRemoteRefSend(Value val,
                                         bool errorIfNotFound = true) {
    auto iter = dataflowAt.find(dataFlowClasses->getOrInsertLeaderValue(val));
    if (iter != dataflowAt.end())
      return iter->getSecond();
    if (!errorIfNotFound)
      return std::nullopt;
    // The referenced module must have already been analyzed, error out if the
    // dataflow at the child module is not resolved.
    if (BlockArgument arg = dyn_cast<BlockArgument>(val))
      arg.getOwner()->getParentOp()->emitError(
          "reference dataflow cannot be traced back to the remote read op "
          "for module port '")
          << dyn_cast<FModuleOp>(arg.getOwner()->getParentOp())
                 .getPortName(arg.getArgNumber())
          << "'";
    else
      val.getDefiningOp()->emitOpError(
          "reference dataflow cannot be traced back to the remote read op");
    signalPassFailure();
    return std::nullopt;
  }

  size_t
  addReachingSendsEntry(Value atRefVal, XMRNode::SymOrIndexOp info,
                        std::optional<size_t> continueFrom = std::nullopt) {
    auto leader = dataFlowClasses->getOrInsertLeaderValue(atRefVal);
    auto indx = refSendPathList.size();
    dataflowAt[leader] = indx;
    refSendPathList.push_back({info, continueFrom});
    return indx;
  }

  void garbageCollect() {
    // Now erase all the Ops and ports of RefType.
    // This needs to be done as the last step to ensure uses are erased before
    // the def is erased.
    for (Operation *op : llvm::reverse(opsToRemove))
      op->erase();
    for (auto iter : refPortsToRemoveMap)
      if (auto mod = dyn_cast<FModuleOp>(iter.getFirst()))
        mod.erasePorts(iter.getSecond());
      else if (auto mod = dyn_cast<FExtModuleOp>(iter.getFirst()))
        mod.erasePorts(iter.getSecond());
      else if (auto inst = dyn_cast<InstanceOp>(iter.getFirst())) {
        inst.cloneWithErasedPortsAndReplaceUses(iter.getSecond());
        inst.erase();
      } else if (auto instChoice =
                     dyn_cast<InstanceChoiceOp>(iter.getFirst())) {
        instChoice.cloneWithErasedPortsAndReplaceUses(iter.getSecond());
        instChoice.erase();
      } else if (auto mem = dyn_cast<MemOp>(iter.getFirst())) {
        // Remove all debug ports of the memory.
        ImplicitLocOpBuilder builder(mem.getLoc(), mem);
        SmallVector<Attribute, 4> resultNames;
        SmallVector<Type, 4> resultTypes;
        SmallVector<Attribute, 4> portAnnotations;
        SmallVector<Value, 4> oldResults;
        for (const auto &res : llvm::enumerate(mem.getResults())) {
          if (isa<RefType>(mem.getResult(res.index()).getType()))
            continue;
          resultNames.push_back(mem.getPortNameAttr(res.index()));
          resultTypes.push_back(res.value().getType());
          portAnnotations.push_back(mem.getPortAnnotation(res.index()));
          oldResults.push_back(res.value());
        }
        auto newMem = MemOp::create(
            builder, resultTypes, mem.getReadLatency(), mem.getWriteLatency(),
            mem.getDepth(), RUWBehavior::Undefined,
            builder.getArrayAttr(resultNames), mem.getNameAttr(),
            mem.getNameKind(), mem.getAnnotations(),
            builder.getArrayAttr(portAnnotations), mem.getInnerSymAttr(),
            mem.getInitAttr(), mem.getPrefixAttr());
        for (const auto &res : llvm::enumerate(oldResults))
          res.value().replaceAllUsesWith(newMem.getResult(res.index()));
        mem.erase();
      }
    opsToRemove.clear();
    refPortsToRemoveMap.clear();
    dataflowAt.clear();
    refSendPathList.clear();
    moduleStates.clear();
  }

  bool isZeroWidth(FIRRTLBaseType t) { return t.getBitWidthOrSentinel() == 0; }

private:
  /// Cached module namespaces.
  DenseMap<Operation *, hw::InnerSymbolNamespace> moduleNamespaces;

  DenseSet<Operation *> visitedModules;
  /// Map of a reference value to an entry into refSendPathList. Each entry in
  /// refSendPathList represents the path to RefSend.
  /// The path is required since there can be multiple paths to the RefSend and
  /// we need to identify a unique path.
  DenseMap<Value, size_t> dataflowAt;

  /// refSendPathList is used to construct a path to the RefSendOp. Each entry
  /// is an XMRNode, with an InnerRefAttr or indexing op, and a pointer to the
  /// next node in the path. The InnerRefAttr can be to an InstanceOp or to the
  /// XMR defining op, the index op records narrowing along path. All the nodes
  /// representing an InstanceOp or indexing operation must have a valid
  /// NextNodeOnPath. Only the node representing the final XMR defining op has
  /// no NextNodeOnPath, which denotes a leaf node on the path.
  SmallVector<XMRNode> refSendPathList;

  llvm::EquivalenceClasses<Value> *dataFlowClasses;
  // Instance and module ref ports that needs to be removed.
  DenseMap<Operation *, llvm::BitVector> refPortsToRemoveMap;

  /// RefResolve, RefSend, and Connects involving them that will be removed.
  SmallVector<Operation *> opsToRemove;

  /// Record the internal path to an external module or a memory.
  DenseMap<size_t, SmallString<128>> xmrPathSuffix;

  CircuitNamespace *circuitNamespace;

  /// Utility to create HerPathOps at a predefined location in the circuit.
  /// This handles caching and keeps the order consistent.
  hw::HierPathCache *hierPathCache;

  /// Per-module helpers for creating operations within modules.
  DenseMap<FModuleOp, ModuleState> moduleStates;
};
