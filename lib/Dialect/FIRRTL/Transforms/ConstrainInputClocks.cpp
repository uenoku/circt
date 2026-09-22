//===- ConstrainInputClocks.cpp - Add input clock domains -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace circt;
using namespace circt::firrtl;

namespace circt::firrtl {
#define GEN_PASS_DEF_CONSTRAININPUTCLOCKS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace circt::firrtl

namespace {
struct ModuleUpdate {
  SmallVector<std::pair<unsigned, PortInfo>> insertions;
  ArrayAttr domainInfo;
};

struct ConstrainInputClocksPass
    : public circt::firrtl::impl::ConstrainInputClocksBase<
          ConstrainInputClocksPass> {
  using Base::Base;

  void runOnOperation() override {
    CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);
    auto circuit = getOperation();
    auto *context = &getContext();

    // There is deliberately one domain *type* for all clocks.  Each clock
    // port gets its own value of that type below, which is what makes the
    // clocks distinct domains.
    DomainOp clockDomain;
    circuit.getBodyBlock()->walk([&](DomainOp op) {
      if (op.getName() == "ClockDomain")
        clockDomain = op;
    });
    if (!clockDomain) {
      OpBuilder builder(circuit.getBodyBlock(),
                        circuit.getBodyBlock()->begin());
      clockDomain = DomainOp::create(builder, circuit.getLoc(),
                                     StringAttr::get(context, "ClockDomain"),
                                     StringAttr(), ArrayAttr::get(context, {}));
    }
    auto clockType = DomainType::getFromDomainOp(clockDomain);

    DenseMap<StringAttr, ModuleUpdate> updates;
    SmallVector<FModuleLike> modules;
    circuit.getBodyBlock()->walk([&](FModuleLike module) {
      // Only public module interfaces are part of the externally visible
      // contract. Private modules are handled by InferDomains' normal
      // interface inference and must not be changed here.
      if (!module.isPublic())
        return;

      modules.push_back(module);
      ModuleUpdate update;
      SmallVector<Attribute> info;
      auto oldInfo = module.getDomainInfoAttr();
      for (unsigned i = 0, e = module.getNumPorts(); i < e; ++i) {
        if (!oldInfo.empty())
          info.push_back(oldInfo[i]);
        else
          info.push_back(ArrayAttr::get(context, {}));
      }

      unsigned oldNumPorts = module.getNumPorts();
      for (unsigned i = 0; i < oldNumPorts; ++i) {
        if (module.getPortDirection(i) != Direction::In ||
            !isa<ClockType>(module.getPortType(i)))
          continue;

        // Preserve an existing ClockDomain association. A clock may also carry
        // associations for other domain kinds, so inspect the associated port
        // type rather than treating any association as ClockDomain.
        bool hasClockDomain = false;
        if (auto associations = dyn_cast<ArrayAttr>(info[i])) {
          for (auto attr : associations) {
            unsigned domainPort = cast<IntegerAttr>(attr).getUInt();
            if (module.getPortType(domainPort) != clockType)
              continue;
            hasClockDomain = true;
          }
        }
        if (hasClockDomain)
          continue;

        unsigned domainPort = oldNumPorts + update.insertions.size();
        auto name = StringAttr::get(
            context, (module.getPortName(i) + "_clock_domain").str());
        update.insertions.push_back(
            {oldNumPorts, PortInfo(name, clockType, Direction::In, StringAttr(),
                                   module.getPortLocation(i), std::nullopt,
                                   ArrayAttr::get(context, {}))});
        info[i] = ArrayAttr::get(
            context,
            {IntegerAttr::get(
                IntegerType::get(context, 64,
                                 IntegerType::SignednessSemantics::Unsigned),
                domainPort)});
        info.push_back(ArrayAttr::get(context, {}));
      }

      if (!update.insertions.empty()) {
        update.domainInfo = ArrayAttr::get(context, info);
        updates.try_emplace(module.getModuleNameAttr(), std::move(update));
      }
    });

    // Update definitions first, then update every instance result list to
    // exactly match its new target signature.  clone...ReplaceUses is
    // important: instance results may already be used by the module body.
    for (auto module : modules) {
      auto it = updates.find(module.getModuleNameAttr());
      if (it == updates.end())
        continue;
      module.insertPorts(it->second.insertions);
      module.setDomainInfoAttr(it->second.domainInfo);
    }

    circuit.walk([&](FInstanceLike instance) {
      auto names = instance.getReferencedModuleNamesAttr();
      if (names.empty())
        return;
      auto it = updates.find(cast<StringAttr>(names[0]));
      if (it == updates.end())
        return;
      auto clone =
          instance.cloneWithInsertedPortsAndReplaceUses(it->second.insertions);
      clone.setDomainInfoAttr(it->second.domainInfo);
      instance->erase();
    });

    // The existing clock connection does not automatically connect the newly
    // inserted domain result. Mirror every instance clock connection onto its
    // corresponding domain port connection.
    circuit.walk([&](FModuleOp module) {
      auto moduleInfo = module.getDomainInfoAttr();
      for (auto instance : module.getBodyBlock()->getOps<FInstanceLike>()) {
        auto names = instance.getReferencedModuleNamesAttr();
        if (names.empty())
          continue;
        auto update = updates.find(cast<StringAttr>(names[0]));
        if (update == updates.end())
          continue;
        unsigned firstInserted =
            instance.getNumPorts() - update->second.insertions.size();
        for (unsigned i = 0, e = firstInserted; i < e; ++i) {
          if (instance.getPortDirection(i) != Direction::In ||
              !isa<ClockType>(instance->getResult(i).getType()))
            continue;
          auto clockDomainInfo = dyn_cast<ArrayAttr>(instance.getPortDomain(i));
          if (!clockDomainInfo || clockDomainInfo.empty())
            continue;
          unsigned domainResult =
              cast<IntegerAttr>(clockDomainInfo[0]).getUInt();
          for (auto *user : instance->getResult(i).getUsers()) {
            auto connect = dyn_cast<FConnectLike>(user);
            if (!connect || connect.getDest() != instance->getResult(i))
              continue;
            auto source = dyn_cast<BlockArgument>(connect.getSrc());
            if (!source)
              continue;
            auto sourceInfo =
                dyn_cast<ArrayAttr>(moduleInfo[source.getArgNumber()]);
            if (!sourceInfo || sourceInfo.empty())
              continue;
            unsigned sourceDomain = cast<IntegerAttr>(sourceInfo[0]).getUInt();
            auto builder = OpBuilder::atBlockEnd(module.getBodyBlock());
            DomainDefineOp::create(builder, connect.getLoc(),
                                   instance->getResult(domainResult),
                                   module.getArgument(sourceDomain));
          }
        }
      }
    });
  }
};
} // namespace
