//===- PowerDomainJSONHandler.cpp - Power Domain JSON from OM -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A handler for generating JSON format from PowerDomain information contained
// in a final MLIR blob (likely compiled with `firtool`).
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "Handler.h"

using namespace llvm;

namespace circt {
namespace handlers {

namespace options {
cl::OptionCategory powerCat{"Power Domain JSON Options"};
} // namespace options

/// A handler that generates Power Domain JSON output from PowerDomain
/// information.
class PowerDomainJSON : public Handler {

public:
  bool shouldHandle(Type type) override {
    auto classType = dyn_cast<om::ClassType>(type);
    return classType && classType.getClassName().getValue() == "PowerDomain";
  }

  LogicalResult handle(const ObjectMap &objectMap) override {
    // Process each PowerDomain object and collect its associations
    for (auto &[objectValue, associations] : objectMap) {
      auto name = cast<StringAttr>(cast<om::evaluator::AttributeValue>(
                                       objectValue->getField("name_out")->get())
                                       ->getAttr());

      // Collect the associations (paths) for this power domain
      SmallVector<StringAttr> domainAssociations;
      for (auto &association : associations) {
        if (auto *p = dyn_cast<om::evaluator::PathValue>(association.get())) {
          domainAssociations.push_back(p->getRef());
        } else {
          emitError(association->getLoc())
              << "expected associations to be a path, but got "
              << association->getType();
          return failure();
        }
      }

      // Store this power domain
      powerDomains.push_back({name, domainAssociations});
    }

    return success();
  }

  LogicalResult emit(raw_ostream &os) override {
    // Only emit if we have power domains to report
    if (powerDomains.empty())
      return success();

    json::OStream json(os, /*indentSize=*/2);
    json.object([&] {
      json.attributeArray("power_domains", [&] {
        for (auto &pd : powerDomains) {
          json.object([&] {
            json.attribute("name", pd.name.getValue());
            json.attributeArray("associations", [&] {
              for (auto assoc : pd.associations)
                json.value(assoc.getValue());
            });
          });
        }
      });
    });

    return success();
  }

  void clear() override { powerDomains.clear(); }

private:
  struct PowerDomainInfo {
    StringAttr name;
    SmallVector<StringAttr> associations;
  };

  SmallVector<PowerDomainInfo> powerDomains;
};

static bool registeredPowerDomainJSONHandler = [] {
  HandlerRegistry::get().registerHandler(std::make_unique<PowerDomainJSON>());
  return true;
}();

} // namespace handlers
} // namespace circt
