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
      PowerDomainInfo info;

      // Extract name (required field)
      info.name = cast<StringAttr>(cast<om::evaluator::AttributeValue>(
                                       objectValue->getField("name_out")->get())
                                       ->getAttr());

      // Extract optional fields if they exist
      {
        auto sourceField = objectValue->getField("source_out");
        if (sourceField != std::nullopt) {
          if (auto *attrVal =
                  dyn_cast<om::evaluator::AttributeValue>(sourceField->get())) {
            info.source = cast<StringAttr>(attrVal->getAttr());
          }
        }
      }

      {
        auto relationshipField = objectValue->getField("relationship_out");
        if (relationshipField != std::nullopt) {
          if (auto *attrVal = dyn_cast<om::evaluator::AttributeValue>(
                  relationshipField->get())) {
            info.relationship = cast<StringAttr>(attrVal->getAttr());
          }
        }
      }

      {
        auto clampValueField = objectValue->getField("clampValue_out");
        if (clampValueField != std::nullopt) {
          if (auto *attrVal = dyn_cast<om::evaluator::AttributeValue>(
                  clampValueField->get())) {
            if (auto intAttr = dyn_cast<om::IntegerAttr>(attrVal->getAttr())) {
              info.clampValue = intAttr.getValue().getValue();
            }
          }
        }
      }

      {
        auto destinationField = objectValue->getField("destination_out");
        if (destinationField != std::nullopt) {
          if (auto *attrVal = dyn_cast<om::evaluator::AttributeValue>(
                  destinationField->get())) {
            info.destination = cast<StringAttr>(attrVal->getAttr());
          }
        }
      }

      // Collect the associations (paths) for this power domain
      for (auto &association : associations) {
        if (auto *p = dyn_cast<om::evaluator::PathValue>(association.get())) {
          info.associations.push_back(p->getRef());
        } else {
          emitError(association->getLoc())
              << "expected associations to be a path, but got "
              << association->getType();
          return failure();
        }
      }

      // Store this power domain
      powerDomains.push_back(info);
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

            // Emit optional fields if present
            if (pd.source)
              json.attribute("source", pd.source.getValue());
            if (pd.relationship)
              json.attribute("relationship", pd.relationship.getValue());
            if (pd.clampValue.has_value())
              json.attribute("clamp_value", pd.clampValue->getSExtValue());
            if (pd.destination)
              json.attribute("destination", pd.destination.getValue());

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
    StringAttr source;
    StringAttr relationship;
    std::optional<APInt> clampValue;
    StringAttr destination;
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
