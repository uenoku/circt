//===- om-evaluator.cpp - Evaluate an OM class from the command line ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

#include <limits>

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::OptionCategory category("om-evaluator Options");
static cl::opt<std::string> inputFile(cl::Positional,
                                      cl::desc("<input MLIR file>"),
                                      cl::init("-"), cl::cat(category));
static cl::opt<std::string> className("class", cl::Required,
                                      cl::desc("OM class to instantiate"),
                                      cl::cat(category));
static cl::opt<std::string>
    inputJSON("input-json",
              cl::desc("JSON object of named parameters; omit to print the "
                       "input schema"),
              cl::cat(category));
static cl::opt<std::string>
    outputFieldPath("output-field-path",
                    cl::desc("Dot-separated path to an output field"),
                    cl::cat(category));

static LogicalResult error(const Twine &message) {
  WithColor::error(errs(), "om-evaluator") << message << '\n';
  return failure();
}

/// Describe the JSON values accepted by parseParameter for an OM type.
static json::Object parameterSchema(Type type) {
  std::string typeName;
  raw_string_ostream(typeName) << type;
  json::Object schema;
  schema["x-om-type"] = typeName;
  if (isa<om::FrozenBasePathType>(type)) {
    schema["type"] = "string";
    schema["const"] = "";
  } else if (isa<om::OMIntegerType>(type)) {
    schema["type"] = "integer";
    schema["minimum"] = std::numeric_limits<int64_t>::min();
    schema["maximum"] = std::numeric_limits<int64_t>::max();
  } else if (isa<om::StringType>(type)) {
    schema["type"] = "string";
  } else if (type.isInteger(1)) {
    schema["type"] = "boolean";
  } else if (auto listType = dyn_cast<om::ListType>(type)) {
    schema["type"] = "array";
    schema["items"] = parameterSchema(listType.getElementType());
  } else {
    schema["not"] = json::Object();
    schema["description"] = "This OM type is not supported by --input-json";
  }
  return schema;
}

static void printInputSchema(om::ClassOp cls) {
  json::Object properties;
  json::Array required;
  for (auto [name, arg] :
       llvm::zip(cls.getFormalParamNames().getAsRange<StringAttr>(),
                 cls.getBodyBlock()->getArguments())) {
    properties[name.getValue()] = parameterSchema(arg.getType());
    required.push_back(name.getValue());
  }
  json::Object schema;
  schema["$schema"] = "https://json-schema.org/draft/2020-12/schema";
  schema["type"] = "object";
  schema["properties"] = std::move(properties);
  schema["required"] = std::move(required);
  schema["additionalProperties"] = false;
  outs() << formatv("{0:2}", json::Value(std::move(schema))) << '\n';
}

static FailureOr<om::EvaluatorValuePtr>
parseParameter(const json::Value &value, Type type, StringRef name) {
  auto *context = type.getContext();
  if (isa<om::FrozenBasePathType>(type)) {
    if (value.getAsString() == StringRef(""))
      return om::EvaluatorValuePtr(
          std::make_shared<om::evaluator::BasePathValue>(context));
  } else if (isa<om::OMIntegerType>(type)) {
    if (auto integer = value.getAsInteger()) {
      auto attr = om::IntegerAttr::get(
          context, IntegerAttr::get(IntegerType::get(context, 64), *integer));
      return om::evaluator::AttributeValue::get(attr);
    }
  } else if (isa<om::StringType>(type)) {
    if (auto string = value.getAsString())
      return om::evaluator::AttributeValue::get(StringAttr::get(*string, type));
  } else if (type.isInteger(1)) {
    if (auto boolean = value.getAsBoolean())
      return om::evaluator::AttributeValue::get(
          BoolAttr::get(context, *boolean));
  } else if (auto listType = dyn_cast<om::ListType>(type)) {
    if (auto *array = value.getAsArray()) {
      SmallVector<om::EvaluatorValuePtr> elements;
      for (const auto &element : *array) {
        auto parsed = parseParameter(element, listType.getElementType(), name);
        if (failed(parsed))
          return failure();
        elements.push_back(*parsed);
      }
      return om::EvaluatorValuePtr(std::make_shared<om::evaluator::ListValue>(
          listType, std::move(elements), UnknownLoc::get(context)));
    }
  }
  std::string typeName;
  raw_string_ostream(typeName) << type;
  (void)error(Twine("unsupported value for ") + name + " (expected " +
              typeName + ")");
  return failure();
}

static FailureOr<json::Value>
toJSON(const om::EvaluatorValuePtr &value,
       SmallPtrSetImpl<const om::evaluator::EvaluatorValue *> &active) {
  if (!value || value->isUnknown())
    return json::Value(nullptr);
  if (!active.insert(value.get()).second) {
    (void)error(
        "output contains a cycle; select a field with --output-field-path");
    return failure();
  }
  auto remove = llvm::scope_exit([&] { active.erase(value.get()); });
  if (auto *object = dyn_cast<om::evaluator::ObjectValue>(value.get())) {
    json::Object result;
    for (auto field : object->getFieldNames().getAsRange<StringAttr>()) {
      auto fieldValue = object->getField(field);
      if (failed(fieldValue))
        return failure();
      auto converted = toJSON(*fieldValue, active);
      if (failed(converted))
        return failure();
      result[field.getValue()] = std::move(*converted);
    }
    return json::Value(std::move(result));
  }
  if (auto *list = dyn_cast<om::evaluator::ListValue>(value.get())) {
    json::Array result;
    for (const auto &element : list->getElements()) {
      auto converted = toJSON(element, active);
      if (failed(converted))
        return failure();
      result.push_back(std::move(*converted));
    }
    return json::Value(std::move(result));
  }
  if (auto *basepath = dyn_cast<om::evaluator::BasePathValue>(value.get())) {
    std::string path;
    raw_string_ostream os(path);
    if (basepath->getPath().getPath().empty())
      return json::Value("");
    Attribute(basepath->getPath()).print(os);
    return json::Value(os.str());
  }
  if (auto *path = dyn_cast<om::evaluator::PathValue>(value.get()))
    return json::Value(path->getAsString().getValue());
  if (auto *attribute = dyn_cast<om::evaluator::AttributeValue>(value.get())) {
    Attribute attr = attribute->getAttr();
    if (auto integer = dyn_cast<om::IntegerAttr>(attr)) {
      APInt number = integer.getValue().getValue();
      if (!number.isSignedIntN(64)) {
        (void)error("integer output exceeds the JSON 64-bit range");
        return failure();
      }
      return json::Value(number.getSExtValue());
    }
    if (auto boolean = dyn_cast<BoolAttr>(attr))
      return json::Value(boolean.getValue());
    if (auto string = dyn_cast<StringAttr>(attr))
      return json::Value(string.getValue());
  }
  (void)error("output contains an unsupported value type");
  return failure();
}

static LogicalResult run(MLIRContext &context) {
  SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler handler(sourceMgr, &context);
  auto module = parseSourceFile<ModuleOp>(inputFile, sourceMgr, &context);
  if (!module)
    return failure();

  auto cls = SymbolTable(module.get())
                 .lookup<om::ClassOp>(StringAttr::get(&context, className));
  if (!cls)
    return error(Twine("unknown OM class: ") + className);

  if (inputJSON.empty()) {
    printInputSchema(cls);
    return success();
  }

  auto buffer = MemoryBuffer::getFile(inputJSON);
  if (!buffer)
    return error(Twine("cannot read ") + inputJSON + ": " +
                 buffer.getError().message());
  auto parsed = json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return error(Twine("invalid JSON: ") + toString(parsed.takeError()));
  auto *object = parsed->getAsObject();
  if (!object)
    return error("input JSON must be an object");
  json::Object parameters = std::move(*object);

  SmallVector<om::EvaluatorValuePtr> actualParams;
  for (auto [name, arg] :
       llvm::zip(cls.getFormalParamNames().getAsRange<StringAttr>(),
                 cls.getBodyBlock()->getArguments())) {
    auto *value = parameters.get(name.getValue());
    if (!value) {
      std::string typeName;
      raw_string_ostream(typeName) << arg.getType();
      return error(Twine(name.getValue()) + ": " + typeName + " is missing");
    }
    auto converted = parseParameter(*value, arg.getType(), name.getValue());
    if (failed(converted))
      return failure();
    actualParams.push_back(*converted);
    parameters.erase(name.getValue());
  }
  if (!parameters.empty())
    return error(Twine("unknown parameter: ") +
                 StringRef(parameters.begin()->first));

  om::Evaluator evaluator(module.get());
  auto evaluated =
      evaluator.instantiate(StringAttr::get(&context, className), actualParams);
  if (failed(evaluated))
    return failure();

  om::EvaluatorValuePtr selected = *evaluated;
  StringRef remaining(outputFieldPath);
  while (!remaining.empty()) {
    auto [field, rest] = remaining.split('.');
    auto *object = dyn_cast<om::evaluator::ObjectValue>(selected.get());
    if (!object)
      return error(Twine("cannot select field ") + field +
                   " from a non-object");
    auto next = object->getField(field);
    if (failed(next))
      return failure();
    selected = *next;
    remaining = rest;
  }
  SmallPtrSet<const om::evaluator::EvaluatorValue *, 16> active;
  auto result = toJSON(selected, active);
  if (failed(result))
    return failure();
  outs() << formatv("{0:2}", *result) << '\n';
  return success();
}

int main(int argc, char **argv) {
  InitLLVM init(argc, argv);
  setBugReportMsg(circtBugReportMsg);
  cl::HideUnrelatedOptions(category);
  cl::ParseCommandLineOptions(argc, argv, "Evaluate an OM class\n");
  DialectRegistry registry;
  circt::registerAllDialects(registry);
  MLIRContext context(registry);
  return failed(run(context));
}
