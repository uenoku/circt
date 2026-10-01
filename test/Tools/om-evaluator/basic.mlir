// RUN: om-evaluator %s --class Top --input-json %S/Inputs/parameters.json --output-field-path intrinsicProperties | FileCheck %s --check-prefix=FIELD
// RUN: om-evaluator %s --class Top --input-json %S/Inputs/parameters.json --output-field-path intrinsicProperties.subsystem_id | FileCheck %s --check-prefix=SCALAR
// RUN: om-evaluator %s --class Top --input-json %S/Inputs/parameters.json | FileCheck %s --check-prefix=ALL
// RUN: not om-evaluator %s --class Top --input-json %S/Inputs/missing.json 2>&1 | FileCheck %s --check-prefix=MISSING

// FIELD: "subsystem_id": 1
// SCALAR: 1
// ALL: "intrinsicProperties": {
// ALL: "subsystem_id": 1
// MISSING: subsystem_id: !om.integer is missing

om.class @Properties(%subsystem_id: !om.integer) -> (subsystem_id: !om.integer) {
  om.class.fields %subsystem_id : !om.integer
}

om.class @Top(%basepath: !om.frozenbasepath, %subsystem_id: !om.integer) -> (intrinsicProperties: !om.class.type<@Properties>) {
  %properties = om.object @Properties(%subsystem_id) : (!om.integer) -> !om.class.type<@Properties>
  om.class.fields %properties : !om.class.type<@Properties>
}
