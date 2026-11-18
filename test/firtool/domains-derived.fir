; RUN: firtool %s --domain-mode=infer-all --strip-domain=false --output-final-mlir=%t.mlir -o %t.sv
; RUN: FileCheck %s --check-prefix=SV < %t.sv
; RUN: FileCheck %s --check-prefix=OM < %t.mlir

; Derive a domain from an inferred instance input domain. Reading the input's
; name must use the assigned domain, not a field of the lowered instance.
; See https://github.com/llvm/circt/issues/11601.
FIRRTL version 7.0.0
circuit Top :
  domain ClockDomain :
    name : String
    source : String
    relationship : String

  extmodule Divider :
    input in : Clock domains [A]
    output out : Clock domains [B]
    input A : Domain of ClockDomain
    input B : Domain of ClockDomain
    defname = Divider

  module Wrapper :
    input in : Clock
    output out : Clock
    inst divider of Divider
    wire name : String
    propassign name, string_concat(divider.A.name, String("_div2"))
    domain divided of ClockDomain(name, divider.A.name, String("synchronous"))
    domain_define divider.B = divided
    connect divider.in, in
    connect out, divider.out

  public module Top :
    input clock : Clock
    output out : Clock
    inst wrapper of Wrapper
    connect wrapper.in, clock
    connect out, wrapper.out

; SV-LABEL: module Wrapper(
; SV:         input in,
; SV-NEXT:    output out
; SV:       Divider divider (
; SV-NEXT:    .in (in),
; SV-NEXT:    .out (out)
; SV-LABEL: module Top(
; SV:         input clock,
; SV-NEXT:    output out
; SV:       Wrapper wrapper (
; SV-NEXT:    .in (clock),
; SV-NEXT:    .out (out)

; OM-LABEL: om.class private @Wrapper_Class(
; OM-SAME: %[[DOMAIN:[^: ]+]]: !om.class.type<@ClockDomain>
; OM: %[[RELATIONSHIP:.*]] = om.constant "synchronous" : !om.string
; OM: %[[SUFFIX:.*]] = om.constant "_div2" : !om.string
; OM: om.object @Divider_Class(%{{[^ ,)]+}}, %[[DOMAIN]], %[[DIVIDED:[^ ,)]+]])
; OM: %[[NAME:.*]] = om.object.field %[[DOMAIN]]["name_out"]
; OM: %[[DIVIDED_NAME:.*]] = om.string.concat %[[NAME]], %[[SUFFIX]] : !om.string
; OM: %[[DIVIDED]] = om.object @ClockDomain(%basepath, %[[DIVIDED_NAME]], %[[NAME]], %[[RELATIONSHIP]])

; OM-LABEL: om.class @Top_Class(
; OM-SAME: %[[DOMAIN:[^: ]+]]: !om.class.type<@ClockDomain>
; OM: %[[SUFFIX:.*]] = om.constant "_div2" : !om.string
; OM: %[[RELATIONSHIP:.*]] = om.constant "synchronous" : !om.string
; OM: om.object @Divider_Class(%{{[^ ,)]+}}, %[[DOMAIN]], %[[DIVIDED:[^ ,)]+]])
; OM: %[[NAME:.*]] = om.object.field %[[DOMAIN]]["name_out"]
; OM: %[[DIVIDED_NAME:.*]] = om.string.concat %[[NAME]], %[[SUFFIX]] : !om.string
; OM: %[[DIVIDED]] = om.elaborated_object @ClockDomain(%[[DIVIDED_NAME]], %[[NAME]], %[[RELATIONSHIP]])
