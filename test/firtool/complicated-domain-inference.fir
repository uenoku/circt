; RUN: rm -rf %t && not firtool %s -domain-mode=infer-all --debug-domains-html=%t -o /dev/null 2>&1 | FileCheck %s
; RUN: FileCheck %s --input-file=%t/ComplicatedDomainInference.domain.html --check-prefix=HTML

; This is a deliberately complicated domain inference example.  Most of the
; private modules have no domain annotations on their interfaces.  The inference
; pass has to infer interface domain ports through a small hierarchy, through
; aggregate ports, and through a locally-created domain that is exported back to
; the parent.  The public top then intentionally connects one inferred local
; clock domain back into a value explicitly annotated with the fast clock domain.

FIRRTL version 7.0.0
circuit ComplicatedDomainInference :
  domain ClockDomain :
    name : String
    frequency : Integer

  domain PowerDomain :
    name : String
    voltage : Integer
    always_on : Bool

  module LeafStage :
    input sel : UInt<1>
    input lhs : UInt<8>
    input rhs : UInt<8>
    output out : UInt<8>

    wire selected : UInt<8>
    node mixed = mux(sel, lhs, rhs)
    connect selected, mixed
    connect out, selected

  module BundleStage :
    input in : { valid : UInt<1>, bits : UInt<8> }
    output out : { valid : UInt<1>, bits : UInt<8> }

    wire tmp : { valid : UInt<1>, bits : UInt<8> }
    connect tmp.valid, in.valid
    connect tmp.bits, in.bits
    connect out, tmp

  module LocalDomainProducer :
    input in : UInt<8>
    output out : UInt<8>

    domain localClock of ClockDomain(String("wrapper-local-clock"), Integer(750))

    wire localValue : UInt<8> domains [localClock]
    node recast = unsafe_domain_cast(in, localClock)
    connect localValue, recast
    connect out, localValue

  module Wrapper :
    input sel : UInt<1>
    input a : UInt<8>
    input b : UInt<8>
    output out : UInt<8>

    inst leaf of LeafStage
    connect leaf.sel, sel
    connect leaf.lhs, a
    connect leaf.rhs, b

    wire payload : { valid : UInt<1>, bits : UInt<8> }
    connect payload.valid, sel
    connect payload.bits, leaf.out

    inst bundle of BundleStage
    connect bundle.in, payload

    inst local of LocalDomainProducer
    connect local.in, bundle.out.bits
    connect out, local.out

  public module ComplicatedDomainInference :
    input fastClock : Domain of ClockDomain
    input slowClock : Domain of ClockDomain
    input corePower : Domain of PowerDomain

    input fastSel : UInt<1> domains [fastClock, corePower]
    input fastA : UInt<8> domains [fastClock, corePower]
    input fastB : UInt<8> domains [fastClock, corePower]

    input slowSel : UInt<1> domains [slowClock, corePower]
    input slowA : UInt<8> domains [slowClock, corePower]
    input slowB : UInt<8> domains [slowClock, corePower]

    output fastOut : UInt<8> domains [fastClock, corePower]
    output slowOut : UInt<8> domains [slowClock, corePower]

    inst fastPath of Wrapper
    connect fastPath.sel, fastSel
    connect fastPath.a, fastA
    connect fastPath.b, fastB

    inst slowPath of Wrapper
    connect slowPath.sel, slowSel
    connect slowPath.a, slowA
    connect slowPath.b, slowB

    ; Both wrapper outputs are in domains inferred from LocalDomainProducer's
    ; localClock, not the original fast/slow input clocks.  These connections
    ; are intentional errors that make the inferred domains visible in the
    ; diagnostic path.
    ; CHECK: error: illegal domain crossing
    connect fastOut, fastPath.out
    connect slowOut, slowPath.out

; HTML-DAG: FIRRTL Domain Inference Debugger
; HTML-DAG: renderLifetimeAnnotations
; HTML-DAG: Why This Failed
; HTML-DAG: Suggested Insertion Points
; HTML-DAG: Insert Domain Crossing Here
; HTML-DAG: Lifetime Annotations
; HTML-DAG: Constraint Graph
