; RUN: firtool %s -output-final-mlir - -o /dev/null                    | FileCheck %s -implicit-check-not om.frozenpath_create -implicit-check-not Domain
; RUN: firtool %s -output-final-mlir - -domain-mode=check -o /dev/null | FileCheck %s -check-prefixes=CHECK,DOMAIN
; RUN: firtool %s -output-final-mlir - -domain-mode=strip -o /dev/null | FileCheck %s -implicit-check-not om.frozenpath_create -implicit-check-not Domain

FIRRTL version 7.0.0
circuit Foo :
  domain ClockDomain :

  domain PowerDomain :

  module gate_inv:
    input A : Domain of ClockDomain
    output B : Domain of ClockDomain
    input in: Clock domains [A]
    output out: Clock domains [B]

    connect unsafe_domain_cast(out) 

  public module Foo :
    input A : Domain of ClockDomain
    input B : Domain of PowerDomain
    input a : UInt<1>
    input b : UInt<1>
