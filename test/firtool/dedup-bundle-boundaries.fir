; RUN: firtool %s --ir-hw -o /dev/null
; RUN: firtool %s --ir-hw --dedup-classes=0 -o /dev/null

FIRRTL version 4.0.0
circuit BundleBoundaries :
  ; These bundle types have the same preorder sequence of types but different
  ; nesting. Deduplicating these modules would crash when fixing up their ports.
  module A :
    input p : { a : { b : UInt<1> }, c : UInt<1>, d : { e : UInt<1> } }
  module B :
    input p : { a : { b : UInt<1>, c : UInt<1> }, d : { e : UInt<1> } }
  public module BundleBoundaries :
    inst a of A
    inst b of B
    invalidate a.p
    invalidate b.p
