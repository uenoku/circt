// RUN: circt-translate --export-firrtl %s -o %t
// RUN: FileCheck %s < %t
// RUN: circt-translate --import-firrtl --mlir-print-debuginfo %t | \
// RUN:   circt-translate --export-firrtl | diff - %t

// CHECK: FIRRTL version 7.0.0
// CHECK: circuit ObjectCreate :
// CHECK:   class Class :
// CHECK-NEXT:     input str : String
// CHECK-NEXT:     output result : String
// CHECK-NEXT:     input count : Integer
// CHECK:     propassign result, str
// CHECK:   extclass External :
// CHECK-NEXT:     input str : String
// CHECK-NEXT:     output result : String
// CHECK-NEXT:     input count : Integer
// CHECK:   extclass RefHolder :
// CHECK-NEXT:     input ref : AnyRef
// CHECK:   module ObjectCreate :
// CHECK:     object obj of Class
// CHECK-NEXT:     propassign obj.str, str
// CHECK-NEXT:     propassign obj.count, count
// CHECK:     object external of External
// CHECK-NEXT:     propassign external.str, str
// CHECK-NEXT:     propassign external.count, count
// CHECK:     object holder of RefHolder
// CHECK-NEXT:     propassign holder.ref, obj
// CHECK:     propassign out, obj.result
firrtl.circuit "ObjectCreate" {
  firrtl.class @Class(
    in %str: !firrtl.string,
    out %result: !firrtl.string,
    in %count: !firrtl.integer
  ) {
    firrtl.propassign %result, %str : !firrtl.string
  }
  firrtl.extclass @External(
    in str: !firrtl.string,
    out result: !firrtl.string,
    in count: !firrtl.integer
  )
  firrtl.extclass @RefHolder(in ref: !firrtl.anyref)

  firrtl.module @ObjectCreate(
    in %str: !firrtl.string,
    in %count: !firrtl.integer,
    out %out: !firrtl.string
  ) {
    %obj = firrtl.object.create(%str, %count) :
      (!firrtl.string, !firrtl.integer) ->
      !firrtl.class<@Class(in str: !firrtl.string, out result: !firrtl.string, in count: !firrtl.integer)>
    %external = firrtl.object.create(%str, %count) :
      (!firrtl.string, !firrtl.integer) ->
      !firrtl.class<@External(in str: !firrtl.string, out result: !firrtl.string, in count: !firrtl.integer)>
    %objRef = firrtl.object.anyref_cast %obj :
      !firrtl.class<@Class(in str: !firrtl.string, out result: !firrtl.string, in count: !firrtl.integer)>
    %holder = firrtl.object.create(%objRef) :
      (!firrtl.anyref) -> !firrtl.class<@RefHolder(in ref: !firrtl.anyref)>
    %result = firrtl.object.subfield %obj[result] :
      !firrtl.class<@Class(in str: !firrtl.string, out result: !firrtl.string, in count: !firrtl.integer)>
    firrtl.propassign %out, %result : !firrtl.string
  }
}
