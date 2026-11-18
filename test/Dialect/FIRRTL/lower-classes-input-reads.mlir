// RUN: circt-opt -firrtl-lower-classes %s | FileCheck %s

// Input property ports can be read in expressions. Replace these reads with
// the assigned values, including expressions used as actual parameters.
// See https://github.com/llvm/circt/issues/11601.
firrtl.circuit "InputReads" {
  firrtl.extclass @Metadata(out name: !firrtl.string)
  firrtl.extmodule private @ExtSink(in a: !firrtl.string, in b: !firrtl.string, out result: !firrtl.string)
  firrtl.extmodule private @ObjectSink(in obj: !firrtl.class<@Metadata(out name: !firrtl.string)>)
  firrtl.module private @Sink(in %a: !firrtl.string, in %b: !firrtl.string, out %result: !firrtl.string) {
    firrtl.propassign %result, %b : !firrtl.string
  }

  firrtl.module @InputReads(in %src: !firrtl.string, in %obj: !firrtl.class<@Metadata(out name: !firrtl.string)>, out %reads: !firrtl.list<string>) {
    %suffix = firrtl.string "_suffix"

    // Read a later input on the same instance in an actual parameter.
    %forward.a, %forward.b, %forward.result = firrtl.instance forward @ExtSink(in a: !firrtl.string, in b: !firrtl.string, out result: !firrtl.string)
    %forwardValue = firrtl.string.concat %forward.b, %suffix : !firrtl.string
    firrtl.propassign %forward.a, %forwardValue : !firrtl.string
    firrtl.propassign %forward.b, %src : !firrtl.string

    // Read an earlier input and another instance's input port. Internal
    // modules must be handled as well.
    %backward.a, %backward.b, %backward.result = firrtl.instance backward @Sink(in a: !firrtl.string, in b: !firrtl.string, out result: !firrtl.string)
    %backwardA = firrtl.string.concat %forward.a, %suffix : !firrtl.string
    %backwardB = firrtl.string.concat %backward.a, %suffix : !firrtl.string
    firrtl.propassign %backward.a, %backwardA : !firrtl.string
    firrtl.propassign %backward.b, %backwardB : !firrtl.string

    // Access a field of a class-typed input property, as in a derived domain.
    %object.obj = firrtl.instance object @ObjectSink(in obj: !firrtl.class<@Metadata(out name: !firrtl.string)>)
    firrtl.propassign %object.obj, %obj : !firrtl.class<@Metadata(out name: !firrtl.string)>
    %name = firrtl.object.subfield %object.obj[name] : !firrtl.class<@Metadata(out name: !firrtl.string)>

    %values = firrtl.list.create %forward.a, %forward.b, %backward.a, %backward.b, %forward.result, %backward.result, %name : !firrtl.list<string>
    firrtl.propassign %reads, %values : !firrtl.list<string>
  }

  // CHECK-LABEL: om.class @InputReads_Class(
  // CHECK-SAME: %[[SRC:[^: ]+]]: !om.string, %[[OBJ:[^: ]+]]: !om.class.type<@Metadata>
  // CHECK: %[[SUFFIX:.*]] = om.constant "_suffix" : !om.string
  // CHECK: %[[FORWARD:.*]] = om.object @ExtSink_Class(%{{[^ ,)]+}}, %[[FORWARD_ARG:[^ ,)]+]], %[[SRC]])
  // CHECK: %[[FORWARD_RESULT:.*]] = om.object.field %[[FORWARD]]["result"]
  // CHECK: %[[FORWARD_ARG]] = om.string.concat %[[SRC]], %[[SUFFIX]] : !om.string
  // CHECK: %[[BACKWARD:.*]] = om.object @Sink_Class(%{{[^ ,)]+}}, %[[BACKWARD_A:[^ ,)]+]], %[[BACKWARD_B:[^ ,)]+]])
  // CHECK: %[[BACKWARD_RESULT:.*]] = om.object.field %[[BACKWARD]]["result"]
  // CHECK: %[[BACKWARD_A]] = om.string.concat %[[FORWARD_ARG]], %[[SUFFIX]] : !om.string
  // CHECK: %[[BACKWARD_B]] = om.string.concat %[[BACKWARD_A]], %[[SUFFIX]] : !om.string
  // CHECK: om.object @ObjectSink_Class(%{{[^ ,)]+}}, %[[OBJ]])
  // CHECK: %[[NAME:.*]] = om.object.field %[[OBJ]]["name"]
  // CHECK: %[[VALUES:.*]] = om.list_create %[[FORWARD_ARG]], %[[SRC]], %[[BACKWARD_A]], %[[BACKWARD_B]], %[[FORWARD_RESULT]], %[[BACKWARD_RESULT]], %[[NAME]] : !om.string
  // CHECK: om.class.fields %[[VALUES]] : !om.list<!om.string>
}
