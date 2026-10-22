// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects,firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects),firrtl-lower-classes))' | FileCheck %s --check-prefix=OM

firrtl.circuit "Partial" {
  firrtl.module @Partial() {}
  firrtl.class @Int(in %input: !firrtl.integer, out %output: !firrtl.integer) {
    firrtl.propassign %output, %input : !firrtl.integer
  }

  // Preserve the order of constants, including one already at block entry.
  // CHECK-LABEL: firrtl.class @Hoisting
  // OM-LABEL: om.class @Hoisting
  firrtl.class @Hoisting() {
    // CHECK-NEXT: %[[BEFORE:.*]] = firrtl.integer 1
    // CHECK-NEXT: %[[AFTER:.*]] = firrtl.integer 2
    // CHECK-NEXT: %[[LAST:.*]] = firrtl.integer 3
    // CHECK-NEXT: %obj = firrtl.object.create(%[[AFTER]])
    // CHECK-NEXT: }
    %before = firrtl.integer 1
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %after = firrtl.integer 2
    %last = firrtl.integer 3
    firrtl.propassign %in, %after : !firrtl.integer
  }

  // A constant-only object can be lowered while its dependent object stays in
  // declaration form. Do not reorder the dependent object's output accesses.
  // CHECK-LABEL: firrtl.class @Mixed
  // OM-LABEL: om.class @Mixed
  firrtl.class @Mixed(out %out: !firrtl.integer) {
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 4
    // CHECK-NEXT: %a = firrtl.object @Int
    %a = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    // CHECK-NEXT: %[[AIN:.*]] = firrtl.object.subfield %a[input]
    %a_in = firrtl.object.subfield %a[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[AOUT:.*]] = firrtl.object.subfield %a[output]
    %a_out = firrtl.object.subfield %a[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %out, %[[AOUT]] : !firrtl.integer
    firrtl.propassign %out, %a_out : !firrtl.integer
    // CHECK-NEXT: %b = firrtl.object.create(%[[C]])
    %b = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %b_in = firrtl.object.subfield %b[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[BOUT:.*]] = firrtl.object.subfield %b[output]
    %b_out = firrtl.object.subfield %b[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %constant = firrtl.integer 4
    // CHECK-NEXT: firrtl.propassign %[[AIN]], %[[BOUT]] : !firrtl.integer
    // CHECK-NEXT: }
    firrtl.propassign %a_in, %b_out : !firrtl.integer
    firrtl.propassign %b_in, %constant : !firrtl.integer
  }

  // Even an expression computed only from constants is not constant-like.
  // CHECK-LABEL: firrtl.module @Expression
  // OM-LABEL: om.class @Expression_Class
  firrtl.module @Expression() {
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 5
    // CHECK-NEXT: %[[WIRE:.*]] = firrtl.wire : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %constant = firrtl.integer 5
    // CHECK-NEXT: %[[SUM:.*]] = firrtl.integer.add %[[C]], %[[C]]
    %sum = firrtl.integer.add %constant, %constant : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    // CHECK-NEXT: %obj = firrtl.object.create(%[[SUM]])
    // CHECK-NEXT: firrtl.propassign %[[WIRE]], %obj : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: }
    firrtl.propassign %in, %sum : !firrtl.integer
  }

  // Cyclic inputs use wires in modules, but remain untouched in classes.
  // CHECK-LABEL: firrtl.module @SelfModule
  // OM-LABEL: om.class @SelfModule_Class
  firrtl.module @SelfModule() {
    // CHECK-NEXT: %[[WIRE:.*]] = firrtl.wire : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[OUT:.*]] = firrtl.object.subfield %[[WIRE]][output]
    %out = firrtl.object.subfield %obj[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %obj = firrtl.object.create(%[[OUT]])
    // CHECK-NEXT: firrtl.propassign %[[WIRE]], %obj : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: }
    firrtl.propassign %in, %out : !firrtl.integer
  }
  // CHECK-LABEL: firrtl.class @SelfClass
  // OM-LABEL: om.class @SelfClass
  firrtl.class @SelfClass() {
    // CHECK-NEXT: %obj = firrtl.object @Int
    %obj = firrtl.object @Int(in input: !firrtl.integer, out output: !firrtl.integer)
    // CHECK-NEXT: %[[IN:.*]] = firrtl.object.subfield %obj[input]
    %in = firrtl.object.subfield %obj[input] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: %[[OUT:.*]] = firrtl.object.subfield %obj[output]
    %out = firrtl.object.subfield %obj[output] : !firrtl.class<@Int(in input: !firrtl.integer, out output: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %[[IN]], %[[OUT]] : !firrtl.integer
    // CHECK-NEXT: }
    firrtl.propassign %in, %out : !firrtl.integer
  }
}
