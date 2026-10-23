// Check explicit verification pass.
// RUN: circt-opt -hw-verify-irn -verify-diagnostics -split-input-file %s
// RUN: circt-opt -hw-verify-irn -verify-diagnostics -split-input-file -mlir-disable-threading %s
// Check verification occurs in firtool pipeline.
// RUN: firtool -verify-diagnostics -split-input-file %s

// #3526
hw.module @B() {}

hw.module @A() {
  // expected-note @below {{see existing inner symbol definition here}}
  hw.instance "h" sym @A @B() -> ()
  // expected-error @below {{redefinition of inner symbol named 'A'}}
  hw.instance "h" sym @A @B() -> ()
}

// -----

// expected-error @below {{operation with symbol: #hw.innerNameRef<@A::@invalid> was not found}}
hw.hierpath private @test [@A::@invalid]

hw.module @A() {
}

// -----

// expected-error @below {{inner symbol reference #hw.innerNameRef<@VerbatimInnerRef::@nonexistent> could not be found}}
sv.verbatim "// {{0}}" {symbols = [#hw.innerNameRef<@VerbatimInnerRef::@nonexistent>]}
hw.module @VerbatimInnerRef() {
  hw.output
}

// -----

// expected-error @below {{inner symbol reference #hw.innerNameRef<@VerbatimExprInnerRef::@nonexistent> could not be found}}
%0 = sv.verbatim.expr "MACRO" : () -> i32 {symbols = [#hw.innerNameRef<@VerbatimExprInnerRef::@nonexistent>]}
hw.module @VerbatimExprInnerRef() {
  hw.output
}

// -----

// expected-error @below {{inner symbol reference #hw.innerNameRef<@VerbatimExprSEInnerRef::@nonexistent> could not be found}}
%0 = sv.verbatim.expr.se "MACRO" : () -> i32 {symbols = [#hw.innerNameRef<@VerbatimExprSEInnerRef::@nonexistent>]}
hw.module @VerbatimExprSEInnerRef() {
  hw.output
}

// -----

// Users collected during table construction must not be verified until all
// symbols are available, including later local symbols and other modules' ports.
sv.macro.decl @ENABLE
hw.module @ForwardUsers() {
  sv.ifdef @ENABLE {
    sv.verbatim "// {{0}} {{1}} {{2}}" {symbols = [
      #hw.innerNameRef<@ForwardUsers::@local>,
      #hw.innerNameRef<@Later::@port>,
      #hw.innerNameRef<@Later::@wire>
    ]}
  }
  %local = sv.wire sym @local : !hw.inout<i1>
  hw.output
}
hw.module @Later(in %p: i1 {hw.exportPort = #hw<innerSym@port>}) {
  %wire = sv.wire sym @wire : !hw.inout<i1>
  hw.output
}

// A top-level operation with regions but no inner-symbol table must still be
// walked for users.
sv.ifdef @ENABLE {
  sv.verbatim "// {{0}}" {symbols = [#hw.innerNameRef<@Later::@wire>]}
}

// -----

sv.macro.decl @ENABLE
hw.module @NestedUser() {
  sv.ifdef @ENABLE {
    // expected-error @below {{inner symbol reference #hw.innerNameRef<@NestedUser::@missing> could not be found}}
    sv.verbatim "// {{0}}" {symbols = [#hw.innerNameRef<@NestedUser::@missing>]}
  }
  hw.output
}

// -----

// Duplicate symbols must still be diagnosed in pre-order, even though users
// are collected in post-order during the same walk.
sv.macro.decl @ENABLE
hw.module @NestedDuplicate() {
  // expected-note @below {{see existing inner symbol definition here}}
  %first = sv.wire sym @duplicate : !hw.inout<i1>
  sv.ifdef @ENABLE {
    // expected-error @below {{redefinition of inner symbol named 'duplicate'}}
    %second = sv.wire sym @duplicate : !hw.inout<i1>
  }
  hw.output
}
