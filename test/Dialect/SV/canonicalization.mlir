// RUN: circt-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @if_dead_condition_0(%arg0: i1) {
// CHECK-NEXT:    return 
func @if_dead_condition_0(%arg0: i1) {
  sv.always posedge %arg0 {
    %false = comb.constant false
    sv.if %false  {
      sv.fwrite "Unreachable"
    }
  }
  return
}

// CHECK-LABEL: func @if_dead_condition_1(%arg0: i1) {
// CHECK-NEXT:    sv.always posedge %arg0  {
// CHECK-NEXT:      sv.fwrite "Reachable"
// CHECK-NEXT:    }
func @if_dead_condition_1(%arg0: i1) {
  sv.always posedge %arg0 {
    %true = comb.constant true
    sv.if %true  {
      sv.fwrite "Reachable"
    }
  }
  return
}

// CHECK-LABEL: func @if_dead_condition_2(%arg0: i1) {
// CHECK-NEXT:    sv.always posedge %arg0  {
// CHECK-NEXT:      sv.fwrite "Reachable"
// CHECK-NEXT:    }
func @if_dead_condition_2(%arg0: i1) {
  sv.always posedge %arg0 {
    %true = comb.constant true
    sv.if %true  {
      sv.fwrite "Reachable"
    } else {
      sv.fwrite "Unreachable"
    } 
  }
  return
}

// CHECK-LABEL: func @if_dead_condition_3(%arg0: i1) {
// CHECK-NEXT:    sv.always posedge %arg0  {
// CHECK-NEXT:      sv.fwrite "Reachable"
// CHECK-NEXT:    }
func @if_dead_condition_3(%arg0: i1) {
  sv.always posedge %arg0 {
    %false = comb.constant false
    sv.if %false  {
      sv.fwrite "Unreachable"
    } else {
      sv.fwrite "Reachable"
    } 
  }
  return
}

// CHECK-LABEL: func @if_dead_condition_4(%arg0: i1) {
// CHECK-NEXT:    sv.always posedge %arg0  {
// CHECK-NEXT:      sv.fwrite "Reachable"
// CHECK-NEXT:      sv.fwrite "Alive"
// CHECK-NEXT:    }
func @if_dead_condition_4(%arg0: i1) {
  sv.always posedge %arg0 {
    %true = comb.constant true
    sv.if %true  {
      sv.fwrite "Reachable"
    } else {
      sv.fwrite "Unreachable"
    } 
    sv.fwrite "Alive"
  }
  return
}