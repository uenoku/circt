// RUN: circt-opt -rtl-cleanup %s | FileCheck %s

//CHECK-LABEL: rtl.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   sv.initial {
//CHECK-NEXT:     sv.fwrite "Middle\0A"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:     sv.fwrite "A1"
//CHECK-NEXT:     sv.fwrite "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg1)  {
//CHECK-NEXT:     sv.fwrite "B1"
//CHECK-NEXT:     sv.fwrite "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   rtl.output
//CHECK-NEXT: }

rtl.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.fwrite "A1"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite "B1"
  }
  sv.initial {
    sv.fwrite "Middle\n"
  }
  sv.alwaysff(posedge %arg0) {
    sv.fwrite "A2"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite "B2"
  }
  rtl.output
}

// CHECK-LABEL: rtl.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "A1"
// CHECK-NEXT:     sv.fwrite "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "B1"
// CHECK-NEXT:     sv.fwrite "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   rtl.output
// CHECK-NEXT: }

rtl.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B2"
  }
  rtl.output
}


// CHECK-LABEL: rtl.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "A1"
// CHECK-NEXT:     sv.fwrite "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "B1"
// CHECK-NEXT:     sv.fwrite "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "C1"
// CHECK-NEXT:     sv.fwrite "C2"
// CHECK-NEXT:   }(asyncreset : posedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "D1"
// CHECK-NEXT:     sv.fwrite "D2"
// CHECK-NEXT:   }
// CHECK-NEXT:   rtl.output
// CHECK-NEXT: }

rtl.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "C1"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite "D1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B2"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "C2"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite "D2"
  }
  rtl.output
}

//CHECK-LABEL: rtl.module @alwaysff_ifdef(%arg0: i1) {
//CHECK-NEXT:  sv.ifdef "FOO" {
//CHECK-NEXT:     sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:       sv.fwrite "A1"
//CHECK-NEXT:       sv.fwrite "B1"
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT:   rtl.output
//CHECK-NEXT: }

rtl.module @alwaysff_ifdef(%arg0: i1) {
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "A1"
    }
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "B1"
    }
  }
  rtl.output
}

// CHECK-LABEL: rtl.module @ifdef_merge(%arg0: i1) {
// CHECK-NEXT:    sv.ifdef "FOO"  {
// CHECK-NEXT:      sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:        sv.fwrite "A1"
// CHECK-NEXT:        sv.fwrite "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
rtl.module @ifdef_merge(%arg0: i1) {
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "A1"
    }
  }
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "B1"
    }
  }
  rtl.output
}

// CHECK-LABEL: rtl.module @ifdef_proc_merge(%arg0: i1) {
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = rtl.constant true
// CHECK-NEXT:      %0 = comb.xor %arg0, %true : i1
// CHECK-NEXT:      sv.ifdef.procedural "FOO"  {
// CHECK-NEXT:        sv.fwrite "A1"
// CHECK-NEXT:        sv.fwrite "%x"(%0) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.ifdef.procedural "BAR"  {
// CHECK-NEXT:        sv.fwrite "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
rtl.module @ifdef_proc_merge(%arg0: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.ifdef.procedural "FOO" {
      sv.fwrite "A1"
    }
    %true = rtl.constant true
    %0 = comb.xor %arg0, %true : i1
    sv.ifdef.procedural "FOO" {
       sv.fwrite "%x"(%0) : i1
    }
     sv.ifdef.procedural "BAR" {
       sv.fwrite "B1"
    }
  }
  rtl.output
}

// CHECK-LABEL: rtl.module @if_merge(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = rtl.constant true
// CHECK-NEXT:      %0 = comb.xor %arg1, %true : i1
// CHECK-NEXT:      sv.if %arg1  {
// CHECK-NEXT:        sv.fwrite "A1"
// CHECK-NEXT:        sv.fwrite "%x"(%0) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.if %0 {
// CHECK-NEXT:        sv.fwrite "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
rtl.module @if_merge(%arg0: i1, %arg1: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.if %arg1 {
      sv.fwrite "A1"
    }
    %true = rtl.constant true
    %0 = comb.xor %arg1, %true : i1
    sv.if %arg1 {
      sv.fwrite "%x"(%0) : i1
    }
    sv.if %0 {
      sv.fwrite "B1"
    }
  }
  rtl.output
}


// CHECK-LABEL: rtl.module @initial_merge(%arg0: i1) {
// CHECK-NEXT:    sv.initial {
// CHECK-NEXT:      sv.fwrite "A1"
// CHECK-NEXT:      sv.fwrite "B1"
// CHECK-NEXT:    }
rtl.module @initial_merge(%arg0: i1) {
  sv.initial {
    sv.fwrite "A1"
  }
  sv.initial {
    sv.fwrite "B1"
  }
  rtl.output
}

//CHECK-LABEL: rtl.module @always_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   sv.initial {
//CHECK-NEXT:     sv.fwrite "Middle\0A"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.always   posedge %arg0   {
//CHECK-NEXT:     sv.fwrite "A1"
//CHECK-NEXT:     sv.fwrite "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.always   posedge %arg1   {
//CHECK-NEXT:     sv.fwrite "B1"
//CHECK-NEXT:     sv.fwrite "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   rtl.output
//CHECK-NEXT: }
rtl.module @always_basic(%arg0: i1, %arg1: i1) {
  sv.always posedge %arg0 {
    sv.fwrite "A1"
  }
  sv.always posedge %arg1 {
    sv.fwrite "B1"
  }
  sv.initial {
    sv.fwrite "Middle\n"
  }
  sv.always posedge %arg0 {
    sv.fwrite "A2"
  }
  sv.always posedge %arg1 {
    sv.fwrite "B2"
  }
  rtl.output
}
