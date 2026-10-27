// RUN: firtool --ir-fir %s | FileCheck %s --check-prefixes=CHECK,ENABLED
// RUN: firtool --ir-fir --disable-object-promotion %s | FileCheck %s --check-prefixes=CHECK,DISABLED
// RUN: firtool --ir-fir --disable-object-promotion=false %s | FileCheck %s --check-prefixes=CHECK,ENABLED

firrtl.circuit "ObjectPromotion" {
  firrtl.extclass @Empty()

  // CHECK-LABEL: firrtl.module @ObjectPromotion
  firrtl.module @ObjectPromotion() {
    // ENABLED-NEXT: %obj = firrtl.object.create() : () -> !firrtl.class<@Empty()>
    // DISABLED-NEXT: %obj = firrtl.object @Empty()
    // CHECK-NEXT: }
    %obj = firrtl.object @Empty()
  }
}
