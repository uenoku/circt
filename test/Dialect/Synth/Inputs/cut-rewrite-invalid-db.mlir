module {
  hw.module @unsupported(in %i0 : i1, in %i1 : i1, out y : i1) {
    %0 = comb.and %i0, %i1 : i1
    hw.output %0 : i1
  }
}
