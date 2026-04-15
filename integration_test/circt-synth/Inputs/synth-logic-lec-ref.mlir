hw.module @xor_inv_ref(in %a : i4, in %b : i4, in %c : i4, out out : i4) {
  %ones = hw.constant -1 : i4
  %not_b = comb.xor %b, %ones : i4
  %0 = comb.xor %a, %not_b, %c : i4
  hw.output %0 : i4
}

hw.module @dot_ref(in %x : i4, in %y : i4, in %z : i4, out out : i4) {
  %ones = hw.constant -1 : i4
  %not_x = comb.xor %x, %ones : i4
  %xy = comb.and %not_x, %y : i4
  %z_or_xy = comb.or %z, %xy : i4
  %0 = comb.xor %not_x, %z_or_xy : i4
  hw.output %0 : i4
}
