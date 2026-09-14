// SRAM macro externs for circt-synth --sram-macros tests.
// Names are descriptive placeholders, not PDK cells.
module {
  hw.module.extern @sram_16x8_1r1w(in %rclk : i1, in %raddr : i4, out rdout : i8, in %wclk : i1, in %we : i1, in %waddr : i4, in %wdin : i8) attributes {synth.sram.macro = #synth.sram_macro<8 x 16, [#synth.sram_port<read, "rclk", "", "raddr", 4, "rdout", 8, "", 0, 0>, #synth.sram_port<write, "wclk", "we", "waddr", 4, "wdin", 8, "", 0, 1>]>}
}
