// SRAM macro externs for the ChipTop synth experiment
// (circt-synth --sram-macros). Covers the four memories over the default
// max-fallback-bits that otherwise explode to flops and hang the flow:
// 16384x64, 8192x64+mask8, 1024x136+mask8, 512x256+mask4. All are 1R1W,
// read latency 0, write latency 1, single shared clock. Names are
// descriptive placeholders, NOT PDK cells: binding real PDK SRAM names
// (sky130 sram cache / ASAP7 compiler output) is later work; the emitted
// netlist instantiates these as blackboxes, mirroring Yosys SRAM handling.
module {
  hw.module.extern @sram_16384x64_1r1w(in %rclk : i1, in %raddr : i14, out rdout : i64, in %wclk : i1, in %we : i1, in %waddr : i14, in %wdin : i64) attributes {synth.sram.macro = #synth.sram_macro<64 x 16384, [#synth.sram_port<read, "rclk", "", "raddr", 14, "rdout", 64, "", 0, 0>, #synth.sram_port<write, "wclk", "we", "waddr", 14, "wdin", 64, "", 0, 1>]>}
  hw.module.extern @sram_8192x64_1r1w_mask8(in %rclk : i1, in %raddr : i13, out rdout : i64, in %wclk : i1, in %we : i1, in %waddr : i13, in %wdin : i64, in %wmask : i8) attributes {synth.sram.macro = #synth.sram_macro<64 x 8192, [#synth.sram_port<read, "rclk", "", "raddr", 13, "rdout", 64, "", 0, 0>, #synth.sram_port<write, "wclk", "we", "waddr", 13, "wdin", 64, "wmask", 8, 1>]>}
  hw.module.extern @sram_1024x136_1r1w_mask8(in %rclk : i1, in %raddr : i10, out rdout : i136, in %wclk : i1, in %we : i1, in %waddr : i10, in %wdin : i136, in %wmask : i8) attributes {synth.sram.macro = #synth.sram_macro<136 x 1024, [#synth.sram_port<read, "rclk", "", "raddr", 10, "rdout", 136, "", 0, 0>, #synth.sram_port<write, "wclk", "we", "waddr", 10, "wdin", 136, "wmask", 8, 1>]>}
  hw.module.extern @sram_512x256_1r1w_mask4(in %rclk : i1, in %raddr : i9, out rdout : i256, in %wclk : i1, in %we : i1, in %waddr : i9, in %wdin : i256, in %wmask : i4) attributes {synth.sram.macro = #synth.sram_macro<256 x 512, [#synth.sram_port<read, "rclk", "", "raddr", 9, "rdout", 256, "", 0, 0>, #synth.sram_port<write, "wclk", "we", "waddr", 9, "wdin", 256, "wmask", 4, 1>]>}
}
