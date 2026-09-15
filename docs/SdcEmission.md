# SDC emission (`write_sdc` equivalent)

## Status: designed, not implemented. Top remaining flow gap.

## Measured input: ChipTop clock inventory (authoritative, mapped MLIR)

226,219 flop clock pins driven by exactly 10 distinct nets:

| Net | Flops | Notes |
|---|---|---|
| `%W0_clk` | 119,436 | verify root: SRAM-domain or gated? |
| `%clock` | 70,567 | main domain (presumed) |
| `%RW0_clk` | 35,931 | verify root |
| `%R0_clk` | 92 | |
| `%io_ungated_clock` | 65 | |
| `%auto_pbus_clock_groups_in_member_pbus_0_clock` | 64 | peripheral bus |
| `%io_jtag_clock` | 46 | async vs core |
| `%clint_domain.clock` | 10 | |
| `%mapped.Y` | 5 | **comb-generated clock** (gated): needs generated-clock or false-path treatment |
| `%fixedClockNode.auto_anon_out_0_clock` | 3 | |

Plus: reset synchronizer chains (`phy_io_outer_reset_catcher/.../sync_0`,
seen as STA recovery checks) want `set_false_path` on their async inputs.

## Minimal viable `synth-emit-sdc`

New ModuleOp pass (options: `--top`, `--clock-period-ns` default 10,
`--output-file`):

1. For the top module, collect flop clock pins; trace each driver back
   through INV/BUF cells and ports to a top-level input port
   (hierarchical walk via the instance graph; stop at clock gates,
   dividers, or comb logic and report them as non-roots).
2. Emit `create_clock -period <p> [get_ports <root>]` per root.
3. Emit `set_clock_groups -asynchronous` between roots that never share
   a source (jtag/pbus vs core at minimum; default everything async
   unless proven otherwise).
4. Emit `set_false_path` to reset-synchronizer data inputs (structural:
   flops whose input cone is a 2+ synchronizer chain from another domain;
   v1 may take an explicit pin list flag instead).
5. Emit `set_input_delay`/`set_output_delay` with flag-supplied defaults
   (0.2 × period is the usual ORFS starting point, not derivable).
6. Warn-list: comb-generated clocks (`%mapped.Y` case), unconstrained
   endpoints, clocks with no flops.

## Validation

- STA with emitted SDC vs the hand-written `sta_*.tcl` stanzas used for
  the scoreboard: same worst paths, no new violations from over-constraint.
- ORFS run consuming netlist + emitted SDC (flow `constraint.sdc` input).
- LEC-irrelevant (constraints don't change logic), but SDC correctness
  gates every timing number in the scoreboard.

## Non-goals (v1)

Generated-clock wave-shaping for divided clocks, SI-aware derates,
multi-corner SDC, power-domain UPF coupling.
