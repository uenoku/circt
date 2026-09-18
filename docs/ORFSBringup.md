# ORFS bringup for CIRCT-synthesized netlists (item 3)

## Status: collateral ready for fifo/sky130hd; blocked on an OpenROAD
binary (none installed; no docker image; building from source is hours).
ChipTop additionally blocked on PDK SRAM macros (item 14).

## What exists today

- Netlist: `/tmp/e2e/momap/fifo_buf.v` (buffered fifo, 9256 inst,
  4.451ns worst arrival, flops exact) — regenerate any time with:
  ```
  circt-synth --liberty <sky130hd lib> --strip-lib-to-extern --top fifo \
    --buffer-high-fanout /tmp/e2e/fifo.mlir -o fifo.mapped.mlir
  circt-opt --test-apply-lowering-options='options=disallowReplicate' \
    --export-verilog fifo.mapped.mlir -o /dev/null > fifo.v
  ```
- SDC: emitted by the same synth run (`--sdc-output fifo.sdc
  --sdc-clock-period 10 --sdc-input-delay 1 --sdc-output-delay 1`);
  verified loading in OpenSTA (`read_sdc` clean, all paths MET).
  Regenerate alongside the netlist — never reuse stale SDC (port lists
  come from the actual hierarchy).
- Scoreboard STA (no-IO-delay hand TCL) and SDC-constrained STA agree
  the design meets timing at period 10.

## To run (once `openroad` exists)

ORFS checkout: `circt-sfc-experiments/OpenROAD-flow-scripts` (untracked
there; do not commit run dirs into it — recreate per run):

```
FLOW=$PWD/circt-sfc-experiments/OpenROAD-flow-scripts/flow
NICK=circt_fifo
mkdir -p $FLOW/designs/sky130hd/$NICK/src/$NICK
cp fifo.v $FLOW/designs/sky130hd/$NICK/src/$NICK/fifo.v
cp fifo.sdc $FLOW/designs/sky130hd/$NICK/constraint.sdc
cat > $FLOW/designs/sky130hd/$NICK/config.mk <<EOF
export DESIGN_NAME = fifo
export DESIGN_NICKNAME = $NICK
export PLATFORM = sky130hd
export VERILOG_FILES = \$(DESIGN_HOME)/src/\$(DESIGN_NICKNAME)/fifo.v
export SDC_FILE = \$(DESIGN_HOME)/\$(PLATFORM)/\$(DESIGN_NICKNAME)/constraint.sdc
export CORE_UTILIZATION = 40
export PLACE_DENSITY_LB_ADDON = 0.20
EOF
cd $FLOW && make DESIGN_CONFIG=./designs/sky130hd/$NICK/config.mk
```

Notes:

- fifo needs no SRAM macros (memories are flop arrays) — it is the
  right first ORFS target. ChipTop needs item-14 macros first.
- `disallowReplicate` export is mandatory: replication syntax chokes
  both OpenSTA and ORFS Verilog readers (item 1).
- Yosys/ABC reference for the same design: 3,478 cells / 23.84ns
  (task.md item 3) — the area/delay gap to close is mostly ABC delay
  mapping + mux4, not place-and-route.
- Do not buffer clock trees here (ideal clocks in STA; CTS is ORFS's
  job). Our inserted `buf_1` cells are ordinary logic buffers.
