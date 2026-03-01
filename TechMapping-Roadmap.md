# TechMapping Roadmap

## Current State

Liberty-imported modules now get `hw.techlib.info` via the `synth-annotate-techlib` pass, which extracts a single scalar delay per input pin from NLDM tables. TechMapper consumes this as `delay = [[d_per_input], ...]` with one integer (picoseconds) per input.

## Planned Improvements

### Multi-Point Delay Sampling

Currently each input pin gets a single delay value (worst-case, center, or first from the NLDM table). Real cells exhibit load- and slew-dependent delay, so a single point loses information.

**Idea:** Sample multiple delay points from the 2D NLDM table and create multiple TechMapper patterns for the same cell, each representing a different operating condition. For example:
- Sample at (low slew, low load), (low slew, high load), (high slew, low load), (high slew, high load)
- Or sample along a grid of N points across both axes

This would let TechMapper make context-sensitive choices — e.g., prefer a cell variant with better delay under the actual fanout conditions of that node.

**Required changes:**
1. `synth-annotate-techlib`: Add a `sampling` strategy that emits multiple `hw.techlib.info` entries (or a richer delay representation)
2. `TechMapper`: Extend pattern matching to handle multiple patterns per cell, selecting the best match based on load/slew context
3. Define how load/slew context is estimated during mapping (e.g., from fanout count or wire load model)

### Slew-Dependent Delay (NLDM 2D Interpolation)

Instead of collapsing the 2D table to a scalar, perform proper bilinear interpolation at mapping time using estimated input slew and output load. This requires:
- Carrying the full NLDM table through to TechMapper (not just a scalar)
- Iterative timing-driven mapping where slew estimates converge

### CCS-Based Delay

For advanced nodes, CCS current-source models provide more accurate delay than NLDM. The `synth.ccs.pilot.arcs` data is already imported. Future work could:
- Use CCS waveform convolution for delay estimation during mapping
- Bridge to the existing `circt-sta` CCS timing engine

### Load-Aware Pattern Selection

TechMapper currently picks patterns purely by area/delay cost. With multiple delay samples per cell, it could:
- Estimate output load from fanout during cut evaluation
- Interpolate delay for the estimated load
- Make load-aware area-delay tradeoffs
