# TechMapping Roadmap

## Current State

Liberty-imported modules now get `hw.techlib.info` via the `synth-annotate-techlib` pass, which extracts a single scalar delay per input pin from NLDM tables. TechMapper consumes this as `delay = [[d_per_input], ...]` with one integer (picoseconds) per input.

## Planned Improvements

### Supergate Generation

**Reference:** [Mishchenko et al., "Technology Mapping with Boolean Matching, Supergates and Choices"](https://people.eecs.berkeley.edu/~alanmi/publications/2005/tech05_map.pdf)

Currently TechMapper matches each cut against **single** library cells via NPN canonical forms. This is limiting because many useful functions require composing 2-3 gates (e.g., AND-OR-INVERT, XOR from NANDs). Supergates address this by pre-computing multi-gate compositions and registering them as additional patterns.

**Concept:** A supergate is a small DAG of library gates whose combined truth table is treated as a single matching pattern. For example, given NAND2 and INV cells, supergate generation would produce patterns for AND (= INV(NAND2)), OR (= NAND2(INV, INV)), etc.

**Representation in the pipeline:**

1. **Supergate library generation (new pass: `synth-gen-supergates`)**
   - Runs as a preprocessing step before TechMapper, after `synth-annotate-techlib`
   - Iteratively composes library cells (those with `hw.techlib.info`) up to configurable limits:
     - `max-inputs`: maximum number of primary inputs (e.g., 6)
     - `max-gates`: maximum number of gates in a supergate (e.g., 3-4)
     - `max-area`: area budget per supergate
   - For each composition, compute the NPN canonical form of the combined truth table
   - Emit the supergate as a new `hw::HWModuleOp` with:
     - Body containing the gate composition (instances of library cells)
     - `hw.techlib.info` with accumulated area and pin-to-pin delay through the DAG
     - A marker attribute (e.g., `synth.supergate = true`) to distinguish from primitive cells

   Pipeline order:
   ```
   import-liberty → annotate-techlib → gen-supergates → tech-mapper → STA
   ```

2. **TechMapper changes: none required initially**
   - Since supergates are just additional `hw::HWModuleOp`s with `hw.techlib.info`, TechMapper already picks them up as library patterns via NPN matching
   - The cut-based algorithm naturally considers supergates alongside primitive cells
   - Area/delay costs propagate correctly because supergates carry accumulated values

3. **Delay computation for supergates**
   - Pin-to-pin delay through a supergate DAG is the max delay along any path from input pin to output
   - Example: AND = INV(NAND2(A, B)) → delay_A = delay_NAND2_A + delay_INV, delay_B = delay_NAND2_B + delay_INV
   - Area = sum of constituent gate areas

4. **DAG-aware supergates (future)**
   - Initial supergates are tree-structured (each gate output feeds exactly one other gate)
   - DAG supergates allow fanout within the supergate, enabling patterns like `Y = NAND(AND(a,b), AND(a,c))` where input `a` fans out
   - This requires extending the composition algorithm to track shared inputs

**Key advantage:** Supergates are transparent to TechMapper — they're just more patterns. This keeps the mapper simple while dramatically expanding coverage. A typical standard cell library of ~100 gates can generate thousands of supergates covering most practical 4-6 input functions.

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
