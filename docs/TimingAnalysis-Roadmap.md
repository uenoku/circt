# Timing Analysis Engine: Enhancements & NLDM/CCS Roadmap

## Current State

The two-stage static timing analysis engine at `lib/Dialect/Synth/Analysis/Timing/` provides:

- **TimingGraph**: DAG representation with Kahn's topological sort
- **ArrivalAnalysis**: Forward arrival time propagation
- **RequiredTimeAnalysis**: Backward propagation with RAT/slack
- **PathEnumerator**: SFXT-based K-worst path enumeration with through-point filtering
- **DelayModel**: Pluggable interface with `UnitDelayModel` and `AIGLevelDelayModel`
- **TimingAnalysis**: Unified API with `reportTiming()` and `runFullAnalysis()`

The `DelayContext` already carries `inputSlew` and `outputLoad` fields, and `DelayResult` returns `outputSlew`, but these are currently unused (always 0). This infrastructure was designed to enable NLDM and CCS without API changes.

Implementation note (2026-03): the timing flow should reuse the existing
`import-liberty` translation pipeline instead of adding a second standalone
Liberty parser under timing analysis.

## TimingAnalysis Status (Excluding Typed NLDM Attribute Migration)

This section tracks the readiness of `TimingAnalysis` itself, independent of
the ongoing `#synth` typed-NLDM attribute work.

### Completed and Stable

- Core analysis pipeline is implemented and exercised:
  - `TimingGraph` construction and DAG traversal
  - `ArrivalAnalysis` forward propagation
  - `RequiredTimeAnalysis` backward RAT/slack propagation
  - `PathEnumerator` K-worst path query engine
- Hierarchical timing flow is active in `TimingAnalysis` (not in
  `LongestPathAnalysis`).
- Timing report integration is established through
  `synth-print-timing-analysis` and `circt-synth`.
- CLI flow decisions are enforced in practice:
  - report pass is separate from `DesignProfiler`
  - report output flag is `--timing-report-dir`
  - report mode requires explicit `--top`
  - report labels include bit-level node names
- Arc context plumbing is in place end-to-end:
  - timing arcs carry `op + inputIndex + outputIndex`
  - delay recomputation uses arc context in arrival/RAT/path analysis
- Graph storage strategy has been updated for scale:
  - nodes remain `unique_ptr`
  - arcs use typed bump allocation

### Major Correctness Fixes Already Landed

- Fixed missing graph-region arcs for use-before-def situations by ensuring
  nodes are materialized via `getOrCreateNode` during arc construction.
- Fixed global K-worst ordering so results are correctly sorted across all
  endpoints (not endpoint-local ordering only).
- Fixed hierarchical endpoint lookup collisions by avoiding insertion of
  endpoint-only nodes into the value lookup map.

### Validation Baseline

- Timing unit tests in `CIRCTSynthTests` are passing, including full
  `TimingAnalysisTest.*` coverage.
- `circt-synth` and timing-report smoke flows build and run successfully.

### Remaining Gaps in TimingAnalysis Core (Not About Typed Attr Format)

- Full NLDM numerical behavior is not complete yet:
  - proper LUT interpolation over `(inputSlew, outputLoad)`
  - transition/slew propagation through nodes
  - output-load modeling from fanout pin capacitances
  - iterative slew/load convergence loop
- Advanced STA features are still pending:
  - multi-clock domain constraints and CDC handling
  - MCMM (multi-corner/multi-mode)
  - incremental re-analysis after local netlist edits
  - parallel path-query execution (documented as deferred)

### Practical Readiness Summary

- `TimingAnalysis` is functionally usable today for graph construction,
  arrival/RAT/slack reporting, and K-worst path reporting in current synth
  flows.
- Signoff-grade NLDM/CCS accuracy remains a roadmap item and depends on the
  slew/load/interpolation milestones below.

---

## Near-term Enhancements

### 1. Slew Propagation Infrastructure

The slew fields exist in the data structures but are not propagated through the analysis. This is the single most important prerequisite for NLDM support.

**What to change:**

- Extend `NodeArrivalData` to store per-node slew alongside arrival time.
- In `ArrivalAnalysis::propagate()`, when processing a fanout arc, pass the current node's slew as `DelayContext::inputSlew` to the delay model.
- Store the returned `DelayResult::outputSlew` on the successor node.
- Gate all slew-related logic behind `DelayModel::usesSlewPropagation()` so that level models (AIG, unit) pay zero overhead.

**Files affected:** `ArrivalAnalysis.h`, `ArrivalAnalysis.cpp`

### 2. Output Load Computation

`DelayContext::outputLoad` is always 0 today. For NLDM, delay is a function of `(inputSlew, outputLoad)`, making load computation essential.

**Design options:**

- **Simple fanout model**: `outputLoad = fanoutCount * defaultPinCapacitance`. Useful for early estimation.
- **Liberty-based model**: Sum input pin capacitances from the Liberty cell library for each fanout gate. This requires the Liberty parser (see below).
- **Wire load model**: Add parasitic wire capacitance based on estimated wire length or a wire load table.

**Suggested approach:** Define a `LoadModel` interface (or extend `DelayModel` with a `getInputCapacitance(pin)` method). Compute total output load for each node before arrival propagation.

**Files affected:** `DelayModel.h`, `TimingGraph.cpp` (annotate nodes with load), `ArrivalAnalysis.cpp`

### 3. Clock Domain Awareness

Currently all endpoints share a single clock period constraint. Real designs have multiple clock domains.

**What to add:**

- Track which clock drives each register (already available from `seq.CompRegOp`/`seq.FirRegOp` clock operand).
- In `RequiredTimeAnalysis`, use per-domain clock periods instead of a single global value.
- Flag or exclude cross-domain paths (CDC paths) from timing analysis.
- Support `set_clock_groups` and `set_false_path` style constraints.

**Files affected:** `TimingGraph.h` (clock domain on nodes), `RequiredTimeAnalysis.h/.cpp`

### 4. Multi-corner / Multi-mode Analysis

Run the same timing graph with different delay models representing process corners (slow, typical, fast) and operating modes.

**Design:**

```
struct CornerAnalysis {
  std::string name;           // e.g., "ss_0.72v_125c"
  const DelayModel *model;
  std::unique_ptr<ArrivalAnalysis> arrivals;
  std::unique_ptr<RequiredTimeAnalysis> required;
};
```

`TimingAnalysis` holds a vector of `CornerAnalysis` and reports worst-case across all corners. The timing graph structure is shared (same netlist), only delays differ.

**Files affected:** `TimingAnalysis.h/.cpp`, `TimingReport.cpp`

### 5. Incremental Analysis

After ECO changes (gate resizing, buffer insertion, net rewiring), avoid full re-analysis.

**Approach:**

- The SFXT (suffix tree) is already per-endpoint, so only endpoints whose fan-in cone was modified need recomputation.
- Track a dirty set of modified nodes; propagate dirtiness forward (for arrival) and backward (for required time).
- Re-run arrival propagation only for nodes reachable from dirty nodes in topological order.

**Files affected:** `ArrivalAnalysis.h/.cpp`, `TimingGraph.h` (change tracking)

### 6. Parallelization Opportunities (Deferred)

Parallelism is feasible in the current architecture, but should be staged to
minimize risk.

**A. Path query parallelism (recommended first):**

- In `PathEnumerator::enumerate()`, endpoint processing is independent:
  - `buildSuffixTree(endpoint, ...)`
  - `extractKPaths(endpoint, ...)`
- Parallelize the per-endpoint loop using thread-local SFXT/work buffers and
  thread-local path result vectors.
- Merge all local results, then perform one global sort and final `maxPaths`
  trim to preserve true global K-worst semantics.
- This is the highest ROI path because it avoids mutating shared graph state.

**B. Multi-query parallelism:**

- Independent path queries can run concurrently on the same `TimingGraph` +
  `ArrivalAnalysis` as read-only data.
- Each query should own its own `PathEnumerator` scratch state/results.

**C. Graph build parallelism (later, higher effort):**

- `TimingGraph::build()` currently mutates shared containers (`nodes`, `arcs`,
  `valueToNode`, fanin/fanout), so naive threading is unsafe.
- A safe approach is a two-phase build:
  1. Parallel collection of per-op/per-instance node/arc intents.
  2. Serial merge that assigns stable node IDs and wires fanin/fanout.
- Hierarchical elaboration and deterministic naming/order must be preserved.

**Recommended order:**

1. Parallelize path enumeration by endpoint.
2. Add optional multi-query parallel execution.
3. Revisit graph-build parallelization with a two-phase deterministic merge.

---

## Path to NLDM Support

NLDM (Non-Linear Delay Model) is the standard Liberty timing model. It uses 2D lookup tables indexed by `(inputSlew, outputLoad)` to compute `delay` and `outputSlew` for each timing arc.

### Step A: Liberty Data Bridge (Reuse ImportLiberty)

Reuse `import-liberty` output attributes (for example
`synth.liberty.library` and `synth.liberty.pin`) as the canonical Liberty
source for timing analysis.

The timing engine should consume normalized NLDM metadata emitted by
`import-liberty` (for example `synth.nldm.time_unit_ps` and
`synth.nldm.arcs` inside pin attrs), rather than re-parsing nested generic
dictionary structures or introducing another independent Liberty parser.

**Key data structures:**

```
LibertyLibrary
  -> LibertyCell (name, area, leakage_power)
    -> LibertyPin (name, direction, capacitance, max_transition)
      -> LibertyTimingArc (related_pin, timing_sense, timing_type)
        -> LUT2D delay       (index_1: input_slew, index_2: output_load)
        -> LUT2D output_slew (index_1: input_slew, index_2: output_load)
```

**LUT2D** stores the index vectors and the value matrix, and provides bilinear interpolation:

```
struct LUT2D {
  SmallVector<double> index1;  // e.g., input transition times
  SmallVector<double> index2;  // e.g., output capacitances
  SmallVector<SmallVector<double>> values;

  double interpolate(double x1, double x2) const;
};
```

**Files:**
- existing Liberty import: `lib/Conversion/ImportLiberty/ImportLiberty.cpp`
- timing-side view/bridge:
  - `include/circt/Dialect/Synth/Analysis/Timing/Liberty.h`
  - `lib/Dialect/Synth/Analysis/Timing/Liberty.cpp`

### Step A.1: Mapping Attributes for Cell/Arc Resolution

To enable unambiguous arc lookup during timing evaluation, mapped operations
may carry timing-oriented attributes, e.g.:

- `synth.liberty.cell = "<cell_name>"`
- optional pin-level hints for non-trivial mappings

This keeps technology mapping and STA coupled through explicit IR metadata.

### Step A.2: Typed NLDM Attribute Migration (`#synth`)

Move from ad-hoc dictionary payloads toward typed Synth attributes for NLDM
metadata, so importer and timing analysis share a stable schema.

**Original sub-goal:**

- Introduce a typed `#synth` NLDM metadata path emitted by `import-liberty`
  (instead of reconstructing NLDM tables from generic nested dictionaries in
  timing analysis), while keeping the timing flow on `TimingAnalysis`.

**Task list (active):**

1. Define typed Synth AttrDefs for normalized NLDM data:
   - module-level time-unit scale (ps)
   - per-arc payload (`related_pin`, `to_pin`, `timing_sense`, rise/fall
     samples)
2. Register and expose the new AttrDefs in the Synth dialect codegen flow.
3. Update `import-liberty` to emit typed `#synth...` attrs instead of generic
   `synth.nldm.*` dictionary keys.
4. Update timing-side consumers (`LibertyLibrary`, `NLDMDelayModel`) to read
   typed attrs first, with temporary fallback to dictionary form.
5. Add tests for importer emission, parser/printer round-trip, and delay-model
   evaluation from typed attrs.
6. Document deprecation path for dictionary-form NLDM metadata once migration is
   complete.

**Immediate next steps:**

1. Land AttrDefs + dialect registration in an isolated commit.
2. Switch ImportLiberty emission to typed attrs in a follow-up commit.
3. Migrate timing bridge/model readers and keep fallback during transition.
4. Add end-to-end regression coverage (`circt-translate` import +
   `TimingAnalysisTest` NLDM checks).
5. Remove legacy dictionary fallback after downstream users are migrated.

### Step B: NLDMDelayModel Implementation

```cpp
class NLDMDelayModel : public DelayModel {
public:
  NLDMDelayModel(const LibertyLibrary &lib);

  DelayResult computeDelay(const DelayContext &ctx) const override {
    // 1. Identify cell type from ctx.op (e.g., via an attribute or op type mapping)
    // 2. Look up LibertyCell in the library
    // 3. Find the timing arc for (ctx.inputValue -> ctx.outputValue)
    // 4. Bilinear interpolation: delay = LUT.interpolate(ctx.inputSlew, ctx.outputLoad)
    // 5. Similarly for output slew
    // 6. Return {scaledDelay, outputSlew}
  }

  StringRef getName() const override { return "nldm"; }
  bool usesSlewPropagation() const override { return true; }

private:
  const LibertyLibrary &library;
  // Map from MLIR operation type/attributes to Liberty cell names
  DenseMap<StringRef, const LibertyCell *> cellMap;
};
```

**Bilinear interpolation** on the LUT handles arbitrary `(slew, load)` points between table entries. Extrapolation beyond table bounds should clamp.

**Delay units:** keep interpolation in `double`, then round to integer
picoseconds when returning `DelayResult::delay`.

**Files:** `DelayModel.h` (add class), `DelayModel.cpp` (implement)

### Step C: Iterative Slew/Load Convergence

NLDM creates a circular dependency: output slew depends on load, load depends on downstream pin capacitances, and slew at downstream gates depends on the output slew of upstream gates. This requires iteration.

**Algorithm:**

```
1. Initialize all slews to a default value (e.g., from Liberty default_input_transition)
2. Compute output loads for all nodes (sum of fanout pin capacitances)
3. Forward propagate: compute delays and output slews using NLDM LUTs
4. Check convergence: max|slew_new - slew_old| < epsilon
5. If not converged, go to step 3 (typically converges in 2-3 iterations)
6. Run backward analysis (required times, slack)
```

This loop should live in `TimingAnalysis::runFullAnalysis()`, gated on `delayModel->usesSlewPropagation()`.

**Files affected:** `TimingAnalysis.cpp`

### Step D: Technology Mapping Bridge

To use NLDM, the netlist must be mapped to specific Liberty cells. This bridges the synthesis flow with timing:

- After technology mapping (e.g., in the Synth dialect transforms), annotate operations with their Liberty cell name.
- `NLDMDelayModel` reads this annotation to look up the correct cell.
- Alternative: maintain a mapping table from MLIR op types to Liberty cells.

---

## Path to CCS Support

CCS (Composite Current Source) is the successor to NLDM, used for advanced nodes (below 28nm) where simple voltage-based delay models lose accuracy.

### How CCS Differs from NLDM

| Aspect | NLDM | CCS |
|--------|------|-----|
| Model | Voltage-based LUT | Current-source waveform |
| Output | delay + slew (two scalars) | Full output waveform |
| Accuracy | ~5% at 45nm+ | ~1% at all nodes |
| Data | 2D delay/slew tables | Current waveform segments + receiver capacitance model |

### Implementation Path

Since the `DelayModel` abstraction hides internal computation, CCS is another subclass:

```cpp
class CCSDelayModel : public DelayModel {
  DelayResult computeDelay(const DelayContext &ctx) const override {
    // 1. Look up CCS current source data for the cell/arc
    // 2. Construct the effective current waveform
    // 3. Simulate driving the RC output load
    // 4. Compute delay from input threshold crossing to output threshold crossing
    // 5. Compute output slew from output waveform threshold crossings
    // 6. Return {delay, outputSlew}
  }
  bool usesSlewPropagation() const override { return true; }
};
```

**Prerequisites:** All of the NLDM infrastructure (slew propagation, load computation, Liberty parser, iterative convergence). CCS additionally needs:

- CCS-specific Liberty data parsing (current source tables, receiver capacitance models)
- RC load simulation (could be simplified to effective capacitance model)
- Waveform threshold crossing computation

### Composite Approach

In practice, many tools support mixed NLDM/CCS: use CCS for critical paths and NLDM elsewhere. This could be implemented by having a `CompositeDelayModel` that delegates to CCS or NLDM based on the cell or criticality.

---

## Suggested Priority Order

| Priority | Enhancement | Unlocks |
|----------|------------|---------|
| 1 | Slew propagation in ArrivalAnalysis | NLDM |
| 2 | Output load computation | NLDM |
| 3 | Liberty parser (`.lib` -> in-memory LUTs) | NLDM |
| 4 | `NLDMDelayModel` with bilinear interpolation | NLDM timing |
| 5 | Iterative slew convergence loop | Accurate NLDM |
| 6 | Clock domain awareness | Multi-clock designs |
| 7 | Multi-corner support | Signoff-quality analysis |
| 8 | CCS model | Advanced node accuracy |
| 9 | Incremental analysis | ECO flows |

Items 1-5 form the NLDM critical path. Items 6-9 are independently valuable and can be parallelized.
