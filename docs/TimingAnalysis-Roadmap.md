# Timing Analysis Engine: Enhancements & NLDM/CCS Roadmap

## Current State

The two-stage static timing analysis engine at `lib/Dialect/Synth/Analysis/Timing/` provides:

- **TimingGraph**: DAG representation with Kahn's topological sort
- **ArrivalAnalysis**: Forward arrival time propagation
- **RequiredTimeAnalysis**: Backward propagation with RAT/slack
- **PathEnumerator**: SFXT-based K-worst path enumeration with through-point filtering
- **DelayModel**: Pluggable interface with `UnitDelayModel` and `AIGLevelDelayModel`
- **TimingAnalysis**: Unified API with `reportTiming()` and `runFullAnalysis()`

`DelayContext` now carries active `inputSlew` and `outputLoad` propagation in
arrival analysis, and `DelayResult::outputSlew` is consumed for
slew-capable models. This infrastructure is in place for NLDM and CCS.

The `DelayModel` interface now also includes a waveform-propagation hook
(`usesWaveformPropagation()` + `computeOutputWaveform(...)`) so CCS can be
introduced without changing analysis call sites again.

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

**Status (2026-03): In progress.**

- Arrival propagation now carries a per-startpoint slew value through
  `ArrivalInfo` and feeds `DelayContext::inputSlew` on each arc.
- `DelayModel::usesSlewPropagation()` now controls whether returned
  `outputSlew` is consumed or input slew is forwarded unchanged.
- A unit regression (`ArrivalAnalysisPropagatesSlewWhenEnabled`) verifies slew
  increases across a combinational chain for a slew-aware delay model.
- Remaining work: connect real NLDM output-slew table evaluation and couple
  with output-load computation and convergence.

### 2. Output Load Computation

`DelayContext::outputLoad` is always 0 today. For NLDM, delay is a function of `(inputSlew, outputLoad)`, making load computation essential.

**Design options:**

- **Simple fanout model**: `outputLoad = fanoutCount * defaultPinCapacitance`. Useful for early estimation.
- **Liberty-based model**: Sum input pin capacitances from the Liberty cell library for each fanout gate. This requires the Liberty parser (see below).
- **Wire load model**: Add parasitic wire capacitance based on estimated wire length or a wire load table.

**Suggested approach:** Define a `LoadModel` interface (or extend `DelayModel` with a `getInputCapacitance(pin)` method). Compute total output load for each node before arrival propagation.

**Files affected:** `DelayModel.h`, `TimingGraph.cpp` (annotate nodes with load), `ArrivalAnalysis.cpp`

**Status (2026-03): In progress.**

- Arrival propagation now computes per-node output load as the sum of fanout
  input capacitances reported by `DelayModel::getInputCapacitance(...)`.
- `DelayContext::outputLoad` is now populated on arc delay evaluation.
- `NLDMDelayModel` implements `getInputCapacitance(...)` using typed Liberty
  pin capacitance metadata through the timing Liberty bridge.
- This hook-based design is CCS-extensible: CCS models can provide nonlinear
  effective input capacitance and receiver models without changing analysis
  traversal logic.

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
`import-liberty` (for example `synth.nldm.time_unit` and
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

**Status (2026-03): In progress.**

- Hierarchical graph construction treats Liberty cell-like `hw.instance`
  operations as black-box timing arcs (instead of elaborating child internals)
  when `synth.liberty.cell` or `synth.liberty.pin` metadata is present.
- This preserves instance-level arc context (`op + inputIndex + outputIndex`)
  for NLDM/CCS lookup.

### Step A.2: Typed NLDM Attribute Migration (`#synth`)

Move from ad-hoc dictionary payloads toward typed Synth attributes for NLDM
metadata, so importer and timing analysis share a stable schema.

**Original sub-goal:**

- Introduce a typed `#synth` NLDM metadata path emitted by `import-liberty`
  (instead of reconstructing NLDM tables from generic nested dictionaries in
  timing analysis), while keeping the timing flow on `TimingAnalysis`.

**Status (2026-03): Completed.**

- Typed Synth AttrDefs are in place and registered:
  - `#synth.nldm_time_unit<...>`
  - `#synth.nldm_arc<...>` (with optional index vectors + sample arrays)
- `import-liberty` emits typed NLDM metadata.
- Timing-side consumers (`LibertyLibrary`, `NLDMDelayModel`) consume typed attrs
  and legacy dictionary fallback has been removed.
- Regression coverage exists for import emission and typed-only timing behavior.

**Immediate next steps (active):**

1. Finalize convergence behavior in `TimingAnalysis::runFullAnalysis()` with
   tuned defaults and diagnostics for slew/load-dependent models.
2. Add/expand end-to-end `circt-synth` NLDM timing-report tests to guard
   pass/CLI integration (`--timing-report-dir`, `--top`, model selection).
3. Implement first CCS pilot model on top of the waveform hook while sharing
   Liberty cell/arc resolution and load modeling infrastructure with NLDM.

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

**Status (2026-03): In progress.**

- `NLDMDelayModel` now consumes typed arc index/value arrays and performs
  clamped interpolation:
  - bilinear interpolation when both index axes are present,
  - linear interpolation for 1D tables,
  - first-sample fallback for degenerate tables.
- `NLDMDelayModel` no longer falls back to AIG-level delay for unmapped arcs;
  unmatched NLDM arcs default to zero delay unless explicit override attrs are
  present.
- Transition-table based output-slew interpolation is being wired using typed
  `rise_transition` / `fall_transition` payloads.
- End-to-end `circt-synth` timing-report coverage now includes NLDM model
  activation and path-delay checks, including chained cell-instance paths.

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

**Status (2026-03): Started.**

- `runFullAnalysis()` now performs iterative arrival analysis for models that
  report `usesSlewPropagation() == true`, with configurable maximum iterations
  and slew-delta convergence threshold.
- The last iteration count is tracked for observability.
- Remaining work: tighten convergence heuristics for models where effective
  load depends on slew/waveform state and expose optional convergence reporting.

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
