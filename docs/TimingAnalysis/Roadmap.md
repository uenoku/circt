# Timing Analysis Engine: Enhancements & NLDM/CCS Roadmap

For implementation details of slew propagation, damping, and convergence
diagnostics, see `docs/TimingAnalysis/SlewConvergence.md`.
For CCS pilot waveform-derived metrics and delay mode details, see
`docs/TimingAnalysis/CCSPilotWaveformDerived.md`.

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

- NLDM numerical behavior is nearly complete:
  - ~~proper LUT interpolation over `(inputSlew, outputLoad)`~~ done
  - ~~transition/slew propagation through nodes~~ done (infrastructure)
  - ~~output-load modeling from fanout pin capacitances~~ done
  - ~~iterative slew/load convergence loop~~ done
  - remaining: output-slew transition-table wiring for full accuracy
- **Rise/fall edge-aware timing is missing** (see dedicated section below).
  The timing graph uses a single delay/arrival per arc/node. Timing sense
  (`positive_unate`, `negative_unate`, `non_unate`) is imported but unused.
  This can cause significant inaccuracy (rise/fall delays can differ 2-3x).
- Golden-reference validation against OpenSTA is missing (see new section below).
- Technology mapping bridge (Step D) needs concrete implementation details for
  how `synth.liberty.cell` annotations are populated by the mapper.
  **Update (2026-03):** `synth-annotate-techlib` pass now bridges Liberty
  metadata to `hw.techlib.info` for TechMapper; see `TechMapping-Roadmap.md`.
- Advanced STA features are still pending:
  - multi-clock domain constraints and CDC handling
  - MCMM (multi-corner/multi-mode)
  - incremental re-analysis after local netlist edits
  - parallel path-query execution (documented as deferred)

### Practical Readiness Summary

- `TimingAnalysis` is functionally usable today for graph construction,
  arrival/RAT/slack reporting, and K-worst path reporting in current synth
  flows.
- Signoff-grade NLDM/CCS accuracy remains a roadmap item; the primary
  remaining gap is golden-reference validation rather than missing infrastructure.

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

### 1.5. Rise/Fall Edge-Aware Timing (Dual-Edge Analysis)

**Status (2026-03): Not started. This is a major correctness gap.**

The timing graph currently tracks a single arrival time per node and a single
delay per arc. The `timingSense` attribute from Liberty (`positive_unate`,
`negative_unate`, `non_unate`) is imported but **never consumed** during
delay computation. This means:

- An inverter (`negative_unate`) reports the same delay regardless of input
  transition direction.
- Chains of inverters do not alternate between `cellRise` and `cellFall`
  delays as they should in reality.
- The wrong transition table may be used for delay and slew lookup
  (currently `cellRise` is tried first as a heuristic fallback).

#### Phase 1: Dual-Edge Data Structures

Extend `TimingNode` and `TimingArc`:

```cpp
// TimingNode additions:
int64_t arrivalRise = 0;   // arrival time for rising output transition
int64_t arrivalFall = 0;   // arrival time for falling output transition
double slewRise = 0.0;     // output slew when rising
double slewFall = 0.0;     // output slew when falling

// TimingArc additions:
int64_t delayRise;         // delay when output rises (from cellRise table)
int64_t delayFall;         // delay when output falls (from cellFall table)
enum TimingSense { PositiveUnate, NegativeUnate, NonUnate };
TimingSense timingSense;
```

#### Phase 2: Dual-Edge Propagation Rules

During forward arrival propagation, apply timing-sense-aware rules:

| Timing Sense     | Input Rise →            | Input Fall →            |
|------------------|-------------------------|-------------------------|
| positive_unate   | Output Rise (cellRise)  | Output Fall (cellFall)  |
| negative_unate   | Output Fall (cellFall)  | Output Rise (cellRise)  |
| non_unate        | Both (use worst-case)   | Both (use worst-case)   |

Pseudocode:
```
if timingSense == positive_unate:
    arrivalRise[to] = max(arrivalRise[to], arrivalRise[from] + cellRiseDelay)
    arrivalFall[to] = max(arrivalFall[to], arrivalFall[from] + cellFallDelay)
    slewRise[to] from riseTransition table
    slewFall[to] from fallTransition table
elif timingSense == negative_unate:
    arrivalFall[to] = max(arrivalFall[to], arrivalRise[from] + cellFallDelay)
    arrivalRise[to] = max(arrivalRise[to], arrivalFall[from] + cellRiseDelay)
    slewFall[to] from fallTransition table (driven by rising input)
    slewRise[to] from riseTransition table (driven by falling input)
elif timingSense == non_unate:
    worstInput = max(arrivalRise[from], arrivalFall[from])
    arrivalRise[to] = max(arrivalRise[to], worstInput + cellRiseDelay)
    arrivalFall[to] = max(arrivalFall[to], worstInput + cellFallDelay)
```

At endpoints, worst-case arrival = `max(arrivalRise, arrivalFall)`.

Start points (input ports) initialize both `arrivalRise` and `arrivalFall`
to the port arrival time.

#### Phase 3: Dual-Edge Slew Convergence

The iterative slew convergence loop (Step C) must now converge two slew
values per node. Input slew for delay lookup is selected based on timing
sense and transition direction:
- For `positive_unate` cellRise lookup: use `slewRise` from driving node
- For `negative_unate` cellRise lookup: use `slewFall` from driving node

#### Phase 4: Transition-Aware Path Reporting

Update path tracing to annotate each arc with its transition:
```
Path 1: delay = 150ps  slack = 0
  Startpoint: a[0] (input port, rise)
  a[0] (rise) --[inv, neg_unate]--> u0/Y (fall, +30ps)
  u0/Y (fall) --[nand, neg_unate]--> u1/Y (rise, +45ps)
  u1/Y (rise) --> y[0]
```

#### Files to Modify

| File | Change |
|------|--------|
| `TimingGraph.h` | Add rise/fall arrivals/slew to `TimingNode`, timing sense to `TimingArc` |
| `TimingGraph.cpp` | Set timing sense from NLDM arc metadata during graph construction |
| `DelayModel.h/cpp` | Separate `computeDelay` into rise/fall variants, or return both |
| `ArrivalAnalysis.cpp` | Dual-edge forward propagation |
| `RequiredTimeAnalysis.cpp` | Dual-edge backward propagation |
| `PathEnumerator.cpp` | Track transition along path |
| `PrintTimingAnalysis.cpp` | Report transition annotations |

#### Interaction with CCS

CCS pilot waveform selection already has partial edge awareness (preferring
rise/fall current table based on input waveform direction). Dual-edge
tracking provides the correct input transition context that CCS needs,
making the two features complementary.

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

**Status update (2026-03): Started.**

- Added `CCSPilotDelayModel` that reuses NLDM scalar delay/load behavior and
  enables waveform propagation via `computeOutputWaveform(...)` for initial
  CCS-plumbing validation.
- Added timing-flow model selection via module attribute
  `synth.timing.model = "ccs-pilot"` in timing report pass flow.
- Added unit and e2e coverage for CCS pilot report/model activation.
- Added optional waveform detail reporting (`--show-waveform-details`) for
  `circt-synth` / `circt-sta` timing reports when waveform-capable models are
  active.

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
- Iterations now feed previous-sweep slew hints into load computation
  (`DelayModel::getInputCapacitance`) so effective-capacitance style models can
  participate in convergence.
- The last iteration count is tracked and emitted in timing reports as
  `Arrival Iterations` for observability.
- Convergence status is now emitted in timing reports as `Slew Converged`.
- Final-iteration slew residual is now emitted as `Max Slew Delta` to aid
  convergence tuning for load/slew-coupled models.
- Load-slew hint updates now support damping (`slewHintDamping`) to improve
  stability for strongly coupled models.
- Optional adaptive damping policy (`adaptiveSlewHintDampingMode` with
  disabled/conservative/aggressive) adjusts applied damping from
  iteration-to-iteration using residual trend.
- Reports now include `Adaptive Slew Damping Mode`, `Applied Slew Hint
  Damping`, and `Slew Delta Trend`.
- Reports optionally include a per-iteration convergence table (`Iter | Max
  Slew Delta`) when `emitSlewConvergenceTable` is enabled.
- Convergence table now also reports per-iteration applied damping to make
  adaptive policy behavior observable.
- Relative convergence tolerance (`slewConvergenceRelativeEpsilon`) is now
  supported, with report diagnostics for normalized residual.
- `synth-print-timing-analysis` / `circt-synth` / `circt-sta` now expose
  convergence-tuning knobs (max iterations, absolute/relative epsilon,
  damping, adaptive mode) for direct report-flow experimentation.
- End-to-end report tests now validate these tuning controls for both
  `circt-synth` and `circt-sta`.
- Timing reporting capability has been removed from `circt-synth`; report flows
  now run as `circt-synth ... -o - | circt-sta ...`.
- Reports now classify convergence trend (`Slew Trend Class`) as
  decreasing/increasing/flat/oscillating for quick diagnostics.
- Reports now include `Slew Reduction Ratio` (final residual / first residual)
  to quantify convergence speed at a glance.
- Reports now include `Slew Advice` to suggest next tuning action based on
  convergence status and trend class.
- NLDM timing-report flow now seeds initial slew from imported Liberty
  `default_input_transition` when available.
- Remaining work: evaluate adaptive damping/tolerance heuristics for models
  where effective load depends on slew/waveform state, including CCS waveform
  coupling.

**Review note (2026-03):** The convergence diagnostics surface has grown
significantly (~15 distinct report fields). Consider gating verbose
diagnostics (per-iteration table, trend class, reduction ratio, advice)
behind a `--verbose-convergence` flag and keeping the default report lean
(iterations, converged, max delta only). This reduces noise for typical
users while preserving full observability for tuning.

**Files affected:** `TimingAnalysis.cpp`

### Step D: Technology Mapping Bridge

To use NLDM, the netlist must be mapped to specific Liberty cells. This bridges the synthesis flow with timing:

- After technology mapping (e.g., in the Synth dialect transforms), annotate operations with their Liberty cell name.
- `NLDMDelayModel` reads this annotation to look up the correct cell.
- Alternative: maintain a mapping table from MLIR op types to Liberty cells.

**Status (2026-03): Not started — needs concrete design.**

The current roadmap assumes `synth.liberty.cell` attributes exist on mapped
operations, but the technology mapper does not yet emit them. Key questions
to resolve:

- Where in the synthesis pipeline does cell selection happen (before or after
  AIG optimization)?
- How are multi-output cells handled (e.g., full adder with sum + carry)?
- Should unmapped operations (e.g., pre-mapping AIG nodes) fall back to
  `AIGLevelDelayModel` automatically, or is mapping a hard prerequisite for
  NLDM?

Without this bridge, NLDM/CCS models can only be exercised on hand-annotated
test IR, not on real synthesis flows.

### Step E: Golden-Reference Validation

**Status (2026-03): Not started.**

NLDM and CCS accuracy claims require comparison against a reference STA tool.
Without this, numerical correctness is asserted only by internal unit tests
with hand-computed expectations.

**Approach:**

1. Select a small but representative benchmark set (e.g., 3-5 mapped designs
   at a few hundred gates using a public Liberty library like ASAP7 or
   FreePDK45).
2. Run the same mapped netlist through OpenSTA and `circt-sta`, both using the
   same Liberty file.
3. Compare per-path arrival times, slews, and slack values. Acceptable
   tolerance: <1% for NLDM, <2% for CCS pilot vs. NLDM-reference.
4. Automate the comparison as a lit test or script under
   `test/Dialect/Synth/Timing/golden/`.

**Prerequisites:**
- Technology mapping bridge (Step D) to produce mapped netlists, OR
  hand-crafted mapped IR for initial validation.
- An OpenSTA installation available in CI (or run as a manual validation step
  initially).

**Benefits:**
- Catches interpolation bugs, unit-scaling errors, and arc-resolution mistakes
  that unit tests with synthetic data cannot.
- Provides confidence for users evaluating `circt-sta` for real flows.

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

### CCS Pilot Graduation Decision

**Review note (2026-03):** The CCS pilot has grown well beyond plumbing
validation — it now includes waveform-derived delay, receiver-cap handling,
mixed NLDM/CCS delegation, load-aware stretching, and waveform-coupled
convergence. Consider either:

- **Graduate to `CCSDelayModel`**: rename and treat it as the real CCS
  implementation, removing the "pilot" qualifier and cleaning up any
  scaffolding that was meant to be temporary.
- **Reset scope**: if the pilot approximations (stretch-based waveform
  shaping) are too coarse for production CCS, freeze the pilot as-is and
  build a proper `CCSDelayModel` with real current-source simulation from
  scratch, reusing only the Liberty parsing and arc resolution.

The current in-between state risks accumulating technical debt in a
component labeled as temporary.

### CCS Remaining Milestones (2026-03)

From the current `CCSPilotDelayModel` state, the remaining major milestones are:

1. Parse CCS-specific Liberty data (current source and receiver data), reusing
   existing typed metadata plumbing.
2. Replace pilot waveform generation with receiver/load-aware waveform solving
   and threshold extraction.
3. Integrate waveform-coupled convergence controls (damping/tolerance heuristics)
   tuned for CCS behavior.
4. Add dedicated CCS e2e validation suite and mixed NLDM/CCS delegation policy.

**Status update (2026-03): Milestone 1 started.**

**Current milestone snapshot (latest):**

- **Milestone 1 (CCS data parsing):** in progress
  - done: typed pilot arc import/lookup + typed receiver-cap import/consumption
  - next: nested `vector(ccs_template)` parsing (`index_3`, `reference_time`)
- **Milestone 2 (receiver/load-aware waveform solving):** in progress
  - done: load-aware stretch, receiver-cap-aware effective stretch,
    waveform-derived `t50` / `slew10-90`
  - next: richer waveform solving beyond pilot stretch approximation
- **Milestone 3 (waveform-coupled convergence):** initial implementation landed
  - done: waveform-coupled convergence heuristics + diagnostics
  - next: tune policy for mixed-cell CCS/NLDM designs
- **Milestone 4 (CCS validation + mixed policy):** substantial progress
  - done: dedicated CCS e2e scenarios and mixed per-cell delegation
  - next: broaden corner/scenario matrix and tighten golden expectations

- `import-liberty` now emits typed CCS pilot arc metadata
  (`#synth.ccs_pilot_arc`) on output pins as `synth.ccs.pilot.arcs` when
  `output_current_rise` / `output_current_fall` tables are present.
- `LibertyLibrary` now supports typed CCS pilot arc lookup by pin name/index.
- `CCSPilotDelayModel` now consumes typed CCS pilot waveform metadata before
  falling back to synthetic two-point ramp generation.
- Regression coverage now includes ImportLiberty CCS pilot attr emission and
  CCS pilot waveform consumption in timing unit tests.
- CCS pilot waveform selection now prefers fall-vs-rise current tables based on
  input waveform edge direction.
- Added unit coverage for both rising and falling edge waveform selection in
  CCS pilot propagation.
- CCS pilot waveform timing now applies load-dependent stretching to begin
  approximating receiver/load-aware waveform behavior.
- Waveform report details now include extracted `t50` delay proxy and
  `slew10-90` transition proxy per arc.
- CCS pilot now supports opt-in waveform-derived delay
  (`synth.ccs.pilot.waveform_delay`) using extracted `t50`, while preserving
  NLDM-delegate delay as the default fallback behavior.
- Added e2e timing-report coverage for waveform-derived CCS delay mode
  (`ccs-pilot-waveform-delay-report.mlir`) in both `circt-sta` direct and
  `circt-synth | circt-sta` pipeline flows.
- Added multi-input CCS pilot coverage to validate pin-specific arc selection
  and waveform-derived delay differences.
- Added mixed NLDM/CCS-pilot delegation mode (`mixed-ccs-pilot`) with
  per-cell policy (`synth.ccs.pilot.cells`) and e2e coverage.
- Added waveform-coupled convergence heuristics (auto relative epsilon,
  damping cap, minimum iterations) for waveform-capable models, with
  diagnostics for effective settings.
- CCS e2e suite now includes waveform-delay, multi-input arc asymmetry, and
  mixed-policy critical-path selection scenarios.
- Started receiver-data handling: ImportLiberty now maps
  `receiver_capacitance{1,2}_{rise,fall}` into typed CCS pilot receiver
  metadata, and CCS pilot waveform shaping consumes this metadata.

**Milestone 1 next parser target (ASAP-style CCS vectors):**

- Landed: nested `vector(ccs_template)` parsing for
  `output_current_rise/fall` (index_1/index_2/index_3/values/reference_time)
  into typed CCS pilot arc metadata.
- Landed: vector-set interpolation by `(inputSlew, outputLoad)` in CCS pilot
  waveform decode, preferring structured bilinear interpolation on regular
  selector grids, then axis-linear interpolation on sparse row/column slices,
  and finally inverse-distance blend, with interpolated `reference_time`
  applied as waveform time offset.
- Landed: template-aware blending for mismatched `index_3` sample grids,
  resampling by time-axis interpolation before value blending.
- Remaining parser follow-up: richer template semantics and policy fidelity
  beyond pilot interpolation.

---

## Suggested Priority Order

### Original NLDM critical path (items 1-5): largely complete

| Priority | Enhancement | Status |
|----------|------------|--------|
| ~~1~~ | ~~Slew propagation in ArrivalAnalysis~~ | Done |
| ~~2~~ | ~~Output load computation~~ | Done |
| ~~3~~ | ~~Liberty data bridge (reuse import-liberty)~~ | Done |
| ~~4~~ | ~~`NLDMDelayModel` with bilinear interpolation~~ | Done (output-slew table wiring remaining) |
| ~~5~~ | ~~Iterative slew convergence loop~~ | Done |

### Updated priority order (2026-03)

With the NLDM infrastructure substantially in place, the priority shifts
toward validation, practical usability, and CCS maturation.

| Priority | Enhancement | Unlocks |
|----------|------------|---------|
| 1 | Rise/fall edge-aware timing (Section 1.5) | Correct delay through inverting cells |
| 2 | Finish NLDM output-slew transition-table wiring | Complete NLDM accuracy |
| 3 | Technology mapping bridge (Step D) | Real synthesis-to-STA flow |
| 4 | Golden-reference validation against OpenSTA (Step E) | Confidence in numerical correctness |
| 5 | Clock domain awareness | Multi-clock designs |
| 6 | CCS pilot graduation decision + real CCS model | Advanced node accuracy |
| 7 | Convergence diagnostic cleanup (verbose flag) | Cleaner default reports |
| 8 | Multi-corner support | Signoff-quality analysis |
| 9 | Incremental analysis | ECO flows |
| 10 | Parallelization | Performance at scale |

Item 1 (rise/fall awareness) is now the top priority — without it, delay
through any inverting cell is incorrect, and slew table selection is
heuristic. Items 2-4 are the critical path for making NLDM practically
usable and trustworthy. Item 5 (clock domains) is independently high-value
since multi-clock designs are the norm. Items 6-10 can be parallelized.
