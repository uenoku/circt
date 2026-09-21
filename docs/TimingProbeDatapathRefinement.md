# Timing-probe-driven datapath refinement

## Status

This document proposes a timing refinement loop that uses
`synth.timing_probe` to carry endpoint requirements from sequential timing
boundaries back into datapath lowering. Phases 1-5 are implemented
(opt-in `--datapath-refinement`, default off); Phase 6 model and
performance improvements remain future work.

The proposed replacement of external report scraping with a backend-neutral
STA library and an initial OpenTimer backend is described in
[Static timing analysis library integration for `circt-synth`](StaticTimingAnalysisLibraryIntegration.md).

What was already implemented is the probe mechanism and its first consumer:
post-map high-fanout buffering guided by probes and STA feedback, described in
[Timing-driven mapping](TimingDrivenMapping.md) ("Timing-guided synthesis:
probes + STA feedback"). That document is the status authority for landed
probe work; this document covers the pre-map arithmetic consumer.

Phase 1 (early probes + explicit cleanup) has landed:

- `synth-derive-probes` runs once right after sequential mapping (decoupled
  from `--buffer-high-fanout`) and is idempotent: D inputs already driven by
  a probe are skipped, so repeated invocation never nests probes.
- probes ride the lowering/mapping pipeline as transparent pass-throughs
  (Comb -> Datapath, `datapath-reduce-delay`, Datapath -> Comb,
  Comb -> Synth, technology mapping all preserve them).
- `synth-strip-timing-probes` is the explicit lifetime terminator; it runs
  after buffering whenever probes could exist and is a no-op otherwise.
  Default-off pipelines add no passes and are output-identical.
- `synth.timing_probe` rejects a negative `required_time` (the verifier
  previously compared unsigned storage against zero, accepting everything).
- tests: `strip-probes.mlir`, `probe-early-pipeline.mlir`, derive-twice
  idempotency in `derive-probes.mlir`, and end-to-end
  `circt-synth/probe-early-buffer.mlir` (early derive -> buffers consume ->
  no leak, plus a strip-only mode).

Phases 2-5 (probe-guided selective refinement loop) have landed:

- shared timing-feedback support (`Synth/Analysis/TimingFeedback`): parses
  the versioned `circt.synth.timing-feedback.v1` schema (s/ms/us/ns/ps)
  and the legacy scoreboard violator list, validates (schema, units,
  finiteness, slack consistency, top match), converts once to picoseconds
  with checked rounding, and resolves hierarchical endpoints/launches
  (ambiguous and missing names fail closed).
- `TimingPressureAnalysis` (`Synth/Analysis`): module-local value-level
  structural arrivals (levels) plus backward required levels, with
  conservative shared-module hierarchical summaries (arrivals maximize,
  requirements minimize over instance contexts; cycles stay unknown).
  Structural slack is non-positive by construction (zero on critical
  paths) and is a ranking score only, never physical slack. Reachability
  (cone membership) is delay-independent, so feedback-violating cones are
  selected even where delay modeling gives up.
- refinement regions (`Synth/Analysis/RefinementRegions`): deterministic
  datapath components (`@Module:preorder`), probe relation, worst slack,
  violation flags, and endpoint strings. Selection order: explicit policy
  manifest > STA-violating cone > structural pressure within the margin.
  Selected policies are recorded as transient `datapath.refine.policy`
  attributes (member datapath ops plus reached comb add/compare roots)
  and in a versioned JSON decision report.
- `synth-discover-refinement-regions` (ModuleOp): selection + recording +
  report; `datapath-reduce-delay` pressure gating
  (`--pressure-margin-levels`, explicit records always win, continuation
  inheritance, legacy behavior preserved when unselected);
  `convert-datapath-to-comb` per-operation Booth/compress-to-add/Wallace
  overrides (disjoint conditions; no-attribute behavior unchanged).
- `synth-import-timing-feedback` (ModuleOp): parses feedback, resolves
  endpoints, creates or tightens (minimum) endpoint probes, reports
  resolved/unresolved/invalid.
- `circt-synth --datapath-refinement`: derive -> import -> CombToDatapath
  -> discover -> gated lowering -> map -> buffer/strip, all opt-in.
  `--timing-feedback` without a consumer warns and leaves lowering
  unchanged (probes still stripped; no leak).
- driver (`--datapath-retry` + `--datapath-refinement`): round-0 STAjudged
  baseline with report, per-path feedback extraction, ranked single-region
  manifests (priors carried), bounded rounds (default 2, at most 4
  candidates each, `--datapath-retry` total budget), whole-design
  objective (target met -> min area; else max WNS, then TNS, violators,
  area; baseline wins ties), signature cycle detection, minimum-
  improvement and target-met termination.
- tests: `discover-regions/hierarchical/feedback/manifest/determinism`,
  `import-feedback` (versioned units, legacy, strict/permissive, top
  mismatch), `datapath-reduce-delay-pressure` (selective proof),
  `refine-policies` (per-op Booth/compress-to-add),
  `circt-synth/datapath-refine` (fake-STA BEST + no-leak output) and
  `datapath-refine-noimprove` (baseline wins ties, emitted byte-exact).
- fixed along the way: `synth.timing_probe` negative-time verifier
  (unsigned comparison), `synth-derive-probes` idempotency, QN-only flop
  launches (ASAP7 DFFHQNx1) in pressure analysis, feedback launch
  resolution, and guided buffering, plus STA path-record field order
  (arrival-before-slack) in the retry driver.

Deliberate MVP deviations from the proposal (documented, not accidental):

- pressure is structural levels (`--datapath-pressure-margin-levels`, not
  picoseconds); Liberty-calibrated arrivals are Phase 6.
- the discovery pass is named `synth-discover-refinement-regions`
  (SynthTransforms library naming).
- per-round candidate cap is fixed at 4 within the `--datapath-retry`
  total budget (no separate `--datapath-refinement-candidates`).
- area-recovery rounds after the target is met are not attempted; the loop
  stops (bounded effort).
- decided regions are not re-targeted unless currently violated.

The recommended first implementation is deliberately incremental:

1. derive probes before datapath lowering instead of after technology mapping;
2. propagate required times backward and identify timing-pressured datapath
   cones;
3. apply existing delay-oriented datapath transformations only in those cones;
4. judge the result with OpenSTA; and
5. extend the existing `--datapath-retry` driver to retry probe-selected
   structural variants without allowing a baseline regression.

Later stages can add per-datapath-region policies, per-bit timing, richer delay
models, and in-process checkpointing. These are useful refinements, but they
are not prerequisites for proving the approach.

## Executive summary

The current synthesis flow has two disconnected timing mechanisms:

- datapath lowering has a forward, arrival-oriented heuristic; and
- `synth.timing_probe` records endpoint required times, but probes are inserted
  only after datapath lowering and are consumed only by high-fanout buffering.

Consequently, a flop input can be known to violate timing without that
information affecting multiplier construction, compressor-tree structure, or
the area-increasing transformations in `datapath-reduce-delay`.

The proposed flow closes that gap:

```text
Liberty import and sequential technology mapping
                         |
                         v
        derive probes on mapped flop D inputs
                         |
                         v
                 Comb -> Datapath
                         |
                         v
       forward arrival + backward required time
                         |
                         v
       identify timing-pressured datapath regions
                         |
                         v
    locally select/refine datapath implementations
                         |
                         v
         Datapath -> Comb -> Synth -> cells
                         |
                         v
                      OpenSTA
                         |
                         v
       import endpoint timing feedback and retry
```

`synth.timing_probe` remains a transparent timing-constraint boundary. It does
not become a container for every transient result from STA. Forward arrivals,
slack, criticality, candidate scores, and iteration history belong in analyses
or feedback records. The probe carries the stable semantic fact: the value at
its operand must be stable no later than `required_time`.

The central selection rule is:

1. meet the probe requirements;
2. among candidates that meet them, minimize area;
3. if none meets them, maximize worst slack, then minimize total negative
   slack and area; and
4. retain the baseline on ties.

The current implementation uses OpenSTA as the acceptance authority. In-flow
estimates are used to focus the search and construct candidates, not to claim
signoff timing. The proposed backend-neutral replacement is described in
[Static timing analysis library integration for `circt-synth`](StaticTimingAnalysisLibraryIntegration.md).

## Motivation

### Arrival-only datapath decisions are incomplete

`ConvertDatapathToComb` can use `IncrementalLongestPathAnalysis` to pass input
arrival estimates into `CompressorTree::withInputDelays`. This helps the
compressor implementation account for late inputs, but it does not answer the
other half of the timing question: when is the result required?

Without required times, the flow cannot distinguish these cases:

- a 5 ns cone feeding a 10 ns endpoint, where area recovery is appropriate;
- the same cone feeding a 4 ns endpoint, where aggressive restructuring is
  warranted;
- a shared cone feeding both endpoints, where the 4 ns requirement dominates;
  and
- a structurally deep cone that is not on any constrained sequential path.

A global `--synthesis-strategy=timing` or `--force-booth` flag applies one
policy to all of these cases. That is too coarse for mixed designs.

### Existing retry is global rather than diagnostic

The existing `--datapath-retry` implementation evaluates a fixed list of
whole-design variants:

- force Booth lowering;
- lower compressors to variadic adds;
- combine Booth and variadic-add lowering;
- disable timing-aware lowering; and
- disable datapath lowering.

This is safe because the baseline is always included and wins ties, but it
does not use the STA result to decide which datapath logic should change. A
single critical multiplier can cause all multipliers in the design to use a
different lowering policy. Conversely, the driver spends time on datapath
variants even when the worst path contains no datapath logic.

### Timing probes already express the right boundary condition

`synth.timing_probe` is a pure pass-through operation with a required time in
picoseconds:

```mlir
%p = synth.timing_probe %driver : i1 {
  required_time = 10000 : i64
}
```

The result equals the operand, so a probe does not change functionality. A
probe is also intentionally a barrier: canonicalization must not bypass it
before all timing-guided consumers have run. This is the right abstraction for
connecting a timing requirement to an otherwise functional dataflow graph.

The problem is placement and lifecycle. Today probes are derived after
technology mapping, immediately before `synth-buffer-high-fanout`, and the
buffer pass erases them. At that point every datapath operation has already
been lowered.

## Goals

The implementation should:

- expose sequential endpoint requirements before datapath lowering;
- calculate timing pressure using both forward arrival and backward required
  time;
- focus delay-oriented transformations on datapath logic that can affect a
  constrained endpoint;
- support feedback from real STA without making the core passes depend on
  OpenSTA;
- preserve deterministic output for a fixed input, feedback file, and policy;
- preserve the current baseline as a legal candidate and never select a worse
  candidate on the configured objective;
- remain opt-in until timing, area, runtime, and determinism are characterized;
- handle unresolved STA names and unsupported clocking conservatively;
- remove every probe before final output unless an explicit debug option asks
  to retain them; and
- provide enough diagnostics to explain why a region was or was not refined.

## Non-goals

The first implementation does not attempt to provide:

- signoff-quality timing inside MLIR;
- placement-aware wire delay;
- crosstalk, derates, process corners, or multi-mode/multi-corner analysis;
- latch borrowing;
- automatic pipelining or register retiming;
- arbitrary module specialization for every hierarchical timing context;
- a complete arithmetic architecture search;
- simultaneous global optimization of datapath lowering, technology mapping,
  buffering, and physical design; or
- a guarantee that a locally faster structural estimate produces a faster
  placed-and-routed design.

The current implementation uses OpenSTA as the timing oracle for candidate
acceptance. The in-flow model exists to reduce the size of the candidate
space.

## Terminology

- **Probe:** A `synth.timing_probe` operation that associates a required time
  with one dataflow value.
- **Endpoint:** A sequential data input or constrained primary output at which
  timing is checked. The initial implementation focuses on mapped flop D
  inputs.
- **Launch point:** A sequential output or constrained primary input from
  which a timing path begins.
- **Timing pressure:** A measure derived from required time and estimated
  arrival. More-negative slack means greater pressure. Small non-negative
  slack may also be treated as pressured through a configurable margin.
- **Datapath region:** A deterministic connected component of relevant
  `datapath.*` operations and explicitly allowed arithmetic glue. Regions are
  used as the unit of policy selection and retry.
- **Feedback record:** An external record, normally derived from OpenSTA,
  which identifies a path endpoint and reports required time, arrival, and
  slack.
- **Baseline:** The result produced with the user's original synthesis options
  and without a new refinement decision. Every retry iteration retains it as
  a candidate.

## Design invariants

The following invariants should be treated as requirements rather than
implementation suggestions.

### Functional transparency

For every probe:

```text
probe.result == probe.input
```

Erasing a probe must be equivalent to replacing all uses of its result with
its input. No refinement may depend on the runtime presence of a probe in the
emitted circuit.

### Required-time units

`required_time` is a signed 64-bit integer measured in picoseconds. Negative
values remain invalid. Analyses must not silently combine picoseconds with the
unit-delay values currently used by longest-path analysis.

An implementation must either:

- use a delay model that returns picoseconds; or
- explicitly operate in normalized structural units and label the resulting
  slack as a ranking score rather than physical slack.

The MVP may use normalized timing pressure for candidate targeting, but any
message that claims a path meets a time constraint must come from STA or a
picosecond-calibrated model.

### Minimum required time at reconvergence

If a value reaches multiple constrained consumers, the effective required
time is the tightest valid requirement:

```text
R(value) = min(R(user) - delay(value -> user))
```

This must be independent of use-list iteration order.

### Baseline preservation

The unmodified baseline must be evaluated or retained in every retry round.
If all refined candidates fail, time out, cannot be analyzed, or score worse,
the baseline is emitted.

### Determinism

No externally visible choice may depend on pointer order, hash-table iteration
order, process IDs, temporary paths, or OpenSTA report ordering. Stable module
order, operation order, numeric IDs, and explicit tie-breakers must be used.

### Explicit probe lifetime

Every pipeline that derives or imports probes must run a consumer or an
explicit stripping pass. Probe cleanup must not be an accidental side effect
of high-fanout buffering.

### Fail closed on uncertain feedback

An unresolved endpoint, ambiguous name, malformed record, unsupported clock
domain, or inconsistent unit must not cause an arbitrary datapath rewrite.
The implementation should report the condition, omit that record from
refinement, and preserve the baseline.

## Current pipeline and insertion point

The relevant current ordering is approximately:

```text
derive mapping costs
map memories
map sequential elements
  |
  +-- hierarchical Comb lowering
      lower Comb multiplication shape
      Comb -> Datapath
      canonicalize and CSE
      optional datapath-reduce-delay
      lower partial products
      lower compressors
      Comb -> Synth
      Synth optimization
  |
technology mapping
deduplicate instances
insert tie cells
derive timing probes, but only if buffering is enabled
buffer high-fanout nets and erase probes
strip library bodies
emit SDC
```

The landed ordering with `--datapath-refinement` is:

```text
derive mapping costs
map memories
map sequential elements
derive or import endpoint probes                    NEW/EARLIER
  |
  +-- hierarchical Comb lowering
      lower Comb multiplication shape
      Comb -> Datapath
      discover datapath regions                    NEW
      calculate timing pressure                    NEW
      selectively run datapath-reduce-delay        CHANGED
      lower partial products using region policy   CHANGED
      lower compressors using region policy        CHANGED
      Comb -> Synth
      Synth optimization
  |
technology mapping
deduplicate instances
insert tie cells
optional probe-guided high-fanout buffering
strip timing probes                                NEW/EXPLICIT
strip library bodies
emit SDC
```

Sequential mapping is the preferred insertion point because:

- mapped flop instances expose Liberty clock-pin metadata;
- the flop instance name can remain stable through combinational lowering;
- the original comb arithmetic still exists on the D-input side; and
- moving derivation earlier does not require teaching it every source-level
  sequential dialect operation immediately.

## Probe model

### Required attributes

The existing operation remains valid:

```mlir
%p = synth.timing_probe %driver : i1 {
  required_time = 10000 : i64
}
```

No new required attribute is necessary for the MVP.

### Optional stable identity

For diagnostics and feedback round-tripping, an optional stable identity is
recommended:

```mlir
%p = synth.timing_probe %driver : i1 {
  required_time = 9820 : i64,
  probe_id = "@Top/u_core/u_alu/result_reg:D"
}
```

This may initially be an ordinary discardable attribute rather than a field in
the operation definition. If multiple passes need it, promoting it to a typed
operation attribute will improve verification and documentation.

A probe ID should be derived from logical hierarchy and the sequential
endpoint, not from SSA names. It must not depend on temporary-file paths or
pointer values.

### What should not be stored on the probe

The following values are iteration-specific analysis results and should not be
part of the core probe semantics:

- measured arrival time;
- measured slack;
- path rank;
- selected datapath strategy;
- candidate number;
- total negative slack; and
- the complete STA path.

These belong in analyses, policy manifests, or external feedback. Optional
debug attributes may mirror them while dumping an intermediate file, but
consumers must not require those attributes to interpret the operation.

### Probe barriers

Passes between derivation and stripping must obey these rules:

- a probe is zero-delay for timing traversal;
- a probe is transparent for functional equivalence;
- canonicalization and CSE must not replace a probe result with its operand;
- structural matching may treat a probe as a cone boundary unless it has an
  explicit transparent traversal rule;
- a rewrite replacing the probe input must update the input normally; and
- cloning a containing module must clone the probe and its identity according
  to the module-cloning policy.

## Probe derivation and feedback import

### Period-derived probes

The existing `synth-derive-probes` pass should be moved earlier without first
changing its clock-tracing contract:

- trace from a mapped flop's Liberty clock pin;
- accept clocks listed by `--probe-clocks`;
- place a probe on the mapped D input; and
- initialize `required_time` from the configured period.

The period-only value is intentionally optimistic because it does not subtract
setup time, clock-to-Q, uncertainty, or skew. It is adequate for establishing
scope and a first ranking, but not for final acceptance.

### Feedback-derived probes

A new `synth-import-timing-feedback` ModuleOp pass should read timing feedback
after sequential mapping and before hierarchical combinational lowering. It
should:

1. parse and validate every feedback record;
2. resolve the hierarchical endpoint to a mapped flop instance or supported
   primary output;
3. locate or create the probe on that endpoint;
4. update the effective required time with the tightest valid requirement;
5. associate measured slack with the probe in an analysis-side table;
6. aggregate multiple instance contexts conservatively; and
7. report resolved, unresolved, ambiguous, ignored, and aggregated records.

If both a period-derived requirement and an STA requirement exist, the
effective requirement should be the minimum unless the feedback format
explicitly identifies the STA value as a replacement for the same constraint.
The choice should be recorded in a diagnostic report.

### Proposed feedback schema

The existing scoreboard violator JSON should remain accepted. A versioned
schema is recommended before more consumers depend on it:

```json
{
  "schema": "circt.synth.timing-feedback.v1",
  "time_unit": "ns",
  "top": "Top",
  "paths": [
    {
      "startpoint": "u_core/u_alu/a_reg/Q",
      "endpoint": "u_core/u_alu/result_reg/D",
      "required": 9.82,
      "arrival": 10.46,
      "slack": -0.64,
      "path_group": "clk"
    }
  ]
}
```

For compatibility, the importer may also accept the current flat list with
keys such as `required_ns`, `arrival_ns`, and `slack_ns`. Internally all values
should be converted once to picoseconds with checked rounding.

Validation should reject or ignore records with:

- unknown schema versions;
- unsupported time units;
- non-finite numbers;
- inconsistent `required - arrival` versus `slack` beyond a tolerance;
- a top name that does not match the selected top;
- endpoints that resolve to clock pins instead of data pins; or
- endpoints belonging to unsupported latch timing.

### Name stability

The initial design relies on mapped sequential instance names remaining stable
between the baseline run and a retry from the same source. This should be
tested directly. If a sequential name cannot be resolved, the importer should
try an explicit probe ID, then report the record as unresolved.

Combinational gate names are not a stable feedback interface and should not be
used to select datapath regions.

## Timing-pressure analysis

### Analysis result

The proposed `TimingPressureAnalysis` should expose, for each relevant value:

```c++
struct TimingPoint {
  std::optional<int64_t> arrivalPs;
  std::optional<int64_t> requiredPs;
  std::optional<int64_t> slackPs;
  TimingConfidence confidence;
};
```

It should also provide convenience queries:

```c++
bool reachesProbe(Value value);
bool isPressured(Value value, int64_t marginPs);
ArrayRef<TimingProbeOp> getConstrainingProbes(Value value);
std::optional<int64_t> getWorstSlack(Operation *op);
```

`TimingConfidence` distinguishes at least:

- `STAFeedback`: a boundary value came from external STA;
- `LibraryEstimate`: delays are in picoseconds using Liberty-derived data;
- `StructuralEstimate`: delays are normalized/ranking-only; and
- `Unknown`: no meaningful timing value is available.

Candidate selection may use all of these for ranking, but only STA feedback
may be presented as measured post-map slack.

### Forward arrival propagation

Forward propagation estimates the latest input arrival at a value:

```text
A(result) = max(A(operand_i) + D(operand_i -> result))
```

Initial launch arrivals are:

- zero or a configured input delay for primary inputs;
- an estimated clock-to-Q for mapped flop outputs when available;
- zero for the structural MVP if no calibrated model is available; and
- unknown for unsupported sequential constructs.

The existing `IncrementalLongestPathAnalysis` can provide structural arrival
information and should be reused where practical. It must be extended or
wrapped so that probes are transparent and units are explicit.

### Backward required-time propagation

Backward propagation starts at each probe:

```text
R(probe.input) = probe.required_time
```

For each combinational use:

```text
R(operand) = min(R(result) - D(operand -> result))
```

For multiple results or arcs, propagation uses the appropriate operand/result
arc. For a pass-through cast or probe, the arc delay is zero. Unknown or
unsupported operations stop propagation and increment a statistic rather than
inventing a requirement.

The implementation should use a deterministic reverse topological traversal.
If the IR is not topologically ordered, construct the order without mutating
the module solely for analysis.

### Slack and pressure

Where both values are available:

```text
S(value) = R(value) - A(value)
```

A value or operation is pressured when:

```text
S <= pressure_margin
```

The default pressure margin should initially be zero. Experiments may justify
a small positive guard band, but it must be opt-in because enlarging the
critical cone trades area and runtime for robustness.

If only structural arrivals are available, use a normalized pressure score:

```text
P(value) = normalized_arrival(value) / normalized_budget(value)
```

Such a score can rank candidate regions but must not be displayed in
picoseconds or labeled as slack.

### Per-value versus per-bit timing

The MVP may associate one time with an entire integer value, using the worst
bit. This is conservative and matches the scalar `required_time` attribute.

Arithmetic timing is inherently bit-sensitive:

- high carry bits often determine adder delay;
- multiplier columns have different compressor depths;
- extracts and concats constrain only selected bit ranges; and
- truncated products can eliminate nominally slow high-order logic.

A later per-bit representation should use sparse ranges rather than allocating
one record for every bit of every value. The analysis API should avoid baking
in the assumption that a value can only ever have one timing point.

## Hierarchical propagation

### Why hierarchy matters

A probe may sit on a flop D input in a parent module while the arithmetic that
feeds it is inside one or more child modules. A module-local backward walk
would stop at an `hw.instance` result and miss the datapath region.

### MVP policy: conservative shared-module summaries

The first hierarchical implementation should summarize requirements across
module boundaries:

1. propagate a requirement on an instance result to the corresponding child
   output operand;
2. analyze the child module;
3. summarize requirements reaching each child input;
4. propagate those summaries to the matching parent instance operands; and
5. iterate over the instance graph in a deterministic order until summaries
   stabilize.

When one module definition has multiple instances with different timing
contexts, use the tightest requirement for the shared definition. This can
over-optimize noncritical copies but cannot make the shared implementation
less capable of meeting the tight context.

The analysis report should include:

- the number of module definitions receiving more than one context;
- the number of contexts merged per definition;
- the minimum required time selected; and
- the amount of pessimism between the tightest and loosest context.

### Later policy: specialization

If conservative merging causes excessive area, a later stage may clone a
module for a timing-critical subset of instances. Specialization must be a
separate explicit transformation because it affects hierarchy, compile time,
verification cost, and emitted names.

Specialization is not required to validate probe-guided datapath refinement.

### Recursion and unsupported hierarchy

Recursive instantiation is not expected in synthesizable hardware. If the
instance graph contains a cycle, hierarchical timing propagation should emit
an error or mark the cycle unknown rather than failing to converge silently.

Extern modules without timing models terminate propagation. Library cells use
their timing model when available.

## Datapath-region discovery

### Region definition

A region starts at a `datapath` operation and includes connected datapath
operations reached through explicitly transparent arithmetic glue. The exact
v1 transparent set should be small and documented. A safe initial set is:

- `datapath.compress`;
- `datapath.partial_product`;
- `datapath.pos_partial_product`;
- width-preserving or statically understood extracts and concats; and
- constants.

Muxes, general comparisons, module boundaries, sequential elements, probes,
and unknown operations should be region boundaries in v1 unless a
transformation explicitly handles them.

### Stable region identity

Each region needs a deterministic ID for diagnostics and optional policy
manifests. A suitable key is:

```text
<module-symbol>:<preorder-index>:<source-location-hash>
```

The preorder index is assigned using block and operation order after
Comb-to-Datapath canonicalization. The source-location component is diagnostic
and must not be the sole identity because generated or fused operations can
share locations.

Pointer values and textual SSA names are forbidden as persistent IDs.

### Region-to-probe relation

For each region, record:

- probes reachable from its outputs;
- the tightest propagated requirement;
- the worst estimated slack;
- whether any associated feedback path violated timing;
- the affected clock/path groups;
- input arrival estimates; and
- whether traversal stopped at an unsupported boundary.

A region is eligible for refinement if it reaches a valid probe and either:

- has negative estimated slack;
- is named by a violating STA endpoint cone; or
- lies within the configured positive pressure margin.

Regions not reaching a probe must retain baseline behavior.

## Datapath refinement policies

### Policy representation

The implementation should separate region selection from lowering mechanics.
A region policy can be represented internally as:

```c++
enum class DatapathRefinementPolicy {
  Baseline,
  ReduceDelay,
  ForceBooth,
  CompressToAdd,
  ForceBoothAndCompressToAdd,
  FastestKnown
};
```

The exact enum may evolve, but `Baseline` must always exist and unspecified
regions must use it.

An optional JSON manifest allows experiments and retries to select policies
without attaching permanent attributes to source IR:

```json
{
  "schema": "circt.synth.datapath-policy.v1",
  "regions": [
    {
      "id": "@ALU:17:9d8404ef",
      "policy": "reduce-delay"
    }
  ]
}
```

Unknown IDs or policies are errors in strict mode and warnings with baseline
fallback in permissive mode.

### Selective `datapath-reduce-delay`

The smallest useful refinement is to restrict the existing
`datapath-reduce-delay` rewrite patterns to pressured regions.

The pass currently greedily applies its patterns to an entire hardware module.
It should gain either:

- a filter callback supplied by a timing-pressure analysis; or
- an option/analysis that marks eligible root operations before greedy
  rewriting.

The filter must account for newly created operations. New operations inherit
the region and policy of the matched root. A rewrite must not spread into an
unselected neighboring region merely because greedy canonicalization revisits
it.

Useful statistics include:

- eligible regions;
- pressured operations;
- matches considered in pressured regions;
- rewrites applied;
- matches skipped outside pressured regions; and
- estimated area growth.

### Partial-product selection

Current partial-product lowering chooses Booth encoding through a heuristic or
a global force flag. Probe-guided lowering should allow a region policy to
override the heuristic only in selected regions.

The first implementation should not assume Booth is always faster. Existing
measurements show width- and context-dependent tradeoffs. Instead:

- baseline preserves the existing heuristic;
- a `ForceBooth` candidate is generated only for selected regions;
- the result is mapped and judged by STA; and
- the baseline wins if Booth does not improve the configured objective.

Future cost models may decide between AND arrays, Booth arrays, or other
partial-product schemes without an external retry, but measured acceptance is
the safer initial mechanism.

### Compressor lowering

Current compressor lowering supports:

- a complete compressor tree whose scheduling can use input arrival; and
- conversion to a variadic add for downstream optimization.

Probe-guided refinement should add the output side of the constraint. A
compressor builder should eventually accept:

```c++
CompressorTimingContext {
  inputArrival(bit, operand),
  outputRequired(bit, result),
  pressureMargin,
  objective
}
```

The MVP can keep the existing compressor implementation and use the region
policy to choose compressor-tree versus variadic-add candidates. A subsequent
stage can make tree construction deadline-aware and compare multiple legal
reduction schedules before materializing one.

### Candidate scoring inside a region

When an in-flow model can estimate both timing and area, local candidate
selection should use lexicographic scoring:

```text
meets_requirement,
worst_slack,
estimated_area,
structural_change_count,
stable_candidate_order
```

`meets_requirement` sorts true before false. Among meeting candidates, lower
area wins. Among violating candidates, higher worst slack wins, then lower
area. Fewer changes and stable declaration order break exact ties.

This local score only decides which candidates deserve full mapping and STA.
The external loop may reject the locally preferred candidate.

## Closed refinement loop

### MVP subprocess architecture

The current retry driver invokes complete synthesis runs as subprocesses. The
MVP should preserve that isolation and extend the candidate flags so they are
scoped by timing feedback:

```text
iteration 0:
  synthesize baseline
  export Verilog and SDC
  run OpenSTA and emit timing-feedback.json

iteration 1:
  rerun from source with timing-feedback.json
  candidate A: reduce-delay in pressured regions
  candidate B: force Booth in pressured regions
  candidate C: compress-to-add in pressured regions
  candidate D: selected combinations
  run mapping and STA for successful candidates
  select baseline or best candidate

iteration 2..N:
  use feedback from the current winner
  target newly exposed or still-violating regions
  retain the current winner as the round baseline
```

This architecture is slower than cloning an in-memory checkpoint, but it has
important initial advantages:

- candidate crashes and timeouts remain isolated;
- each candidate starts from identical source input;
- existing command reconstruction, export, STA, counting, timeout, and cleanup
  code can be reused;
- debugging artifacts are easy to retain; and
- no mutable analysis state is shared between candidates.

### Candidate scope

The initial scoped candidates may apply one policy to all pressured regions
feeding violating endpoints. Once stable region manifests exist, candidates
should become more selective:

1. rank regions by worst endpoint slack, estimated contribution, and fanout;
2. choose the highest-pressure untried region;
3. vary one policy at that region while keeping accepted prior decisions;
4. optionally evaluate a small batch of independent regions together; and
5. record rejected `(region, policy, feedback-signature)` tuples to prevent
   cycling.

Changing one region at a time makes attribution and rollback straightforward.
It may miss interactions between regions, so a bounded combined candidate can
be considered after all single-region moves stall.

### Whole-design objective

For a clocked design, compare successful candidates using:

1. whether the configured slack target is met;
2. worst negative slack (WNS);
3. total negative slack (TNS), when available;
4. violating endpoint count;
5. area or flattened instance count;
6. runtime, optionally as a final tie-breaker; and
7. stable candidate order.

Recommended selection behavior:

- if any candidate meets the slack target, select minimum area among meeting
  candidates;
- otherwise select maximum WNS;
- use TNS and violating endpoint count to distinguish similar WNS results;
- never select a candidate that is strictly worse than the round baseline on
  the configured lexicographic objective; and
- keep the baseline on exact ties.

For a clockless design, use minimum maximum arrival followed by area, matching
the current retry behavior. Probe-guided sequential refinement is primarily a
clocked-flow feature, so clockless mode should remain secondary.

### Termination

The loop terminates when any of these conditions holds:

- the design meets the target and no tested area-recovery candidate improves
  area;
- no candidate improves WNS by at least `min-improvement-ps`;
- no new pressured datapath region is found;
- all applicable policies for the selected regions have been tried;
- the feedback signature and policy state repeat;
- the maximum iteration count is reached;
- the wall-clock budget is reached; or
- all new candidates fail synthesis, export, or STA.

Reasonable experimental defaults are:

```text
max iterations:       2
max candidates/round: 4
minimum improvement:  10 ps
per-candidate timeout: existing retry timeout
```

All defaults should remain opt-in until measured on representative designs.

### Feedback signature

To prevent cycles, calculate a deterministic signature from:

- resolved endpoint or probe IDs;
- rounded required time and slack;
- selected region policies; and
- relevant synthesis options.

If a signature repeats without a better objective, stop and emit the best
previous candidate.

### Later in-process architecture

After behavior stabilizes, the driver may avoid reparsing and rerunning the
front of the pipeline:

1. build a checkpoint immediately after Comb-to-Datapath canonicalization and
   probe import;
2. clone the checkpoint for each region-policy candidate;
3. run the remaining pipeline independently on each clone;
4. serialize only candidates that reach STA; and
5. retain the winning clone or rerun its deterministic policy once.

This requires careful handling of analysis managers, pass instrumentation,
symbol tables, temporary output, and library modules. It should be justified
by profiles rather than included in the MVP.

## Command-line interface (landed)

### Probe creation and import

```text
--probe-clocks=<comma-separated ports>
--probe-period-ns=<period>
--timing-feedback=<path.json>
--timing-feedback-strict
--timing-feedback-report=<path.json>
```

`--probe-clocks` causes early probe derivation even when
`--buffer-high-fanout` is disabled.

### Probe-guided datapath refinement

```text
--datapath-refinement
--datapath-refinement-policy=<baseline|reduce-delay|booth|compress-to-add|fastest|auto>
--datapath-policy-file=<path.json>
--datapath-policy-strict
--datapath-pressure-margin-levels=<integer>
--datapath-refinement-report=<path.json>
```

Pressure is structural levels, hence `--datapath-pressure-margin-levels`
(not picoseconds); negative disables structural selection.

### Closed-loop control

```text
--datapath-refinement-iterations=<N>   (default 2)
--retry-min-improvement-ps=<ps>        (default 10)
--datapath-retry=<N>                   (total candidate budget)
--retry-slack-target=<ns>
--retry-timeout=<seconds>
--retry-keep-temps
```

Per-round candidates are capped at 4 within the `--datapath-retry` total
budget. Supplying a feedback file without enabling a consumer parses and
reports the file but leaves lowering unchanged, with a clear warning.

## Proposed passes and ownership

### `synth-derive-probes`

Existing pass, changed placement and lifecycle.

Responsibilities:

- create period-derived probes at supported sequential endpoints;
- assign stable probe IDs when requested;
- avoid duplicate probes on repeated invocation; and
- report unsupported clocks and endpoints.

It should not parse OpenSTA reports or select datapath policies.

### `synth-import-timing-feedback`

New ModuleOp pass.

Responsibilities:

- parse the versioned feedback format and legacy violator list;
- resolve hierarchical endpoints;
- create or update endpoint probes;
- convert time units;
- aggregate shared-module contexts; and
- produce resolution diagnostics.

It should not rewrite arithmetic.

### `synth-discover-refinement-regions` (landed)

Implemented as a standalone ModuleOp pass (named with the `synth-` prefix
per library style) backed by `Synth/Analysis/RefinementRegions`.

Responsibilities (all landed):

- partition relevant datapath operations into stable regions;
- connect regions to reachable probes;
- calculate pressure summaries;
- import explicit region policies; and
- record selected policies as transient attributes plus a JSON report.

Region IDs survive passes that need them via temporary discardable
attributes removed with the probes.

### `datapath-reduce-delay`

Existing pass, extended with eligibility filtering.

Responsibilities:

- preserve current whole-module behavior when no filter is supplied;
- apply rewrites only to selected regions in refinement mode; and
- preserve/inherit region identity across rewrites.

### `convert-datapath-to-comb`

Existing pass, extended to query region policy and timing context.

Responsibilities:

- retain existing global options as defaults;
- allow a selected region to override partial-product lowering;
- allow a selected region to override compressor lowering; and
- pass input arrival and, later, output required times to compressor
  construction.

### `synth-strip-timing-probes`

New explicit cleanup pass.

Responsibilities:

- replace each probe result with its input;
- erase the probe;
- remove temporary refinement attributes if configured;
- report the number of stripped probes; and
- optionally fail if an unsupported probe use prevents cleanup.

This pass should run whether or not high-fanout buffering is enabled.

### Retry driver

The driver owns:

- candidate enumeration;
- subprocess execution and timeout;
- Verilog/SDC export;
- OpenSTA invocation;
- feedback generation;
- whole-design objective comparison;
- iteration state and cycle detection;
- temporary artifact retention; and
- final winner emission.

Core IR passes must remain usable from `circt-opt` without invoking OpenSTA.

## Suggested source layout

The exact organization can follow reviewer preference, but a likely layout is:

```text
include/circt/Dialect/Synth/Analysis/TimingPressureAnalysis.h
lib/Dialect/Synth/Analysis/TimingPressureAnalysis.cpp

include/circt/Dialect/Synth/Transforms/SynthPasses.td
lib/Dialect/Synth/Transforms/ImportTimingFeedback.cpp
lib/Dialect/Synth/Transforms/StripTimingProbes.cpp

include/circt/Dialect/Datapath/DatapathPasses.td
lib/Dialect/Datapath/Transforms/ReduceDelay.cpp

lib/Conversion/DatapathToComb/DatapathToComb.cpp

tools/circt-synth/DatapathRetry.cpp
tools/circt-synth/DatapathRetry.h
tools/circt-synth/circt-synth.cpp
```

If feedback parsing is shared by buffering and datapath refinement, factor the
schema and parser into a small Synth support component rather than maintaining
two JSON parsers.

## Diagnostics and statistics

Probe-guided optimization is difficult to trust without an explanation of its
decisions. At minimum, collect the following.

### Probe statistics

```text
probes-derived
probes-imported
probes-updated
probes-duplicated-skipped
probes-stripped
probe-endpoints-unsupported
```

### Feedback statistics

```text
feedback-records
feedback-resolved
feedback-unresolved
feedback-ambiguous
feedback-invalid
feedback-contexts-merged
```

### Analysis statistics

```text
values-with-required-time
values-with-arrival
values-with-slack
unknown-delay-boundaries
hierarchical-boundaries-crossed
shared-module-contexts
```

### Region statistics

```text
datapath-regions
regions-reaching-probes
regions-pressured
regions-refined
regions-unresolved
region-policies-by-kind
```

### Retry statistics

```text
iterations
candidates-attempted
candidates-succeeded
candidates-timed-out
candidates-rejected
baseline-retained
wns-improvement-ps
tns-improvement-ps
area-delta
```

### Decision report

An optional JSON report should make each selected or skipped region auditable:

```json
{
  "region": "@ALU:17:9d8404ef",
  "probes": ["@Top/u_core/u_alu/result_reg:D"],
  "worst_slack_ps": -640,
  "confidence": "STAFeedback",
  "baseline_policy": "baseline",
  "selected_policy": "reduce-delay",
  "reason": "violating endpoint cone",
  "estimated_area_delta": 12
}
```

Debug logs must avoid dumping the full path graph by default on large designs.

## Error handling

### Malformed feedback

In strict mode, malformed JSON, an unsupported schema, invalid units, or an
inconsistent numeric record fails the pass. In permissive mode, invalid
records are diagnosed and ignored.

### Missing OpenSTA

If closed-loop refinement was explicitly requested and OpenSTA cannot be
executed, the driver should fail clearly rather than silently treating a
structural estimate as STA. A standalone probe-guided pass run from
`circt-opt` does not require OpenSTA.

### Candidate failure

A candidate that fails synthesis, export, STA, or resource counting is marked
invalid and excluded. Other candidates and the baseline remain eligible.

### Unresolved endpoint

An unresolved feedback endpoint produces no rewrite. The report must retain
the original endpoint string and the reason resolution failed.

### Probe leakage

The normal final pipeline should fail a postcondition check if any probe
survives after the stripping point. A debug option may intentionally stop
before stripping, but final Verilog export must not depend on that mode.

## Testing strategy

### Operation and lifecycle tests

Extend `test/Dialect/Synth/probe.mlir` to cover:

- valid integer and wider-value probes;
- rejection of a negative required time;
- explicit stripping;
- multiple uses of a probe result;
- nested probes, with the tightest requirement preserved by analysis; and
- optional probe IDs round-tripping through the parser/printer.

Add an early-pipeline test proving that a probe placed on a mapped flop D input
survives:

- Comb-to-Datapath conversion;
- canonicalization;
- CSE;
- `datapath-reduce-delay`;
- Datapath-to-Comb conversion; and
- subsequent Synth lowering until its intended consumer.

### Required-time analysis tests

Use small modules with explicit delays or structural units:

1. one probe and one linear cone;
2. two probes sharing a prefix, with the tighter deadline winning;
3. reconvergent fanout, independent of use-list order;
4. a zero-delay probe boundary;
5. an unsupported operation that terminates propagation;
6. no probe, yielding no required-time result;
7. a relaxed path with positive slack;
8. a violated path with negative slack; and
9. deterministic results after textual operation reallocation.

### Hierarchy tests

Cover:

- probe in a parent with datapath logic in a child;
- probe and datapath logic separated by two module levels;
- one child definition instantiated under two deadlines;
- conservative minimum requirement for shared definitions;
- unresolved extern-module timing;
- multiple top-level clocks; and
- generated/unsupported clock paths remaining unprobed.

### Selective transformation tests

Construct two otherwise identical datapath regions:

- one feeds a violated probe and must be transformed;
- one feeds a relaxed probe and must remain byte-for-byte structurally
  unchanged.

Add cases for:

- `datapath-reduce-delay` filtering;
- local Booth override;
- local compressor-to-add override;
- baseline fallback for an unknown policy;
- stable region IDs; and
- region identity inheritance after a rewrite creates new operations.

### Feedback importer tests

Test both the versioned schema and legacy scoreboard list:

- endpoint resolution;
- ns-to-ps conversion;
- checked rounding;
- missing optional arrival;
- duplicate endpoint records;
- inconsistent slack;
- ambiguous hierarchy;
- missing endpoint;
- top mismatch;
- invalid time unit; and
- strict versus permissive behavior.

### Driver tests

Extend the hermetic fake-STA setup used by `datapath-retry.mlir`:

1. baseline violates at endpoint A;
2. feedback resolves A to a probe;
3. only A's datapath region changes;
4. one candidate improves slack;
5. another candidate improves area but misses the target;
6. the meeting candidate wins;
7. a second iteration either finds endpoint B or terminates;
8. the winning MLIR and SDC are emitted; and
9. timeout/failure candidates do not displace the baseline.

Add a no-improvement case that proves the baseline file is emitted exactly.

### Equivalence tests

Every structural policy needs an existing unit equivalence argument plus
end-to-end LEC on representative arithmetic designs. At minimum:

- multiplier with baseline and forced Booth candidates;
- compressor tree versus variadic-add candidate;
- selectively transformed mixed module; and
- accepted multi-iteration winner versus original input.

### Determinism tests

Run identical refinement commands twice and compare:

- final MLIR;
- selected policy manifest;
- probe/region report;
- candidate ordering; and
- summary statistics, excluding elapsed time and temporary paths.

### Default-regression tests

With all new options disabled:

- existing `circt-synth` output must remain byte-identical;
- no probes may be derived earlier;
- no additional analysis should run;
- compile time should remain within measurement noise; and
- existing global datapath flags must retain their current behavior.

## Experimental evaluation

### Microbenchmarks

Use arithmetic kernels that isolate structural choices:

- unsigned and signed multiplication across several widths;
- squaring, which has a distinct partial-product lowering;
- multiply-add;
- three-operand and many-operand compression;
- truncated products;
- add/compare combinations handled by `datapath-reduce-delay`; and
- two independent arithmetic cones with different endpoint deadlines.

For each benchmark, record:

- baseline and refined WNS/TNS;
- area or flattened cell count;
- selected policy per region;
- number of candidates and iterations;
- synthesis and STA runtime; and
- whether the critical path actually intersects the selected region.

#### Measured results (ASAP7 RVT, real OpenSTA)

Method: ASAP7 RVT CCS libraries (AO/OA/INVBUF/SEQ/SIMPLE) converted
once from 1ps to 1ns units (all time quantities scaled; capacitances
kept) so circt-synth, SDC, Yosys/ABC, and OpenSTA agree on nanoseconds.
IMPORTANT: all `--liberty` files in one run must share a time unit -
`DeriveMappingCost` assumes nanoseconds, so mixed-unit libraries distort
selection (measured 21% area swing on twomul12: 1341 vs 1112 instances
for identical input). Same binary, same libraries, same SDC period per
design; `sta -no_init -exit` with `report_checks -path_delay max`.
Designs are hierarchical (`u_sub` + flops) so feedback endpoints
resolve to real instance paths. Yosys 0.68 reference: `proc/opt/fsm/
memory/techmap`, `dfflibmap`, `abc -liberty` on one merged liberty
(202 unique cells), `stat -liberty` area, same SDC judged by the same
OpenSTA. Yosys `abc -D` timing targets (ps) did not move results past
the default script at this scale (already optimal / not engaging).

| design | period | config | inst | area | WNS (ns) | arrival (ns) |
|---|---|---|---|---|---|---|
| muladd8 | 0.22 | yosys | 175 | 20.46 | -0.034 | 0.242 |
| muladd8 | 0.22 | baseline | 241 | 28.10 | -0.0730 | 0.279 |
| muladd8 | 0.22 | probe-scoped booth | 227 | 27.19 | -0.0446 | 0.250 |
| muladd8 | 0.22 | global retry | baseline kept | | | |
| mul16 | 0.40 | yosys | 671 | 71.98 | -0.053 | 0.440 |
| mul16 | 0.40 | baseline | 886 | 98.31 | -0.0535 | 0.439 |
| mul16 | 0.40 | global booth | 750 | 83.69 | -0.0099 | 0.396 |
| mul16 | 0.40 | probe-scoped booth | 750 | 83.69 | -0.0099 | 0.396 |
| twomul12 | 0.50 | yosys | 970 | 99.07 | -0.077 | 0.565 |
| twomul12 | 0.50 | baseline | 1122 | 116.20 | -0.0757 | 0.561 |
| twomul12 | 0.50 | global / probe-scoped retry | 1122 | 116.20 | kept | 0.561 |
| mixed12 | 0.50 | yosys | 649 | 67.58 | -0.078 | 0.566 |
| mixed12 | 0.50 | baseline | 794 | 84.02 | -0.0757 | 0.561 |
| mixed12 | 0.50 | probe-scoped retry | baseline kept (793) | -0.0757 | 0.561 |

Reading the table:

- Yosys+ABC beats CIRCT TechMapper on area by 20-30% at comparable
  timing on all four kernels. That gap lives in the mapper, not the
  refinement layer: refinement improves CIRCT's own baseline where the
  structure allows (muladd8 scoped Booth: +28ps, -14 inst; mul16: +44ps
  with -15% area), matches the global best, and fails closed elsewhere
  (twomul12, mixed12: baseline retained).
- In-process reduce-delay alone changes nothing on these kernels (no
  matching add-chains); movement comes from Booth/compress policies via
  the driver, which is the intended division of labor.
- Single runs take ~13s, full retry runs ~75s (5 candidates x
  synth+export+STA+count on tiny designs); winners are byte-identical
  across repeated driver runs.

Against the success criteria: zero default regressions (lit suites),
deterministic repeats (report diff test plus byte-identical driver
winners), LEC coverage (`datapath-refine-lec.mlir`: selective
reduce-delay, per-op Booth, per-op compress-to-add), baseline fallback
on real STA (twomul12, mixed12), reproducible WNS fixes over the
baseline (muladd8 scoped Booth: +28ps; mul16: +44ps with -15% area,
matching the global best), and a Yosys/ABC reference comparison (below).
System-scale (FIFO/ChipTop) numbers and default-on guidance remain open.

#### Yosys/ABC reference (same SDC, same OpenSTA)

Yosys 0.68 (`proc/opt/fsm/memory/techmap`, `dfflibmap`, `abc -liberty`
on one merged 202-cell liberty, `stat -liberty` area) judges the same
four kernels under identical constraints. `abc -D` timing targets did
not move results past the default script at this scale. Areas are
library square microns on both sides (CIRCT area sums liberty cell
areas over mapped instances).

| design | Yosys inst / area / WNS | CIRCT best | CIRCT best inst / area / WNS |
|---|---|---|---|
| muladd8 | 175 / 20.46 / -0.034 | scoped booth | 227 / 27.19 / -0.0446 |
| mul16 | 671 / 71.98 / -0.053 | scoped booth (= global best) | 750 / 83.69 / -0.0099 |
| twomul12 | 970 / 99.07 / -0.077 | baseline (both retries keep) | 1112 / 115.37 / -0.076 |
| mixed12 | 649 / 67.58 / -0.078 | baseline (scoped keeps) | 778 / 82.93 / -0.076 |

Yosys+ABC is 20-30% smaller at comparable timing on every kernel: the
area gap lives in technology mapping, not in refinement selection.
Refinement's job is selectivity inside CIRCT's flow (improve CIRCT's
baseline where structure allows, with diagnostics and rollback), not
beating ABC. Closing the mapper gap is separate future work.

#### Why muladd8 is heavy: cell-mix analysis

CIRCT muladd8 baseline (241 cells) vs Yosys (175) decomposes as follows
(same function, same library, both deterministic):

| cell family | CIRCT | Yosys | delta |
|---|---|---|---|
| MAJ (full-adder carry) | 0 | 15 | -15 |
| XNOR (sums) | 21 | 45 | -24 |
| NAND | ~0 | 24 | -24 |
| NOR | ~60 | 3 | +57 |
| INV | ~30 | 20 | +10 |
| flops | 16 | 16 | 0 |

Yosys builds full adders from MAJ+XNOR; CIRCT covers the same cones
with NOR/INV/AOI soup. Isolated controls prove the MAJ patterns exist
and win on clean cones: a bare majority maps to MAJx3, a minority to
MAJIxp5, a 3-input add to 7 MAJIxp5, a Wallace-style carry
`(a&b)|((a^b)&c)` to MAJx3, and flopped adders (including QN-flop
variants) to 7 MAJs. Two structures defeat it in context:

- Wallace/compress region: an 8-bit multiplier alone maps with 0 MAJ
  (151 vs 122 cells). The compressor's carry cones are functionally
  MAJ but never selected -- almost certainly an area-flow search gap
  on shared/phase-scattered AIG (no structural-choice accumulation in
  the ABC `dch` sense), since identical functions map cleanly.
- Doubled addends: `add(a,a,c)` maps with 0 MAJ at 81 cells vs
  `add(a,b,c)` with 7 MAJ at 70 cells; Yosys strength-reduces `a+a` to
a shift (free wiring) and maps it in 27 cells / 2.78 area. The
  `add(r0,r0,c)` cone in muladd8 hits exactly this case.

Ruled out by isolation: flop counts (16/16), bit-blast glue folding,
QN-output phases (flopped adders still reach MAJ), timing-vs-area
strategy (identical result), and determinism (5/5 identical runs).
Separately, this analysis exposed a sequential-mapping correctness bug
(QN-only cells got uninverted drivers, inverting every flop output;
fixed with driver compensation plus a `seq-tech-mapper-qn` regression
test). All numbers above are post-fix, functionally correct circuits.

### System designs

The existing FIFO and ChipTop experiments exercise different failure modes:

- FIFO timing has strong high-fanout/broadcast effects, so datapath refinement
  should not claim improvements that are really produced by buffering.
- ChipTop contains logic-depth paths where post-map broadcast buffering has
  little effect, making it a useful test of whether a violating path actually
  intersects an arithmetic region.

Evaluate at least these configurations:

```text
baseline
probe-guided datapath only
high-fanout buffering only
probe-guided datapath + high-fanout buffering
existing global datapath retry
probe-scoped datapath retry
```

Keep clock constraints, Liberty files, mapper effort, and buffer settings
identical across comparisons.

### Success criteria

Before enabling the feature by default, require:

- zero default-output regressions when disabled;
- deterministic repeated runs;
- no functional mismatches in LEC coverage;
- baseline fallback in every failed/no-improvement experiment;
- at least one reproducible case where probe-scoped refinement improves WNS
  or meets a target that the baseline misses;
- lower area than the corresponding whole-design timing variant on a mixed
  design, or a clear runtime/search-space advantage;
- bounded iteration count and candidate runtime; and
- diagnostics that correctly explain non-datapath critical paths.

The feature should remain opt-in if it only reproduces global retry results
with additional complexity.

## Performance and scalability

### Analysis complexity

For a flattened acyclic dataflow graph, forward and backward timing traversal
should be linear in values plus use edges. Hierarchical summaries should avoid
enumerating every full instance path when a shared-module aggregate is enough.

Expected complexity for the conservative shared-module model is roughly:

```text
O(operations + SSA uses + instance edges + summary updates)
```

Summary updates need a convergence bound. With minimum-required-time merging,
requirements move monotonically tighter; an acyclic instance graph permits a
scheduled traversal rather than an unbounded fixed-point loop.

### Memory

Store timing records only for values that are reachable forward from a launch
or backward from a probe. A dense record for every bit in ChipTop-scale IR is
not acceptable. Per-bit timing should use sparse intervals or region-local
storage.

### Candidate explosion

If there are `R` pressured regions and `P` policies, exhaustive search has
`P^R` combinations and is not viable. The loop must remain bounded through:

- pressure ranking;
- one-region-at-a-time changes;
- a fixed candidate count;
- remembered rejected decisions;
- early termination; and
- optional combined candidates only after single-region progress stalls.

### Caching opportunities

The subprocess MVP can cache:

- parsed feedback;
- deterministic region manifests;
- failed `(region, policy)` decisions for the same feedback signature; and
- unchanged front-end artifacts if the tool gains an explicit checkpoint.

Do not add caches whose keys omit synthesis flags, Liberty identity, or timing
constraints.

## Interaction with other timing work

### High-fanout buffering

Datapath refinement and buffering solve different problems:

- datapath refinement changes logic depth and arithmetic structure;
- buffering changes load distribution after mapping (landed v1, see
  [Timing-driven mapping](TimingDrivenMapping.md)).

The same endpoint may need both. Candidate reports should distinguish whether
the critical path is dominated by arithmetic depth, a high-fanout stage, or an
unsupported boundary. The pipeline should run datapath refinement before
mapping and buffering after mapping.

### Load-aware technology mapping

Probe-derived required times can eventually guide technology mapping as well
as datapath lowering (load-aware delay status, see
[Timing-driven mapping](TimingDrivenMapping.md)). That is a separate consumer
of the same timing-pressure
analysis. It should not be bundled into the first datapath patch because
mapper cover iteration and datapath structural search have different rollback
and cost-accounting requirements.

### Multi-output mapping

Datapath refinement may change the half-adder/full-adder patterns visible to
bundle extraction. Measurements should report bundle extraction and mapping
counts when both features are enabled. The refinement objective should be
based on final area/timing, not preservation of any particular intermediate
pattern.

### SDC emission

Probe clocks and emitted SDC clocks must describe the same domains. The driver
should warn if feedback references a path group whose clock was not eligible
for probe derivation. Eventually, probe required times should be derived from
the same constraint model used to emit SDC rather than from parallel command
line options.

### General required-time-driven mapping

The analysis and feedback importer should be written as shared Synth
infrastructure. Datapath refinement is the first pre-map consumer, not a
reason to encode required-time propagation inside Datapath-specific classes.

## Failure modes and mitigations

### Estimated improvement, STA regression

**Cause:** structural or Liberty estimates omit load, slew, reconvergence, or
mapping effects.

**Mitigation:** retain the baseline, run STA on the candidate, and reject the
candidate under the whole-design objective.

### Area growth outside the critical cone

**Cause:** global greedy rewriting or a shared module definition applies a
critical policy to noncritical copies.

**Mitigation:** filter by region, report shared-context merging, and consider
module specialization only after measuring the cost.

### Critical path moves after refinement

**Cause:** fixing one path exposes a second endpoint.

**Mitigation:** iterate with feedback from the current winner and bound the
number of rounds. This is expected behavior, not by itself a failure.

### Oscillation between strategies

**Cause:** candidate A fixes endpoint X but worsens Y, while candidate B does
the reverse.

**Mitigation:** use whole-design WNS/TNS, remember policy and feedback
signatures, require a minimum improvement, and stop on repetition.

### Endpoint names change

**Cause:** a pass renames or replaces sequential instances between runs.

**Mitigation:** derive explicit probe IDs after sequential mapping, test name
stability, and fall back safely when an ID cannot be resolved.

### Probe is optimized away too early

**Cause:** a canonicalization or generic identity fold bypasses the operation.

**Mitigation:** provide no canonicalization that forwards through the probe,
add pipeline survival tests, and strip it only through the explicit cleanup
pass.

### Unit mismatch

**Cause:** combining structural depth, nanoseconds from JSON, and picoseconds
from the operation.

**Mitigation:** convert external values at import, encode confidence/units in
analysis results, and prohibit implicit arithmetic between different units.

### Unsupported timing boundary hides the true cone

**Cause:** generated clocks, latches, black boxes, or unknown operations stop
propagation.

**Mitigation:** report the boundary, leave the affected region at baseline,
and allow explicit user constraints or timing models in later work.

## Alternatives considered

### Continue using global retries only

This is simple and already implemented, but it scales poorly with additional
structural axes and cannot preserve area in relaxed datapath regions. It also
does not explain whether a timing failure is influenced by datapath lowering.

### Store slack directly on every operation

Slack is not an intrinsic operation property. It changes with endpoint
constraints, mapping, loads, and iteration. Materializing it broadly creates
stale attributes and invalidation problems. Analysis-side timing points and a
stable endpoint probe are cleaner.

### Run OpenSTA inside an MLIR pass

This would couple core transformations to an external executable and netlist
export, complicate pass reproducibility, and make `circt-opt` use awkward. The
driver should orchestrate STA; passes should consume a file or analysis.

### Insert probes only after mapping

This preserves the current buffering flow but cannot affect datapath lowering.
Post-map probes remain useful for buffering, but the endpoint constraint must
exist earlier for pre-map consumers.

### Optimize every path reaching a probe

Every synchronous cone normally reaches a probe, so reachability alone is too
broad. Refinement should require measured violation, estimated negative slack,
or an explicit positive margin.

### Begin with per-bit signoff timing

Per-bit, hierarchical, Liberty-calibrated timing is valuable but substantially
larger than needed to test the core hypothesis. Value-level pressure plus STA
acceptance gives a safer incremental route.

## Implementation phases

### Phase 0: measurement and contracts

Before behavior changes:

- add a test that sequential instance names remain stable across existing
  global datapath variants;
- record baseline output hashes with all new flags disabled;
- record current retry WNS, area, and runtime on selected benchmarks;
- define and test the timing-feedback schema; and
- document which passes are allowed to see a probe.

Deliverable: tests and format definitions, with no synthesis change.

### Phase 1: early probes and explicit cleanup

- move period-derived probe placement after sequential mapping;
- decouple it from high-fanout buffering;
- make relevant conversions tolerate/preserve probes;
- add `synth-strip-timing-probes`;
- enforce the no-leak postcondition; and
- prove default-off output parity.

Deliverable: probes can span the datapath pipeline without affecting output.

### Phase 2: timing-pressure analysis

- implement probe-transparent forward/backward traversal;
- start with module-local value-level timing;
- add conservative hierarchical summaries;
- provide region/probe diagnostic reports; and
- characterize analysis runtime on a large design.

Deliverable: the tool identifies which datapath regions feed tight probes,
without rewriting them.

### Phase 3: selective existing transformation

- add eligibility filtering to `datapath-reduce-delay`;
- inherit region identity through rewrites;
- add selective-versus-relaxed unit tests; and
- measure one-pass structural and STA results.

Deliverable: the first probe-driven netlist change, still without automatic
retry.

### Phase 4: feedback import and one-round scoped retry

- import scoreboard/OpenSTA endpoint feedback;
- scope existing variant families to pressured regions;
- reuse candidate subprocesses and baseline-safe selection;
- emit a decision report; and
- add fake-STA integration tests.

Deliverable: one baseline STA run followed by bounded probe-scoped candidates.

### Phase 5: iterative per-region policy

- materialize stable region IDs;
- add policy manifests;
- change one ranked region at a time;
- carry accepted decisions into the next iteration;
- add TNS/violator-count scoring and cycle detection; and
- evaluate combined candidates only when single-region moves stall.

Deliverable: a complete bounded datapath timing refinement loop.

### Phase 6: model and performance improvements

Only after measurements justify them:

- add per-bit timing for arithmetic operations;
- use Liberty-calibrated abstract delays;
- make compressor scheduling deadline-aware;
- add module specialization for divergent instance contexts;
- checkpoint the pre-lowering IR for in-process candidates; and
- share required-time information with technology mapping.

## Patch decomposition

A reviewable patch series could be:

1. **Specify feedback JSON and add parser tests.**
2. **Add explicit probe stripping and lifecycle tests.**
3. **Move optional probe derivation before comb/datapath lowering.**
4. **Add module-local `TimingPressureAnalysis`.**
5. **Add hierarchical required-time summaries.**
6. **Add deterministic datapath-region discovery and reports.**
7. **Pressure-gate `datapath-reduce-delay`.**
8. **Add feedback import at early probes.**
9. **Scope Booth/compressor overrides to selected regions.**
10. **Extend `DatapathRetry` to one feedback-guided round.**
11. **Add iterative region policies and cycle detection.**
12. **Add system measurements and update default-on guidance.**

Each behavior-changing patch should include its unit tests and preserve a
standalone `circt-opt` path where practical.

## Open design questions

Answered by the landed implementation:

4. Feedback `required` is combined with period-derived probes by minimum
   (tightest wins); the importer records created/updated/kept per endpoint.
5. Stable region IDs (`@Module:preorder`) were needed and are implemented;
   manifests round-trip through them and the retry driver ranks by them.

Still open (need measurements on larger designs): 1, 2, 3, 6, 7, 8, 9, 10.
In particular, value-level structural pressure plus STA acceptance is the
implemented ranking; per-bit timing, Liberty calibration, and
deadline-aware compressor scheduling remain Phase 6.

## Recommended MVP

The recommended first end-to-end milestone is narrower than the complete
design, and it has landed (see Status): derive probes after sequential
mapping, preserve them through Comb-to-Datapath, propagate structural
arrival and required times (module-local plus hierarchical summaries),
identify pressured or feedback-violating regions, refine selectively,
judge with OpenSTA, keep the baseline on ties, and strip all probes.

This milestone answers the most important feasibility question: can a probe
select a datapath-local change that improves measured timing without imposing
the corresponding area cost on unrelated arithmetic? If not, richer region
policies and per-bit models should wait until the failure is understood.

## Completion criteria

Status of each criterion (opt-in `--datapath-refinement`, default off):

- derives or imports endpoint probes before datapath lowering: DONE
  (`synth-derive-probes` early + idempotent, `synth-import-timing-feedback`).
- propagates enough timing information to identify pressured datapath
  regions: DONE (`TimingPressureAnalysis` + hierarchical summaries +
  region discovery with decision reports).
- applies at least one datapath structural refinement selectively: DONE
  (gated `datapath-reduce-delay`, per-region Booth/compress-to-add/Wallace).
- evaluates refined candidates with OpenSTA: DONE (retry driver STA judging;
  hermetic fake-STA lit coverage).
- retains the baseline when refinement is not beneficial: DONE (baseline
  wins ties; byte-exact emission proven by `datapath-refine-noimprove`).
- supports a bounded feedback iteration: DONE (ranked single-region rounds
  with priors, signature cycle detection, budget/min-improvement stops).
- emits deterministic decision diagnostics: DONE (reports byte-identical
  across runs; `discover-determinism`).
- removes probes from final output: DONE (buffer/strip terminators;
  `CHECK-NOT` coverage on winner MLIR).
- passes unit, integration, determinism, and equivalence tests: DONE
  (`datapath-refine-lec.mlir` covers selective reduce-delay, per-op
  Booth, and per-op compress-to-add end to end; driver-level multi-round
  winners are not yet LEC-checked against the source).
- documented area, WNS/TNS, and runtime results on microbenchmarks and at
  least one system-scale design: PARTIAL (microbenchmarks with real STA
  plus a Yosys/ABC reference above; system-scale FIFO/ChipTop designs are
external to this repo and remain open).
