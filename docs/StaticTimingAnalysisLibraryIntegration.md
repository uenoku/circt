# Static timing analysis library integration for `circt-synth`

## Status

This is a design proposal. It defines a backend-neutral static timing analysis
(STA) interface and an initial OpenTimer backend for the timing refinement loop
described in
[Timing-probe-driven datapath refinement](TimingProbeDatapathRefinement.md).

The proposed implementation is opt-in until its timing coverage and candidate
ranking have been compared against an independent reference flow. It is an
optimization oracle for synthesis. It is not a signoff timer.

## Decision summary

Integrate STA behind a CIRCT-owned C++ library interface. Use OpenTimer as the
first backend and keep the interface capable of supporting a smaller native
backend later.

The first implementation should:

- build a flattened, bit-level timing graph directly from mapped HW IR;
- pass constraints through typed CIRCT data structures instead of parsing SDC;
- use OpenTimer's programmatic graph and query APIs;
- avoid OpenTimer's Verilog and SDC parsers;
- return WNS, TNS, failing endpoints, and worst paths as structured data;
- identify objects with CIRCT-owned IDs rather than strings recovered from a
  text report;
- run the linked timer inside each existing candidate subprocess, preserving
  the retry driver's timeout and crash containment; and
- keep the current external OpenSTA flow temporarily as an offline reference
  and migration aid, without linking or distributing OpenSTA.

Do not begin by implementing a complete STA engine in CIRCT. A native timer is
reasonable only if the OpenTimer experiment fails a stated coverage, accuracy,
performance, maintenance, or licensing gate.

## Context

### Current refinement loop

The current `--datapath-retry` flow performs these steps for each candidate:

1. invoke a child `circt-synth` process to produce mapped MLIR and SDC;
2. export the mapped design to Verilog;
3. invoke the `sta` executable with Liberty, Verilog, and SDC inputs;
4. parse `report_checks` text into a candidate score and path records;
5. count mapped instances;
6. compare the candidate against the current winner; and
7. convert path records to timing feedback for the next refinement round.

The refinement passes themselves are appropriately independent of the timer.
They consume probes, `TimingFeedback`, `TimingPressureAnalysis`, and refinement
manifests. That separation should remain.

### Problems at the timer boundary

The executable boundary was a useful way to establish the refinement loop, but
the report text is not a reliable API:

- OpenSTA prints explanatory text after `Startpoint:` and `Endpoint:` names.
  The current parser retains the whole line suffix, while feedback resolution
  expects a bare hierarchical pin name.
- A default `report_checks -path_delay max` invocation reports a bounded set of
  paths. Summing those printed path slacks does not produce whole-design TNS or
  a complete failing endpoint count.
- Timing object identity is reconstructed from names. Hierarchy separators,
  escaped identifiers, bus syntax, renamed instances, and display annotations
  can all make resolution fail.
- Verilog export, SDC emission, process startup, and text parsing are repeated
  for every candidate even though the mapped MLIR is already in memory in the
  candidate process.
- Backend capabilities and unsupported constraints are not represented in the
  result, so a numerically valid report can conceal incomplete coverage.

These are interface problems. Improving the structural timing heuristic does
not fix them.

## Goals

The design must:

- provide one typed result model for candidate scoring and feedback generation;
- calculate whole-design WNS, TNS, and failing endpoint count with documented
  aggregation semantics;
- preserve a stable relationship between timer endpoints and CIRCT objects;
- support the max-delay setup analysis needed by the first refinement loop;
- model combinational cell delay, input slew, output load, clock-to-Q, and setup
  checks from NLDM Liberty data;
- elaborate hierarchical HW IR without requiring OpenTimer's Verilog parser;
- reject or explicitly report unsupported constructs and constraints;
- remain optional at build time;
- preserve candidate timeouts and isolation;
- support deterministic JSON diagnostics for tests and experiments; and
- permit an independently implemented backend without changing refinement
  passes or candidate comparison.

## Non-goals for the first implementation

The first implementation does not promise:

- signoff correlation;
- hold or minimum-delay analysis;
- statistical or multi-corner multi-mode analysis;
- signal integrity, crosstalk, or variation models;
- extracted interconnect delay or SPEF;
- propagated clocks, clock-tree latency, or CPPR;
- generated clocks, clock groups, false paths, or multicycle paths;
- latch borrowing;
- asynchronous recovery or removal checks;
- arbitrary user SDC compatibility;
- incremental updates across separately synthesized candidates; or
- timing of partially lowered behavioral IR.

Unsupported features must produce coverage diagnostics. They must not silently
fall back to optimistic zero delay.

## Architecture

### Layers

The integration has four layers:

```text
  datapath refinement driver
              |
              v
      CIRCT TimingOracle API
              |
       +------+------+
       |             |
       v             v
  OpenTimer       future native
   adapter          backend
       |
       v
  OpenTimer core library

  mapped HW IR
       |
       v
  TimingDesignBuilder ---- typed constraints
       |
       v
  flattened TimingDesign
```

The refinement driver owns policy selection, candidate comparison, iteration,
and stopping rules. `TimingDesignBuilder` owns CIRCT semantics and object
identity. A backend owns delay propagation and timing queries.

No backend may attach timing state to MLIR operations. The graph and result are
ordinary C++ objects whose lifetime is bounded by one analysis request.

### Process topology

Library integration does not require putting the timer in the long-lived retry
supervisor. Preserve the current process topology:

```text
parent circt-synth (--datapath-retry)
  |
  +-- candidate child 0
  |     synthesize -> build graph -> run linked timer -> write result JSON
  |
  +-- candidate child 1
  |     synthesize -> build graph -> run linked timer -> write result JSON
  |
  +-- candidate child N
        synthesize -> build graph -> run linked timer -> write result JSON
```

The parent applies the existing wall-clock timeout to each child process. A
timer assertion, excessive memory use, deadlock, or malformed library therefore
invalidates one candidate rather than the entire retry run. The parent reads
only the versioned result JSON and mapped output.

A direct `circt-synth` invocation may also request a timing result. This is
useful for debugging and differential measurement, but core MLIR passes must
not invoke a timer.

### Candidate lifecycle

For each candidate:

1. run the synthesis pipeline through technology mapping and all post-map
   transformations that affect connectivity or load;
2. resolve the requested top module explicitly;
3. load and validate the configured Liberty files;
4. elaborate the mapped hierarchy into `TimingDesign`;
5. translate the synthesis timing options into `TimingConstraints`;
6. check the request against backend capabilities;
7. run max/setup timing;
8. normalize backend values into a `TimingResult`;
9. serialize the result atomically; and
10. write the mapped MLIR candidate.

Timing must run after buffer insertion, tie-cell insertion, and mapped instance
deduplication because they change the graph or load. Probe stripping is timing
neutral and may occur before graph construction. Library module bodies may
already have been converted to extern declarations as long as cell type and pin
identity remain available and the original Liberty files are supplied.

## Proposed library organization

Use a separate library so enabling OpenTimer does not add a dependency to every
consumer of `CIRCTSynthAnalysis`:

```text
include/circt/Dialect/Synth/Timing/
  TimingTypes.h
  TimingDesign.h
  TimingDesignBuilder.h
  TimingOracle.h
  TimingJSON.h

lib/Dialect/Synth/Timing/
  TimingDesign.cpp
  TimingDesignBuilder.cpp
  TimingOracle.cpp
  TimingJSON.cpp
  OpenTimerTimingOracle.cpp       # optional target
  CMakeLists.txt
```

Suggested targets:

- `CIRCTSynthTiming`: backend-neutral graph, types, JSON, and builder;
- `CIRCTSynthOpenTimer`: optional adapter and the approved OpenTimer source
  subset; and
- `circt-synth`: links `CIRCTSynthTiming`, and links/registers
  `CIRCTSynthOpenTimer` when enabled.

OpenTimer headers must not appear in CIRCT public headers. The adapter should
use a private implementation so that OpenTimer types, macros, and include paths
do not become part of CIRCT's API or ABI.

## Public data model

The declarations below illustrate the contract. Exact LLVM container choices
may change during implementation, but the separation and units are required.

### Time and transition types

Convert external time values at the boundary and represent them explicitly:

```c++
struct TimePs {
  int64_t value = 0;
};

struct CapacitanceFf {
  double value = 0.0;
};

enum class TimingTransition { Rise, Fall };
enum class TimingAnalysisKind { SetupMax };
```

Do not use an unqualified `double` in backend-neutral interfaces. OpenTimer can
use its native floating-point units internally. The adapter converts to
picoseconds with checked rounding. CLI values such as
`--retry-slack-target=<ns>` are converted once before comparison.

If sub-picosecond precision becomes material, replace `TimePs` centrally with a
finer integral unit. Do not add backend-specific floating-point fields to the
shared result.

### Runtime node identity and persistent object identity

Two forms of identity serve different purposes:

```c++
struct TimingNodeId {
  uint32_t value;
};

enum class TimingObjectKind { Instance, InstancePin, TopPort };

struct TimingObjectKey {
  TimingObjectKind kind;
  SmallVector<StringAttr> instancePath;
  StringAttr moduleName;
  unsigned portIndex;
  StringAttr portName;
  unsigned bit;
};
```

`TimingNodeId` is a dense handle valid only for one `TimingDesign`. It makes
graph construction and backend maps efficient.

`TimingObjectKey` is the persistent semantic key used in result and feedback
files. An instance pin is identified by the explicit top-relative instance
path, referenced module, port index, port name, and bit. A top port has an empty
instance path. Including both indices and names lets importers diagnose a
renamed or reordered interface instead of resolving the wrong object.

The builder maintains both mappings:

```text
MLIR object <-> TimingNodeId <-> TimingObjectKey <-> backend handle
```

The OpenTimer adapter should assign synthetic names such as `g42`, `n108`, and
`p17`. It must recover results through maps created at insertion time. It must
not parse synthetic or hierarchical names to identify a CIRCT object.

Display names are generated only for diagnostics. They are never authoritative
input to feedback resolution.

### Timing graph

`TimingDesign` is an immutable, backend-neutral flattened netlist:

```c++
struct TimingCellInstance {
  uint32_t id;
  StringAttr cellType;
  TimingObjectKey instance;
  SmallVector<TimingPinConnection> pins;
};

struct TimingNet {
  uint32_t id;
  std::optional<TimingNodeId> driver;
  SmallVector<TimingNodeId> loads;
};

struct TimingDesign {
  StringAttr topModule;
  SmallVector<TimingCellInstance> cells;
  SmallVector<TimingNet> nets;
  SmallVector<TimingPrimaryPort> ports;
  TimingDesignDiagnostics diagnostics;
};
```

The neutral graph stores cell type, pin, net, direction, and object identity.
It does not copy Liberty tables into a second CIRCT timing model. The OpenTimer
backend loads Liberty and binds graph cells to its library objects.

### Constraints

Represent the supported subset without Tcl or SDC syntax:

```c++
struct TimingClock {
  StringAttr name;
  TimingObjectKey source;
  TimePs period;
  TimePs riseEdge;
  TimePs fallEdge;
};

struct TimingPortConstraint {
  TimingObjectKey port;
  std::optional<TimePs> lateArrival;
  std::optional<TimePs> lateRequired;
  std::optional<TimePs> riseSlew;
  std::optional<TimePs> fallSlew;
  std::optional<CapacitanceFf> load;
};

struct TimingConstraints {
  SmallVector<TimingClock> clocks;
  SmallVector<TimingPortConstraint> ports;
  TimingAnalysisKind analysis = TimingAnalysisKind::SetupMax;
};
```

The initial source of these constraints is the same resolved `circt-synth`
options used by `synth-emit-sdc`: clock ports and periods, input delays and
transitions, output delays, and output loads. SDC emission can continue for
downstream tools, but the timing library receives typed values directly.

Do not call OpenTimer `read_sdc`. A future general SDC importer should parse
into `TimingConstraints` once and report unsupported commands before any
backend runs.

### Request and capability contract

```c++
struct TimingCapabilities {
  bool setupMax;
  bool holdMin;
  bool multipleClocks;
  bool generatedClocks;
  bool timingExceptions;
  bool latches;
  bool interconnectDelay;
};

struct TimingQuery {
  unsigned maxPaths = 100;
  unsigned maxPathsPerEndpoint = 1;
  bool includePathPoints = false;
};

struct TimingRequest {
  const TimingDesign &design;
  ArrayRef<std::string> libertyFiles;
  const TimingConstraints &constraints;
  TimingQuery query;
};
```

Before analysis, a shared validator compares the request with backend
capabilities. An unsupported required feature is an error. An unsupported
feature that is present but intentionally excluded from the requested analysis
is recorded in diagnostics.

### Result

```c++
enum class TimingResultStatus {
  Complete,
  Incomplete,
  Failed,
};

struct TimingTransitionResult {
  TimingTransition transition;
  TimePs arrival;
  TimePs required;
  TimePs slack;
  std::optional<TimingObjectKey> worstLaunch;
};

struct TimingEndpointResult {
  TimingObjectKey endpoint;
  StringAttr pathGroup;
  SmallVector<TimingTransitionResult> transitions;
  TimePs worstSlack;
};

struct TimingPathPoint {
  TimingObjectKey object;
  TimingTransition transition;
  TimePs arrival;
};

struct TimingPathResult {
  TimingObjectKey endpoint;
  TimingTransition transition;
  TimePs arrival;
  TimePs required;
  TimePs slack;
  SmallVector<TimingPathPoint> points;
};

struct TimingCoverage {
  uint64_t expectedEndpoints = 0;
  uint64_t constrainedEndpoints = 0;
  uint64_t analyzedEndpoints = 0;
  uint64_t unconstrainedEndpoints = 0;
  uint64_t unsupportedEndpoints = 0;
};

struct TimingResult {
  TimingResultStatus status;
  StringAttr backend;
  StringAttr backendVersion;
  std::optional<TimePs> wns;
  std::optional<TimePs> tns;
  uint64_t failingEndpoints = 0;
  TimingCoverage coverage;
  SmallVector<TimingEndpointResult> endpoints;
  SmallVector<TimingPathResult> worstPaths;
  SmallVector<TimingDiagnostic> diagnostics;
};
```

The oracle interface is deliberately small:

```c++
class TimingOracle {
public:
  virtual ~TimingOracle() = default;
  virtual StringRef getName() const = 0;
  virtual TimingCapabilities getCapabilities() const = 0;
  virtual FailureOr<TimingResult> analyze(const TimingRequest &) = 0;
};
```

Construction is by a registry or factory selected by a command-line backend
name. Refinement code receives only `TimingResult`.

## Metric semantics

Candidate ordering must not depend on a backend's report defaults. CIRCT owns
the aggregate definitions.

For the initial `SetupMax` mode:

- evaluate late/max rise and fall states for every constrained endpoint;
- define an endpoint's slack as the minimum valid slack over its analyzed
  transitions;
- define WNS as the minimum endpoint slack;
- define TNS as the sum of `min(endpointSlack, 0)` once per endpoint; and
- define `failingEndpoints` as the number of endpoints with negative endpoint
  slack.

This avoids counting a single endpoint twice because both transitions fail and
avoids treating a top-K path report as the complete endpoint set. Backends may
use native WNS, TNS, or FEP queries as consistency checks, but the normalized
values used by the retry comparator follow the definitions above.

An endpoint without a valid required time does not participate in WNS or TNS.
It increments `unconstrainedEndpoints`. A candidate result is `Complete` only
when all endpoints expected for the requested mode are either analyzed or
explicitly classified by a permitted exclusion.

The existing candidate objective remains:

1. a candidate meeting the slack target beats one that does not;
2. among candidates meeting the target, fewer mapped instances wins;
3. otherwise higher WNS wins, then higher TNS, then fewer failing endpoints,
   then fewer mapped instances; and
4. the baseline wins ties.

The comparator accepts only `Complete` results by default. A diagnostic option
may print incomplete results, but they must not win a refinement run.

## Graph construction from mapped HW IR

### Preconditions

The builder accepts an `mlir::ModuleOp`, an explicit `hw.module` top symbol,
and a set of library cell names. The design must be technology mapped. Every
timed leaf instance must bind to a cell in one of the configured Liberty files.

The builder validates:

- the top exists and is unique;
- the reachable design instance graph is acyclic;
- every net bit has at most one driver;
- every instance port maps to a known module or library cell port;
- every timed library cell and pin exists in Liberty;
- widths agree at every connection; and
- no functional operation remains unless it is on the explicit transparent
  plumbing allowlist.

Validation errors include a location and persistent object key when available.

### Hierarchical elaboration

Elaborate from the configured top, creating a separate flattened copy for each
instance path. This preserves instance-specific loads and constraints even when
multiple instances share one `hw.module` definition.

The elaborator carries an `igraph::InstancePath` and appends the current
instance when entering a child module. It uses explicit top selection and never
infers a root by scanning for uninstantiated modules.

Recursive instantiation is rejected. Empty extern design modules are black
boxes and require an explicit timing model; otherwise endpoints crossing them
are unsupported.

### Bit-level connectivity

STA is pin and net based, so aggregate values are expanded into bits. Each bit
gets a unique net identity. The initial transparent plumbing allowlist should
cover the operations that can remain after mapping, including:

- integer bitcasts that preserve width;
- `comb.extract`;
- `comb.concat`;
- replicated constants and tie-cell connections;
- module input and output forwarding; and
- other pure wiring operations identified during implementation.

These operations contribute connectivity and zero delay. They do not become
OpenTimer gates.

Any remaining arithmetic, mux, comparison, sequential, or unknown operation is
an error. Adding an operation to the transparent allowlist requires proving
that it has wiring semantics.

### Library cells

Each mapped library instance becomes one timing gate. Bind pins by the HW
module's explicit port order and Liberty pin names. Direction, clock pin,
sequential data pin, and timing arc classification come from Liberty rather
than hard-coded names such as `D`, `Q`, or `CLK`.

This is also the point to remove the current feedback resolver's dependency on
a pin literally named `D`. The persistent endpoint key records the actual data
pin index and name selected from the library timing check.

Cells absent from Liberty, ports absent from a cell, unsupported three-state
behavior, and multiple-driver nets fail graph construction. Do not omit the
cell and continue.

### Primary inputs and outputs

Top-level ports become OpenTimer primary inputs or outputs per bit. Inout ports
are unsupported in the first implementation unless they have been lowered to
separate input, output, and enable signals.

Primary input arrival and slew and primary output required time and load come
from `TimingConstraints`. Defaults must be explicit in the request and printed
in the result. The adapter must not rely on undocumented OpenTimer defaults.

### Sequential timing

The initial mode analyzes register-to-register, input-to-register,
register-to-output, and input-to-output max paths when their endpoints are
constrained.

Clock-to-Q and setup arcs come from Liberty. Clock sources are ideal top-level
clocks. Designs with latches or unsupported sequential timing checks produce an
incomplete result and cannot win a retry candidate comparison.

The first milestone may require exactly one active clock. Multiple clocks may
be enabled only after clock-domain behavior and unconstrained crossings have
explicit differential tests. The capability report must reflect the actual
implementation rather than OpenTimer's theoretical feature set.

## OpenTimer backend

### API use

The adapter should use OpenTimer's builder and query API:

- load Liberty with `read_celllib` or the corresponding core API;
- create nets, gates, primary inputs, and primary outputs directly;
- connect pins directly;
- create clocks and set arrival, required, slew, and load values directly;
- call `update_timing`;
- query each endpoint's max rise/fall arrival, required time, and slack; and
- use `report_timing` only to obtain the requested worst path traces.

Do not use OpenTimer's Verilog reader. Its documented input is a small flat
gate-level subset and hierarchy and buses remain limited. CIRCT already has a
precise representation of the candidate graph.

Do not use OpenTimer's SDC reader. Its supported command set is limited, and its
parser introduces a separate licensing boundary. Typed constraints are less
ambiguous and easier to validate.

### Result extraction

At graph insertion time, record maps from every OpenTimer gate, pin, net, and
endpoint handle to `TimingNodeId`. Query results through those handles or the
synthetic names assigned by the adapter. Convert to `TimingObjectKey` through
the CIRCT graph maps.

OpenTimer `Path` objects expose path points and slack. The adapter converts each
point immediately while the timer and graph are alive. No OpenTimer references
escape `analyze`.

The adapter enumerates the full constrained endpoint set to compute normalized
WNS, TNS, and failing endpoint count. It does not derive those metrics from the
top-K path list.

### Threading and determinism

Expose a timing thread count separately from synthesis pass threading. Start
with one timing thread in deterministic tests. Production experiments may use
more threads after confirming that aggregate results and path ordering remain
stable.

Sort serialized endpoints by `TimingObjectKey` and paths by slack, endpoint
key, transition, and launch key. Floating-point values are converted once to
integral picoseconds before sorting or comparison.

### OpenTimer version and source boundary

Pin an audited OpenTimer revision. The local revision evaluated while writing
this document was `cd0565fc7a720368e75751c717be92d76350ee8d` (2025-12-25).
The pin should change only through an explicit dependency update with timing
and license review.

Do not consume OpenTimer's top-level CMake project unchanged. It sets global
build behavior, builds parsers and shell components that CIRCT does not need,
and makes the dependency boundary hard to audit. Define an explicit source
manifest for the timer, Liberty, unit, and required utility components. Disable
OpenTimer tools, examples, tests, Verilog parsing, SDC parsing, SPEF parsing,
and shell code unless a later milestone justifies them.

If OpenTimer's headers unconditionally include excluded parser headers, carry a
small reviewed patch that makes those components conditional. Keep such patches
next to the dependency pin and submit them upstream where practical.

Relevant upstream material for dependency review:

- [OpenTimer repository](https://github.com/OpenTimer/OpenTimer);
- [public `Timer` API](https://github.com/OpenTimer/OpenTimer/blob/master/ot/timer/timer.hpp);
- [top-level MIT license](https://github.com/OpenTimer/OpenTimer/blob/master/LICENSE);
- [documented SDC subset](https://github.com/OpenTimer/OpenTimer/blob/master/wiki/io/sdc.md);
- [documented Liberty/NLDM model](https://github.com/OpenTimer/OpenTimer/blob/master/wiki/io/library.md);
  and
- [documented Verilog subset](https://github.com/OpenTimer/OpenTimer/blob/master/wiki/io/verilog.md).

## Licensing boundary

OpenTimer's top-level license is MIT at the revision inspected for this design.
Its bundled Synopsys TAP-in SDC parser has a separate license with source
distribution and commercial contributor terms. Therefore the proposed CIRCT
integration excludes:

- `ot/sdc/*`;
- `Timer::read_sdc` and its implementation;
- shell code that depends on the SDC parser; and
- generated or copied SDC parser artifacts.

The implementation patch must include a source-level dependency manifest and a
license inventory for OpenTimer and its transitive components, including its
task runtime. CI should make it possible to verify that excluded SDC parser
sources are not compiled or packaged.

This document records an engineering boundary, not a legal determination. The
exact pinned source subset and distribution plan need the project's normal
license review before the backend is enabled in distributed binaries.

OpenSTA remains outside the linked implementation. Its upstream distribution is
GPLv3 with a commercial-license option. A developer may configure an existing
external OpenSTA executable for differential tests where local policy allows
it. CIRCT must not fetch, link, or package OpenSTA as part of this design.

## Build configuration

Add an option with a conservative initial default:

```text
CIRCT_ENABLE_OPENTIMER=OFF
CIRCT_OPENTIMER_SOURCE_DIR=<optional local source override>
```

When enabled without a source override, CMake may fetch the pinned revision in
the same style as other optional CIRCT dependencies. Release packaging should
use a pre-fetched or vendored, audited source archive.

`CIRCTSynthTiming` always builds. `CIRCTSynthOpenTimer` builds only when the
option is enabled. A build without OpenTimer can parse timing result JSON,
construct graphs for tests, and report that the `opentimer` backend is
unavailable.

Do not place `CIRCT_ENABLE_OPENTIMER` conditionals in refinement analyses or
transforms. Conditional code belongs in backend registration and command-line
selection.

## Command-line integration

The proposed user-facing controls are:

```text
--timing-backend=<none|opentimer|external-opensta>
--timing-result=<path>
--timing-max-paths=<N>
--timing-threads=<N>
--timing-require-complete
```

Names may be adjusted to existing `circt-synth` conventions during review.
Their behavior should be:

- `none`: do not run library timing;
- `opentimer`: require a build with the OpenTimer backend;
- `external-opensta`: retain the current executable path during migration and
  for developer comparison;
- `--timing-result`: emit the versioned structured result for a direct run;
- `--timing-max-paths`: bound detailed traces without limiting endpoint
  aggregates; and
- `--timing-require-complete`: fail the process if timing coverage is
  incomplete.

When `--datapath-retry` is enabled, the parent adds a private result path to
each candidate child and selects the configured timing backend. Candidate
children never recursively enter the retry supervisor.

The first release should require an explicit backend. An `auto` default can be
considered after OpenTimer passes the rollout gates.

## Structured result format

Use a new result schema rather than overloading timing feedback:

```json
{
  "schema": "circt.synth.timing-result.v1",
  "time_unit": "ps",
  "top": "Top",
  "backend": {
    "name": "opentimer",
    "version": "cd0565fc7a72"
  },
  "analysis": "setup-max",
  "status": "complete",
  "summary": {
    "wns": -312,
    "tns": -744,
    "failing_endpoints": 3
  },
  "coverage": {
    "expected_endpoints": 128,
    "constrained_endpoints": 128,
    "analyzed_endpoints": 128,
    "unconstrained_endpoints": 0,
    "unsupported_endpoints": 0
  },
  "endpoints": [
    {
      "endpoint": {
        "kind": "instance-pin",
        "instance_path": ["u_core", "result_reg"],
        "module": "DFFHQNx1_ASAP7_75t_L",
        "port_index": 0,
        "port_name": "D",
        "bit": 0
      },
      "path_group": "clk",
      "worst_slack": -312,
      "transitions": [
        {
          "transition": "rise",
          "arrival": 5312,
          "required": 5000,
          "slack": -312
        }
      ]
    }
  ],
  "paths": [],
  "diagnostics": []
}
```

All arrays have deterministic ordering. JSON numbers use integral
picoseconds. The writer uses LLVM's JSON support so names are escaped
correctly; do not construct JSON with string concatenation.

### Feedback projection

Add `circt.synth.timing-feedback.v2` with structured startpoint and endpoint
keys. Generate it directly from `TimingResult` for endpoints that need another
refinement round. Preserve the existing v1 string schema as an import-only
compatibility format during migration.

Feedback import proceeds as follows:

1. resolve the configured top explicitly;
2. walk the structured instance path;
3. validate the referenced module, port index, port name, and bit;
4. identify the actual sequential data operand using Liberty metadata; and
5. create or tighten the corresponding timing probe.

Resolution failures are classified as missing, renamed, reordered, wrong cell,
wrong pin role, or ambiguous. Strict mode rejects the file. Permissive mode
skips the record and reports the reason.

## Failure handling

Use three levels of failure:

### Request failure

Malformed constraints, missing Liberty files, top mismatch, graph validation
errors, unsupported required features, or backend initialization errors return
`failure()` and make the candidate invalid.

### Incomplete analysis

Known unsupported endpoints or intentionally omitted timing checks produce a
serializable `Incomplete` result with coverage counts. The default retry driver
does not allow it to win.

### Candidate process failure

Timeout, signal termination, assertion failure, or missing result JSON is
reported by the parent using the existing candidate failure path. Other
candidates continue. If the baseline fails, the refinement run fails rather
than selecting an unjudged candidate.

Write results to a temporary file and rename only after successful
serialization. This prevents a killed child from leaving a valid-looking
partial result.

## Caching and performance

The first implementation rebuilds the timer per candidate. Candidate netlists
can differ globally after mapping, and correctness is more important than
cross-candidate incremental reuse.

Safe initial caching opportunities are:

- parse each Liberty file once per candidate process;
- reserve graph storage from mapped instance and bit counts;
- intern cell, pin, module, and instance names;
- avoid collecting path points when only aggregate scoring is requested; and
- request detailed paths only for failing or top-ranked endpoints.

Later, a long-lived timing worker process could cache Liberty across candidates
while retaining isolation from the retry supervisor. That is a separate
optimization and requires a framed request protocol, memory limits, and a way
to discard all design state between requests.

Incremental OpenTimer edits within one candidate are useful only after CIRCT
can express a local mapped-netlist delta. The current retry loop resynthesizes
whole candidates, so introducing a fragile graph differ is premature.

## Relationship to existing timing analyses

`LongestPathAnalysis` and `TimingPressureAnalysis` remain useful:

- `LongestPathAnalysis` provides fast unit-delay dataflow analysis and path
  debugging before technology mapping;
- `TimingPressureAnalysis` propagates structural pressure and probe
  reachability to choose refinement regions; and
- `TimingOracle` judges mapped candidates with Liberty timing.

They should share identity helpers where useful, especially
`igraph::InstancePath`, but they serve different accuracy and lifecycle needs.
Do not make either structural analysis depend on OpenTimer.

The existing CIRCT Liberty importer intentionally does not yet provide the
complete sequential timing model needed for setup analysis. Reusing its NLDM
attributes can be explored later, but the first OpenTimer adapter should use
OpenTimer's audited Liberty model rather than silently combining two partial
models.

## Why OpenTimer first

OpenTimer already supplies the difficult parts needed by the optimizer:

- NLDM cell delay and slew propagation;
- combinational and sequential timing arcs;
- load calculation;
- arrival and required-time propagation;
- WNS, TNS, failing endpoint, and path queries; and
- a direct graph-construction API.

Its parser limitations are avoidable because CIRCT can construct the graph and
constraints directly. Its timing limitations are acceptable for a bounded
setup/max optimization oracle if they are surfaced through capabilities and
coverage.

A native CIRCT timer would need, at minimum, Liberty parsing and validation,
table interpolation, unateness, rise/fall propagation, slew, capacitance,
clock-to-Q, setup checks, required-time propagation, endpoint aggregation,
path reconstruction, loop handling, and extensive correlation tests. That work
should be justified by measured OpenTimer shortcomings rather than assumed in
advance.

## Validation strategy

Validation has four parts.

### Interface and graph tests

Add backend-neutral tests for:

- deterministic `TimingObjectKey` serialization;
- hierarchy elaboration with repeated module instances;
- escaped names and names containing hierarchy-like punctuation;
- buses, extracts, concats, and bit ordering;
- library port binding by index and name;
- top selection when several root-like modules exist;
- multiple drivers, recursive hierarchy, missing cells, and width errors;
- result JSON round trips and schema rejection; and
- complete versus incomplete coverage classification.

### Timer microbenchmarks

Use small Liberty fixtures with hand-checkable values for:

- one combinational arc;
- positive and negative unate arcs;
- reconvergent paths;
- fanout load and output load;
- slew-dependent and load-dependent interpolation;
- register-to-register clock-to-Q plus setup timing;
- input-to-register and register-to-output paths;
- rise/fall endpoint normalization; and
- WNS/TNS aggregation over more endpoints than `maxPaths`.

These tests should run only when the backend is enabled. Backend-neutral tests
remain available in all builds.

### Differential reference runs

For development, compare OpenTimer results with the existing external OpenSTA
flow on mapped ASAP7 and Sky130 designs using the same Liberty files and the
supported constraint subset. Capture:

- endpoint coverage and the set of unconstrained endpoints;
- WNS and worst endpoint identity;
- per-endpoint slack distribution;
- TNS and failing endpoint count under CIRCT's normalization;
- top path launch and endpoint pairs;
- candidate ordering for every refinement round; and
- analysis runtime and peak memory.

Differences must be reduced to a small reproducer or classified as an expected
model difference. Comparing only the single worst slack is insufficient.

### Refinement regression tests

Replace fake textual STA output with a structured fake result only for driver
control-flow tests. Separately test the real graph builder and OpenTimer adapter
with actual tiny libraries. Preserve tests for:

- baseline wins ties;
- target-met candidates compare by area;
- violating candidates compare by WNS, TNS, failing endpoints, then area;
- feedback selects the intended region;
- repeated feedback signatures stop iteration;
- timeout and malformed result handling; and
- final output contains no probes or transient policy attributes.

## Rollout gates

The backend advances through three stages.

### Stage A: experimental backend

- build option defaults off;
- direct timing report and differential tooling are available;
- retry driver continues to default to the existing configured judge; and
- unsupported features always appear in coverage diagnostics.

Exit criteria:

- all interface, graph, and microbenchmark tests pass;
- the approved source manifest contains no excluded SDC parser code; and
- result JSON is deterministic across repeated runs.

### Stage B: opt-in refinement judge

- `--timing-backend=opentimer` can judge retry candidates;
- structured v2 feedback drives refinement; and
- experiments run both OpenTimer and the independent reference flow.

Exit criteria on the agreed benchmark set:

- every expected setup endpoint is accounted for;
- unexplained WNS outliers are eliminated;
- TNS and failing endpoint differences are understood;
- candidate winner agreement is at least 95 percent; and
- no OpenTimer-selected winner regresses reference WNS beyond the agreed
  tolerance.

The benchmark owner should set a technology-appropriate numerical tolerance
before looking at results. A suggested starting point is the larger of 50 ps
and 5 percent of the reference critical path delay.

### Stage C: preferred optimization oracle

OpenTimer may become the preferred retry backend after Stage B is satisfied on
both a small-cell library and a system-scale design. Keep independent signoff
or reference measurement in release qualification.

Default-on timing refinement is a separate product decision. Successful timer
integration alone does not establish that refinement's runtime and PPA tradeoff
is suitable as a default.

## Implementation plan

### Patch 1: shared result and identity types

- add `TimingTypes`, `TimingResult`, JSON reader/writer, and schema tests;
- replace private `StaPath` and timing fields in `CandidateResult` with the
  shared result types;
- make the candidate comparator operate on integral picoseconds; and
- retain an adapter from the current external report path during migration.

Acceptance: existing retry behavior is preserved, while scoring consumes a
structured `TimingResult`.

### Patch 2: typed feedback v2

- add structured object keys to timing feedback;
- resolve from the explicitly configured top;
- validate port index, name, bit, and sequential role;
- keep v1 as import-only compatibility; and
- generate JSON with LLVM JSON APIs.

Acceptance: escaped names and real report annotations cannot affect endpoint
resolution, and feedback round trips without parsing a display name.

### Patch 3: backend-neutral timing graph builder

- add hierarchy elaboration and bit-level connectivity;
- add cell and port binding validation;
- add graph diagnostics and coverage accounting; and
- test mapped micro-designs without OpenTimer.

Acceptance: a deterministic `TimingDesign` is produced for supported mapped IR,
and every rejected construct has an actionable diagnostic.

### Patch 4: OpenTimer dependency and adapter

- add the pinned, explicit source subset and license inventory;
- keep parser and shell components excluded;
- implement direct graph construction and typed constraints;
- extract every endpoint before computing aggregates; and
- add OpenTimer microbenchmarks.

Acceptance: hand-calculated tests match within one picosecond after rounding,
and `maxPaths` does not change WNS, TNS, or failing endpoint count.

Status (implemented): `CIRCT_ENABLE_OPENTIMER` (default off) with an explicit
26-file source manifest (timer core, Liberty, units, utilities; Verilog/SDC/
SPEF parsers, shell, tools, and tests excluded) plus `test.cpp`, which the
timer core links unconditionally. The adapter (`CIRCTSynthOpenTimer`) builds
the timer graph directly with synthetic names and exact back-maps, pins time
to picoseconds and capacitance to femtofarads via `set_time_unit`/
`set_capacitance_unit`, and normalizes aggregates with the shared helpers.
Microbenchmark (`OpenTimerOracleTest.AndIntoFlop`, extracted ASAP7 subset in
`test/Dialect/Synth/Inputs/tiny-timing.lib`) matches OpenSTA within rounding
(arrival 15.36 vs 15.365ps, required exact at 485.666ps).

Findings that constrain the design (all validated empirically):

- Liberty files must be loaded per split (`read_celllib(lib, MIN/MAX)`);
  a single load leaves one corner without cells.
- Primary inputs/outputs arrive pre-connected to implicit nets named after
  the pin; the adapter merges onto those nets and never reconnects them.
- OpenTimer needs explicit input slews and both-split boundary conditions
  (setup tests read the opposite corner's clock arrival).
- `report_timing` only materializes violated paths, so it cannot enumerate
  met endpoints. Full-coverage aggregates use the new `Timer::report_endpoints()`
  (see `opentimer-report-endpoints.patch`; CMake fails closed on unpatched
  sources). Path traces stay best-effort for violated endpoints; launch
  points are omitted in v1 (optional in the schema).

### Patch 5: candidate child integration

- add timing backend and result options;
- run timing after final connectivity-changing passes;
- write results atomically;
- have the retry parent read result JSON; and
- preserve timeout and candidate failure behavior.

Acceptance: an OpenTimer-backed retry completes without Verilog export, SDC
parsing, OpenSTA report parsing, or a `sta` executable.

Status (implemented): `--timing-backend={sta,opentimer}` (default sta),
`--timing-result`, `--timing-clock-period-ns`, `--timing-clock-port`, and
`--timing-input-slew-ps`. The opentimer backend runs in a `--timing-run-only`
child (fork+exec under the existing candidate timeout; crash/timeout contained
like other children) over the synthesized candidate MLIR: extern plus
Liberty-annotated modules become library cells, constraints use the clock
port plus default input boundary, and the result JSON returns over a file
(atomic writes both ways) for judging with the existing TimingResult
scoreboard. Any backend failure falls back to STA, so the baseline is never
lost. `test/circt-synth/datapath-retry-opentimer.mlir` (REQUIRES: opentimer)
covers it; candidate isolation already fork+execs, which satisfies the
dedicated-child containment requirement.

### Patch 6: differential harness and measurements

- run OpenTimer and external OpenSTA on identical candidate artifacts;
- normalize reference endpoint aggregates independently of top-K reports;
- report coverage and per-endpoint deltas; and
- measure ASAP7, Sky130, and at least one system-scale design.

Acceptance: Stage B rollout criteria are either met or the discrepancies are
documented with a decision to fix, constrain, or stop the backend rollout.

Status (partial): `tmp/measure/diff-timing.py` compares OpenTimer result JSON
against OpenSTA `report_checks` logs with independent per-endpoint
normalization (group by endpoint, worst transition) plus WNS and coverage.
Measured:

- Tiny flop (extracted ASAP7 subset, 1 endpoint): DIFFERENTIAL PASS,
  per-endpoint slack delta 0.3ps, WNS 0.4700 vs 0.470301ns.
- muladd candidate (merged ASAP7, 16 flop-D endpoints, `--strip-lib-to-
  extern` for export): OpenTimer WNS 159ps at 0.4ns, 0 failing, worst
  endpoints r0_bit7/r1_bit7 tied. OpenSTA worst is r0_bit7 at 128ps, then
  r1_bit7 at 153ps. Per-endpoint: r1_bit7 matches within 6ps (OT 159 vs
  STA 153), r0_bit7 differs by 31ps (OT 159 vs STA 128). Required times
  match exactly in both; the gap is arrival-only (OT optimistic). r0 is
  the multiplier cone (O2A1O1Ixp5/OA211-heavy), r1 the adder cone, so the
  gap is cell- or path-specific, not a global offset. Root-cause update:
  launch-sweep (boosting STA's worst launch b[5] by 200ps moves OT's r0
  arrival by the same 200ps) proves both timers see the same worst path;
  single-cell microbenchmarks (AND2, O2A1O1Ixp5) match within 0.3ps. The
  33ps gap over 10 stages (~3ps/stage) is per-stage accumulation, likely
  slew-propagation or LUT-interpolation divergence amplifying with depth.
  Root-cause update (per-stage join on uniquified instances): AND4 matches
  exactly; XNOR2xp5/NOR2 stages diverge (up to 15ps at one NOR2, XNOR swings
  both ways stage to stage). Single-XNOR isolation reproduces 3.3ps: XNOR
  has state-dependent arcs (`when B` positive-unate vs `when !B`
  negative-unate, ~3ps apart at the operating point) and both inputs switch
  together, so neither side condition statically holds. OpenTimer takes the
  faster arc, OpenSTA the worst case. Mechanism confirmed: OT's Liberty
  parser merges timing groups by `related_pin` alone (`t = ti` overwrite),
  so the second `when` variant silently replaces the first and only one
  arc reaches propagation. Fixed: retain every timing variant at parse
  time (merge only into power-only placeholders); forward propagation
  already maxes over fanin arcs, so worst-case selection is automatic.
  Single-XNOR now matches (15 vs 15.297ps), muladd r0_bit7 matches
  (arrival 257 vs 257.4ps, slack 128 vs 128.229ps), mul16 r0_bit14 matches
  (180 vs 179.8ps). Meanwhile the backend is
  usable for candidate ranking where relative order holds, but
  absolute-WNS decisions need the STA cross-check. The backend still
  succeeds without export (elaborates the top only), which the STA flow
  cannot.
- mul16 candidate (merged ASAP7, 32 flop-D endpoints): OpenTimer worst
  r0_bit14 at 180ps, OpenSTA worst r0_bit14 at 180ps (0.6ns period, exact
  after the when-variant fix; was 208ps before). Same worst endpoint.
  (STA side reported only the worst endpoint despite a high path count;
  full STA-side coverage needs broader report flags, a harness follow-up,
  not a backend gap.)
- FIFO (merged ASAP7, 2106 flop-D endpoints, 2ns period): OpenTimer WNS
  -23.65ns, OpenSTA WNS -23.37ns (1%). Same worst endpoints (rdata bits);
  address-counter flops fail in both (OT -2708 vs STA -2665ps on
  fifo_reader/addr_bit0, <2%; STA omits them from top-K reports but
  confirms on direct query). First system-scale agreement (flop-based;
  SRAM macros remain future work and block memory-based SoCs).
- Sky130 and system-scale (FIFO/SmallBoom) differentials are deferred:
  no Sky130 PDK is available in this environment, and system scale needs
  the export prerequisite above plus SRAM timing models (Patch 3 follow-up).

### Yosys reference gap (same Liberty, same SDC/period, STA-measured)

| Design | Flow | Inst | Area (um2) | WNS (ns) |
|---|---|---|---|---|
| muladd @0.4ns | CIRCT baseline | 243 | 28.42 | 0.128 |
| muladd @0.4ns | CIRCT best (booth) | 230 | 27.26 | 0.128 |
| muladd @0.4ns | CIRCT + ABC rewrite + booth | 186 | 23.01 | 0.128 |
| muladd @0.4ns | CIRCT + ABC + booth + bubble-push | 183 | 22.85 | 0.128 |
| muladd @0.4ns | CIRCT + ABC + booth + supergates | 171 | 19.92 | 0.138 |
Supergates help muladd (-12) but regress mul16 (569->714); not universal,
stay opt-in. Mapping portfolio (supergates/no-bundles as retry variants,
merged via scoreboard selection): muladd tries 8 candidates, baseline
(bundles, 150 inst) correctly beats supergates (171) and no-bundles
(201). True per-cone choice merging needs cover-set refactoring; portfolio
selection captures most variance today. `--merge-modules` (opt-in)
composes best submodule bodies across portfolio runs by transitive
instance count (supergates expanded) and judges the composition as an
extra candidate: safe across module boundaries, never worse than the
portfolio best, wins when submodule preferences diverge (pending a
demonstrating design; ties correctly on current benchmarks).
FIFO control logic barely responds to ABC/bundles/bubble
(-46 cells). Polarity analysis (FIFO full flow): every complemented
driver also feeds positive logic (zero all-complemented fanouts), so no
covering can remove the 1780 INVs without breaking positive users;
duplicating drivers as complements is area-neutral at best. Yosys's
NAND/OAI-heavy structure (2470 NAND2, 2127 OAI21, 533 BUFs) vs CIRCT's
AND/OR/INV-heavy structure comes from restructuring (ABC mapping with
choices isolates polarities), not covering. Decision: polarity-aware
covering is complete (bubble pushing optimal for single-fanout);
closing the control gap needs choice-based restructuring (native choices
or ABC-map integration — AIGER boundary loses choices today), a larger
future project.
| muladd @0.4ns | CIRCT + ABC + booth + bundles | 150 | 21.01 | 0.121 |
| muladd @0.4ns | Yosys 0.68 + ABC | 175 | 20.46 | 0.147 |
| mul16 @0.6ns | CIRCT baseline | 839 | 90.73 | 0.180 |
| mul16 @0.6ns | CIRCT best (booth) | 753 | 84.81 | 0.186 |
| mul16 @0.6ns | CIRCT + ABC rewrite + booth | 636 | 73.40 | 0.200 |
| mul16 @0.6ns | CIRCT + ABC + booth + bundles | 569 | 66.97 | 0.170 |
| mul16 @0.6ns | Yosys 0.68 + ABC | 671 | 71.98 | 0.145 |
| fifo @2ns | CIRCT + buffer-high-fanout | 12324 | 1540.6 | 1.649 (STA) / 1.331 (OT) |
| fifo @2ns | CIRCT full (ABC+bundles+bubble+buffer) | 12278 | 1531.5 | 1.594 (STA) / 1.308 (OT) |
| fifo @2ns | Yosys 0.68 + ABC | 10376 | 1199.4 | 1.721 |

Findings: Yosys leads area by 15-25% without ABC rewriting (CIRCT's
Wallace/compress structures are larger); enabling `--abc-commands`
(balance/rewrite/refactor, via `--abc-path` to yosys-abc) closes muladd
to 6% inst / 12% area (186 vs 175, 23.01 vs 20.46um2) with identical
timing. Timing is mixed: Yosys wins muladd (+19ps), CIRCT wins mul16
(+41ps). Remaining area gap is mapping quality (ABC's mapper vs
TechMapper), not rewriting. MAJ gap closed by `--extract-bundles`
(now default-on, opt-out via `=false`): muladd 183->150 inst with 10 FAx1
full adders (total 15 majorities vs Yosys 14); mul16 636->569. Bundling
is neutral on control logic (FIFO -8 inst) and the full lit suite passes
unchanged, so it flipped to default. Flop QN+INV swapping analyzed and
rejected: QN outputs fan out to mixed-polarity users, so no economical
swap exists (would need dual-output flops or add a flop to remove an
inverter). FIFO without
buffering is broken in CIRCT (21ns unbuffered inverter, -23ns WNS);
`--buffer-high-fanout` restores competitiveness (+1.65ns STA). Open item (refined): buffered-FIFO OT-vs-STA differ by 318ps setup
(OT 1331 vs STA 1649ps, same worst endpoint mem_r71_b0). Single cells
match (AND2x4 15 vs 15.365ps), so the gap is contextual, not cellular:
OT's worst path to shared pins comes from flop launches (sequential
depth) while STA's traced path launches from primary inputs, and buffer
(HB1) high-load extrapolation likely diverges. Per-stage join shows OT
arrivals 5x STA from the first gate on the STA path, confirming different
worst paths rather than per-arc error. Needs launch-matched comparison
(same startpoint) to isolate; backend remains valid for ranking.

### Patch 7: remove transitional text parsing

After Stage B succeeds:

- remove OpenSTA text parsing from the normal retry path;
- retain external reference tooling outside the production library boundary;
- update timing refinement documentation and command help; and
- decide whether the OpenTimer build option remains off by default.

## Criteria for starting a native CIRCT backend

Begin a native backend only when at least one of these is demonstrated:

- the approved OpenTimer source subset cannot satisfy project licensing or
  distribution requirements;
- required Liberty constructs cannot be added or maintained reasonably;
- graph or query APIs prevent complete endpoint analysis;
- candidate ranking misses the Stage B target after known modeling differences
  are addressed;
- runtime or memory prevents the bounded retry loop from being useful; or
- the project needs incremental IR-aware timing updates that cannot be exposed
  through OpenTimer.

If started, the native backend should implement the same `TimingOracle`
contract and result normalization. The first native scope should remain
setup/max, one corner, ideal clocks, NLDM, and the typed constraint subset. It
should reuse the graph builder, object identity, JSON, differential harness,
and rollout gates from this design.

## Open questions

The implementation reviews should resolve these questions with experiments:

1. Which OpenTimer source files and transitive dependencies form the minimum
   functional and license-approved subset?
2. Does OpenTimer expose every sequential endpoint and transition needed to
   compute CIRCT-normalized aggregates, or is a small upstream API extension
   required?
3. Which wiring operations remain after all supported mapped pipelines, and
   which should be canonicalized before graph construction?
4. Should the first backend reject multiple clocks or analyze them with all
   cross-domain paths enabled?
5. Are picoseconds sufficiently precise for candidate ordering on the target
   libraries?
6. Which constraints should be stored as IR attributes for reproducible timing
   outside `circt-synth`?
7. Is instance count sufficient for the area tie-break, or should the shared
   candidate result later carry Liberty cell area?
8. What benchmark-specific numerical tolerance should Stage B use before
   differential results are collected?

None of these questions requires coupling the refinement transformations to a
specific timing backend.

## Completion criteria

This design is implemented when:

- a build can enable OpenTimer without compiling its Verilog or SDC parsers;
- mapped HW IR is translated directly into a validated timing graph;
- typed constraints reach the backend without SDC parsing;
- the result reports complete endpoint coverage, WNS, TNS, failing endpoints,
  and bounded worst paths;
- result and feedback identities round-trip without parsing display strings;
- candidate scoring uses normalized whole-design endpoint aggregates;
- candidate timing runs inside the existing isolated child process;
- a retry run requires neither Verilog export nor an external `sta` process;
- unsupported timing semantics prevent an incomplete candidate from winning;
- deterministic tests cover graph construction, result serialization, metrics,
  feedback, timeout handling, and baseline retention; and
- differential results and the rollout decision are recorded for ASAP7,
  Sky130, and a system-scale design.
