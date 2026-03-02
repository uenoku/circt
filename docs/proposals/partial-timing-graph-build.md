# Partial Timing Graph Build for Targeted Path Queries

## Motivation

In CI workflows, users often want to check timing on specific dataflow paths
(e.g., `--filter-start "a[0]" --filter-end "y[0]"`) rather than running full
STA. Today, the entire timing graph is built and arrival analysis runs on all
nodes before path filtering happens at the enumeration stage. For large designs,
this is wasteful when only a small slice of the design is queried.

## Use Case

```bash
# CI check: verify a specific dataflow path meets timing
circt-sta design.mlir --top chip \
  --filter-start "core/alu/result_reg" \
  --filter-end "core/wb/data_out" \
  --partial-build
```

## Current Flow

```
1. TimingGraph::build()          — processes ALL ops, creates ALL nodes/arcs
2. ArrivalAnalysis::propagate()  — forward propagation through ALL nodes
3. PathEnumerator::enumerate()   — suffix tree only for matched endpoints (already selective)
```

Steps 1–2 dominate runtime. Step 3 is already efficient.

## Proposed Design

### Core Idea

If `fromPatterns` and `toPatterns` are known before graph construction:
1. **Backward cone**: walk MLIR operand chains backward from endpoint values
2. **Forward cone**: walk MLIR use chains forward from startpoint values
3. **Intersect**: only ops in both cones enter the timing graph

### API

```cpp
// TimingGraph — new build mode
LogicalResult buildPartial(ArrayRef<std::string> fromPatterns,
                           ArrayRef<std::string> toPatterns,
                           const DelayModel *delayModel = nullptr);

// TimingAnalysis — targeted query entry point
LogicalResult runPathQuery(const PathQuery &query,
                           SmallVectorImpl<TimingPath> &results);
```

`runPathQuery` builds the partial graph, runs arrival analysis on the subgraph,
and enumerates matching paths. Falls back to full build when patterns are empty.

### Cone Extraction

Two static helpers walk the MLIR SSA graph (not the timing graph, which doesn't
exist yet):

```cpp
// Walk backward through operand chains from endpoints.
// Stops at block arguments (input ports) and sequential op outputs.
DenseSet<Operation *> computeBackwardCone(ArrayRef<Value> endpoints);

// Walk forward through use chains from startpoints.
// Stops at sequential op inputs and output ops.
DenseSet<Operation *> computeForwardCone(ArrayRef<Value> startpoints);
```

### Pattern Matching Before Graph Exists

A `collectMatchingValues` function scans the module to find seed Values matching
glob patterns. It matches against:
- Input port names (start points)
- Register names / register `_D` names (start/end points)
- Output port names (end points)

#### Hierarchical Name Matching

For hierarchical mode, patterns like `inst1/reg_name` must match registers
inside child modules. `collectMatchingValues` recurses into `hw.instance` ops,
building hierarchical names as `contextPath/localName`, and maps matched child
values back to parent-level instance results/operands so the top-level cone walk
can include the relevant instances.

**Key challenge**: the cone extraction walks MLIR SSA at the top level, so child
module values must be projected back to instance ports. When a child register
matches, the instance itself becomes a relevant op, and
`buildPartialHierarchicalGraph` fully elaborates that instance while skipping
irrelevant ones.

### Partial Graph Builders

```cpp
// Flat mode: process only ops in relevantOps set
LogicalResult buildPartialFlatGraph(const DelayModel &model,
                                    const DenseSet<Operation *> &relevantOps);

// Hierarchical mode: skip irrelevant instances at top level,
// fully elaborate relevant child modules via existing buildModuleInContext
LogicalResult buildPartialHierarchicalGraph(
    const DelayModel &model,
    const DenseSet<Operation *> &relevantOps);
```

### circt-sta Integration

Add `--partial-build` flag. When set alongside `--filter-start`/`--filter-end`,
the `PrintTimingAnalysis` pass uses `runPathQuery` instead of `runFullAnalysis`.

**Note**: backward (required time) analysis is skipped in partial mode since
slack computation requires the full graph. Reports show delay and path structure
but not meaningful slack values.

## Files to Modify

| File | Change |
|------|--------|
| `TimingGraph.h` | Add `buildPartial()`, partial builder decls |
| `TimingGraph.cpp` | Cone extraction, `collectMatchingValues`, partial builders |
| `TimingAnalysis.h` | Add `runPathQuery()` |
| `TimingAnalysis.cpp` | Implement `runPathQuery()` |
| `SynthPasses.td` | Add `partialBuild` option to `PrintTimingAnalysis` |
| `PrintTimingAnalysis.cpp` | Use `runPathQuery` when partial build enabled |
| `circt-sta.cpp` | Add `--partial-build` CLI flag |
| `TimingAnalysisTest.cpp` | Tests for partial build (flat, hierarchical, name matching) |

## Open Issues

1. **Hierarchical name projection**: mapping child-module values back to
   parent-level instance ports is lossy. A child register match currently
   includes *all* instance results, relying on the cone walk to prune. A more
   precise approach would track which child output port the matched register
   reaches.

2. **Slack reporting**: partial graphs don't support meaningful required-time
   analysis. The report should clearly indicate when slack values are unavailable.

3. **Output port matching for partial OutputOp**: the partial builder must
   selectively process `hw.output` only for operands connected to relevant ops.
   Block arguments (input ports) similarly need filtering to avoid pulling in
   unrelated cones.

## Saved Implementation

A working (but incomplete for hierarchical name matching) implementation is saved
at `/tmp/partial-build-patches/`. Apply with:

```bash
git am /tmp/partial-build-patches/*.patch
```
