# Plan: Save/Load Timing Analysis Results as MLIR Attributes

## Context

Commercial STA tools save analysis results to session databases so users can
build the timing graph once and run many queries without re-analyzing. We
implement this by storing timing results as MLIR attributes on the IR itself.
The "database" IS the annotated MLIR module — standard MLIR serialization
(text or bytecode) handles persistence, results are inspectable and composable
with downstream passes.

**Workflow:**
```bash
# Step 1: Run full analysis, annotated IR is written to -o
circt-sta design.mlir --top chip --save-timing -o design-timed.mlir

# Step 2: Load cached results, run queries without re-analyzing
circt-sta design-timed.mlir --top chip --load-timing \
  --filter-start "alu/*" --filter-end "wb/*"
```

## Attribute Scheme

### Unified approach: DistinctAttr IDs + hierpath per instance

Every op/port gets a `synth.timing.id` DistinctAttr. Timing results are stored
centrally on the parent ModuleOp, grouped by instance context (hierpath).

```mlir
hw.module @leaf(in %a : i1 {synth.timing.id = distinct[0]<>}, out x : i1) {
  %r = seq.firreg %a clock %clk {synth.timing.id = distinct[1]<>} : i1
  %c = comb.and %r, %r {synth.timing.id = distinct[2]<>} : i1
  hw.output %c : i1
}

hw.module @top(in %a : i1 {synth.timing.id = distinct[3]<>}, out x : i1, out y : i1) {
  %x0 = hw.instance "u1" sym @u1 @leaf(a: %a: i1) -> (x: i1)
  %y0 = hw.instance "u2" sym @u2 @leaf(a: %a: i1) -> (x: i1)
  hw.output %x0, %y0 : i1, i1
}

// One hierpath per unique instance path (NOT per node)
hw.hierpath @hp_u1 [@top::@u1]
hw.hierpath @hp_u2 [@top::@u2]

module attributes {
  synth.timing.session = {
    top = "top", delay_model = "nldm",
    clock_period = 100 : i64, worst_slack = -3 : i64,
    arrival_iterations = 4 : i32
  },
  synth.timing.results = [
    // Top-level nodes (no path key)
    {nodes = [
      {id = distinct[3]<>, arrival_rise = 0 : i64, arrival_fall = 0 : i64,
       slew_rise = 0.1 : f64, slew_fall = 0.1 : f64}
    ]},
    // Per-instance results
    {path = @hp_u1, nodes = [
      {id = distinct[0]<>, arrival_rise = 0 : i64, arrival_fall = 0 : i64,
       slew_rise = 0.1 : f64, slew_fall = 0.1 : f64},
      {id = distinct[1]<>, arrival_rise = 0 : i64, arrival_fall = 0 : i64,
       slew_rise = 0.0 : f64, slew_fall = 0.0 : f64},
      {id = distinct[2]<>, arrival_rise = 5 : i64, arrival_fall = 4 : i64,
       slew_rise = 0.2 : f64, slew_fall = 0.18 : f64}
    ]},
    {path = @hp_u2, nodes = [
      {id = distinct[0]<>, arrival_rise = 2 : i64, ...},
      {id = distinct[1]<>, ...},
      {id = distinct[2]<>, ...}
    ]}
  ]
}
```

**Key properties:**
- Hierpath points to INSTANCE, not individual ops → N instances = N hierpaths
  (not N × M). For deeply nested `top/mid/leaf`:
  `hw.hierpath @hp [@top::@mid_inst, @mid::@leaf_inst]`
- `DistinctAttr` identity survives serialization round-trips
- Same distinct ID appears in both the op attribute and the centralized table
- On load: walk ops to build `DistinctAttr → Value` map, then populate arrivals
- Top-level nodes use entries without a `path` key
- Endpoints include `required` time in their dict

### Design decisions
- Store **per-edge (rise/fall)** arrival and slew — matches commercial reports
- Store worst-case across bit positions for same Value (scalar, not per-bit)
- Required time only on endpoints (output ports, register D inputs)
- Use `DictionaryAttr` with `i64`/`f64` for timing values
- `DistinctAttr::create(UnitAttr::get(ctx))` for node IDs
- `HierPathCache` for efficient hierpath deduplication
- Instance ops need `inner_sym` for hierpath references (add if missing)

## Implementation Steps

### Step 1: Add external population methods to ArrivalAnalysis (~30 lines)

**Files:** `ArrivalAnalysis.h`, `ArrivalAnalysis.cpp`

Add to `ArrivalAnalysis`:
```cpp
/// Externally populate arrival data (for loading from attributes).
void setNodeArrival(const TimingNode *node, int64_t arrivalRise,
                    int64_t arrivalFall, double slewRise, double slewFall);
```

This sets both the per-edge fields and the max fields on `NodeArrivalData`.

### Step 2: Create TimingAnnotation save/load utilities (~300 lines)

**New file:** `include/circt/Dialect/Synth/Analysis/Timing/TimingAnnotation.h`
**New file:** `lib/Dialect/Synth/Analysis/Timing/TimingAnnotation.cpp`

```cpp
namespace circt::synth::timing {

LogicalResult annotateTimingResults(
    mlir::ModuleOp circuit, hw::HWModuleOp topModule,
    TimingGraph &graph, ArrivalAnalysis &arrivals,
    RequiredTimeAnalysis *required = nullptr);

LogicalResult loadTimingFromAttributes(
    mlir::ModuleOp circuit, hw::HWModuleOp topModule,
    TimingGraph &graph, ArrivalAnalysis &arrivals);

bool hasTimingAnnotations(mlir::ModuleOp circuit);
void clearTimingAnnotations(mlir::ModuleOp circuit);
}
```

**`annotateTimingResults` logic:**

1. **Assign IDs**: Walk all ops in all modules reachable from top. For each op
   result and each block argument, set `synth.timing.id = DistinctAttr::create(UnitAttr)`.
   Build `Value → DistinctAttr` map.

2. **Ensure inner_sym on instances**: Walk instance ops, add `inner_sym` if
   missing (needed for hierpath references). Use `hw::InnerSymbolNamespace`
   to avoid collisions.

3. **Create hierpaths**: For each unique instance context path in the timing
   graph (obtained from `node->getContextPath()`), create a hierpath via
   `HierPathCache`. The contextPath string `"u1/u2"` maps to
   `[@top::@u1_sym, @mid::@u2_sym]`.

4. **Build results array**: Group timing nodes by context path. For each group:
   - Look up the node's Value → get its DistinctAttr
   - Read arrival/slew from `ArrivalAnalysis`, required from `RequiredTimeAnalysis`
   - Create dict: `{id, arrival_rise, arrival_fall, slew_rise, slew_fall, [required]}`
   - Wrap group: `{path = @hierpath_sym, nodes = [...]}`
   - Top-level group (empty context): `{nodes = [...]}`

5. **Set module attrs**: `synth.timing.results` ArrayAttr and
   `synth.timing.session` DictionaryAttr on parent ModuleOp.

**`loadTimingFromAttributes` logic:**

1. Read `synth.timing.results` array from parent ModuleOp
2. Build `DistinctAttr → timing data` map from all node entries
3. Walk all ops/ports in modules reachable from top, read `synth.timing.id`
   → build `DistinctAttr → Value` map
4. Walk timing graph nodes. For each node, look up its Value's DistinctAttr,
   then look up timing data from step 2. Call `setNodeArrival()`.

**Existing utilities to reuse:**
- `HierPathCache` (`include/circt/Dialect/HW/HierPathCache.h`)
- `hw::InnerSymbolNamespace` for safe symbol generation
- `TimingGraph::getNodes()` for iterating all nodes
- `ArrivalAnalysis::getMaxArrivalTime/Slew` for reading results
- `node->getValue()`, `node->getContextPath()` for mapping nodes to IR

### Step 3: Wire into TimingAnalysis (~30 lines)

**Files:** `TimingAnalysis.h`, `TimingAnalysis.cpp`

```cpp
LogicalResult saveTimingAnnotations();
LogicalResult loadTimingAnnotations();
```

`saveTimingAnnotations`: calls `annotateTimingResults(...)` after full analysis.

`loadTimingAnnotations`:
1. `buildGraph()` — creates nodes/arcs structure
2. Create empty `ArrivalAnalysis`
3. Call `loadTimingFromAttributes(...)` to populate arrivals
4. Create `PathEnumerator` from graph + arrivals
5. Ready for path queries without re-running analysis

### Step 4: CLI and pass integration (~40 lines)

**`SynthPasses.td`:** Add `saveTiming` and `loadTiming` bool options to
`PrintTimingAnalysis`

**`PrintTimingAnalysis.cpp`:**
- When `saveTiming`: call `analysis->saveTimingAnnotations()` after analysis
- When `loadTiming`: call `analysis->loadTimingAnnotations()` instead of
  `runFullAnalysis()`

**`circt-sta.cpp`:**
- Add `--save-timing` and `--load-timing` flags, wire to pass options

### Step 5: CMakeLists.txt (~2 lines)

**File:** `lib/Dialect/Synth/Analysis/Timing/CMakeLists.txt`

Add `TimingAnnotation.cpp` to sources.

### Step 6: Tests (~100 lines)

**File:** `unittests/Dialect/Synth/TimingAnalysisTest.cpp`

- `TimingAnnotationRoundTrip`: flat IR — run analysis, save, load, verify
  arrival times match
- `TimingAnnotationHierRoundTrip`: hierarchical IR — verify hierpaths created,
  round-trip arrival times
- `TimingAnnotationPathQuery`: load from annotated IR, enumerate paths, verify
- `TimingAnnotationClear`: annotate then clear, verify no `synth.timing.*` remain

## Files Summary

| File | Change |
|------|--------|
| `include/circt/Dialect/Synth/Analysis/Timing/TimingAnnotation.h` | **New** — annotation API |
| `lib/Dialect/Synth/Analysis/Timing/TimingAnnotation.cpp` | **New** — annotate/load/clear impl |
| `include/circt/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.h` | Add `setNodeArrival` |
| `lib/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.cpp` | Implement `setNodeArrival` |
| `include/circt/Dialect/Synth/Analysis/Timing/TimingAnalysis.h` | Add save/loadTimingAnnotations |
| `lib/Dialect/Synth/Analysis/Timing/TimingAnalysis.cpp` | Implement save/load |
| `include/circt/Dialect/Synth/Transforms/SynthPasses.td` | Add saveTiming/loadTiming options |
| `lib/Dialect/Synth/Analysis/PrintTimingAnalysis.cpp` | Wire save/load into pass |
| `tools/circt-sta/circt-sta.cpp` | Add --save-timing/--load-timing flags |
| `lib/Dialect/Synth/Analysis/Timing/CMakeLists.txt` | Add TimingAnnotation.cpp |
| `unittests/Dialect/Synth/TimingAnalysisTest.cpp` | Round-trip + hier + query tests |

## Verification

1. `ninja CIRCTSynthTests && ./unittests/Dialect/Synth/CIRCTSynthTests --gtest_filter="TimingAnalysisTest.TimingAnnotation*"`
2. `ninja CIRCTSynthTests && ./unittests/Dialect/Synth/CIRCTSynthTests --gtest_filter="TimingAnalysisTest.*"` — all existing tests pass
3. `ninja circt-sta` — builds with new flags
4. Manual: `circt-sta test.mlir --top X --save-timing -o out.mlir` then inspect for `synth.timing.*` attrs and `hw.hierpath` ops
