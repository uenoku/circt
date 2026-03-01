# Slew Propagation and Convergence in CIRCT TimingAnalysis

This note explains how slew is modeled in CIRCT Synth static timing analysis,
how convergence works, and how to read the resulting diagnostics.

## Scope

This document covers the implementation in `TimingAnalysis` and `ArrivalAnalysis`
used by timing report flows (`synth-print-timing-analysis`, `circt-synth`, and
`circt-sta`). It focuses on NLDM-style delay/slew models and the iterative
load-slew loop.

## Why Slew Matters

For NLDM-like timing, arc delay is not a constant. It depends on:

- input transition (`inputSlew`)
- output capacitance (`outputLoad`)

So each arc computes both:

- delay contribution (`DelayResult::delay`)
- propagated output transition (`DelayResult::outputSlew`)

This is why a single forward pass with fixed assumptions is often not enough:
load estimation may itself depend on the latest slew hints.

## Data Flow

At each arc traversal (`u -> v`):

1. Build `DelayContext`:
   - current op and arc context (input/output pin indices)
   - `inputSlew` from the chosen predecessor arrival
   - `outputLoad` from fanout pin capacitance accumulation
2. Call `DelayModel::computeDelay(context)`
3. Update successor arrival time and successor slew with returned values

Key interfaces:

- `include/circt/Dialect/Synth/Analysis/Timing/DelayModel.h`
- `include/circt/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.h`

Core implementation points:

- `lib/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.cpp`
- `lib/Dialect/Synth/Analysis/Timing/TimingAnalysis.cpp`

## Convergence Loop (runFullAnalysis)

When `delayModel->usesSlewPropagation()` is true, `TimingAnalysis::runFullAnalysis()`
runs iterative arrival analysis.

High-level algorithm:

```text
initialize previousSlews from initialSlew
for iter in [1..maxSlewIterations]:
  runArrivalAnalysisWithLoadSlewHints(previousSlews)
  maxDelta = max_i |slew_new[i] - previousSlews[i]|
  relativeDelta = maxDelta / firstIterationDelta  (if firstIterationDelta > 0)
  if maxDelta <= absEps or relativeDelta <= relEps:
    converged
    break
  optionally adapt damping based on residual trend
  previousSlews = previousSlews + damping * (slew_new - previousSlews)
```

Where:

- `absEps` is `slewConvergenceEpsilon`
- `relEps` is `slewConvergenceRelativeEpsilon` (`0` means disabled)
- `damping` is `slewHintDamping`, optionally adapted by
  `adaptiveSlewHintDampingMode`

## Damping and Adaptive Damping

Fixed damping:

- `slewHintDamping = 1.0`: full update (fastest when stable)
- smaller values: relaxed updates (more stable for oscillatory coupling)

Adaptive policy modes:

- `Disabled`
- `Conservative`
- `Aggressive`

Adaptive modes react to residual trend between iterations (growth/stall vs
rapid reduction) and scale damping accordingly.

## Report Diagnostics

Timing reports now include:

- `Arrival Iterations`
- `Slew Converged`
- `Max Slew Delta`
- `Relative Max Slew Delta`
- `Relative Slew Epsilon`
- `Slew Hint Damping`
- `Adaptive Slew Damping Mode`
- `Applied Slew Hint Damping`
- `Slew Delta Trend`

Optional table (`emitSlewConvergenceTable` / `--show-convergence-table`):

```text
--- Slew Convergence ---
Iter | Max Slew Delta
1    | ...
2    | ...
```

Primary report code:

- `lib/Dialect/Synth/Analysis/Timing/TimingReport.cpp`
- `lib/Dialect/Synth/Analysis/PrintTimingAnalysis.cpp`

## Practical Tuning Guidance

Start with:

- `slewHintDamping = 1.0`
- `adaptiveSlewHintDampingMode = Conservative`
- `slewConvergenceEpsilon = 1e-6`
- `slewConvergenceRelativeEpsilon = 0` (off)

Then adjust based on observed trend:

- If residual oscillates or stalls, lower damping or enable aggressive mode.
- If residual drops quickly then plateaus at tiny values, add relative epsilon
  to cap iteration count.
- If convergence is too loose, tighten absolute epsilon and disable relative
  epsilon.

## Related Tests

Convergence behavior and diagnostics are covered in:

- `unittests/Dialect/Synth/TimingAnalysisTest.cpp`
  - `FullPipelineRunsSlewConvergenceLoop`
  - `FullPipelineDetectsNonConvergence`
  - `ConvergenceLoopUpdatesLoadFromSlewHints`
  - `SlewHintDampingChangesConvergenceTrajectory`
  - `AdaptiveSlewHintDampingAdjustsAppliedFactor`
  - `AdaptiveDampingPolicyAffectsAppliedFactor`
  - `RelativeSlewConvergenceEpsilonCanTerminateEarly`

E2E report coverage:

- `test/circt-sta/nldm-timing-report.mlir`
- `test/circt-sta/nldm-timing-report-chain.mlir`

## Background References

- Liberty NLDM concepts: standard Liberty timing tables indexed by transition
  and load (`cell_rise/fall`, `rise/fall_transition`).
- Classical fixed-point / relaxed iteration methods for coupled nonlinear
  systems (used here in a lightweight form via damping + residual checks).
