# Upstream Plan (scope: steps 0–3)

Base: `origin/main` at `e368ee4dc1` (2026-10-03).
Work branch: `dev/hidetou/openroad-2` → planning branch: `dev/hidetou/upstream-0-3`.
Actual PR branches should be cut fresh from `origin/main` and filled by
cherry-pick / `git checkout <work-branch> -- <paths>`.

This is a working doc. Do not upstream it.

## Out of scope (steps 4–5, explicitly excluded)

- Timing OpenTimer backend:
  `lib/Dialect/Synth/Timing/OpenTimer/*`,
  `unittests/Dialect/Synth/OpenTimerOracleTest.cpp`,
  `test/circt-synth/datapath-retry-opentimer.mlir`,
  `test/lit.cfg.py` + `test/lit.site.cfg.py.in` opentimer feature,
  `CMakeLists.txt` `CIRCT_ENABLE_OPENTIMER` option.
- New dialects + integration:
  `lib/Dialect/{Ematch,Equivalence,Rover}/*`,
  `include/circt/Dialect/{Ematch,Equivalence,Rover}/*`,
  `include/circt/InitAllDialects.h` + `InitAllPasses.h` dialect additions,
  `lib/Dialect/CMakeLists.txt` dialect additions,
  `tamagoyaki-integration/*`, `.gitmodules` Tamagoyaki entries,
  `test/Dialect/{Ematch,Equivalence}/*`, `test/Dialect/Rover/*`,
  `test/circt-synth/rover*.mlir`, `docs/Dialects/{Ematch,Equivalence,Rover}.md`.
- Do not upstream: `tmp/*`, `status.md`, `task.md`,
  measurement-only docs (`StaticTimingAnalysisLibraryIntegration.md`,
  `TimingProbeDatapathRefinement.md`), all `??` junk in `git status`,
  `*.orig` / `*.rej` / `*.bck`, `OpenTimer/` untracked clone.

## 0. Hygiene PR (first, no functional change)

- Commit or stash dirty `lib/Dialect/Synth/Transforms/SeqTechMapper.cpp`,
  `test/Dialect/Synth/seq-tech-mapper.mlir`.
- Rebase / verify clean `git status`, `git diff origin/main --stat`.
- `clang-format -i` on touched C++, `yapf` on touched Python.
- Gate:
  ```
  ninja -C build bin/circt-opt bin/circt-synth
  ninja -C build check-circt
  ```

## 1. Small fixes (4 PRs, any order)

- 1a ExportVerilog: `lib/Support/LoweringOptions.cpp`,
  `include/circt/Support/LoweringOptions.{h,Parser.h}`,
  `lib/Conversion/ExportVerilog/ExportVerilog.cpp`,
  `test/Conversion/ExportVerilog/disallow-replicate.mlir`.
- 1b Seq/HW/Datapath: `lib/Dialect/Seq/Transforms/RegOfVecToMem.cpp`
  (check upstream #11106 first — may only need rebase),
  `lib/Dialect/HW/Transforms/HWAggregateToComb.cpp`,
  `lib/Conversion/DatapathToComb/DatapathToComb.cpp`,
  `lib/Dialect/Datapath/Transforms/ReduceDelay.cpp` (+ `CMakeLists.txt`),
  tests `reg-of-vec-to-mem.mlir`, `hw-aggregate-to-comb.mlir`,
  `datapath-reduce-delay-pressure.mlir`.
- 1c LLHD/Verif: `lib/Dialect/LLHD/Transforms/LowerSyncProcesses.cpp`,
  `StripInitDrives.cpp`, `lib/Dialect/Verif/Transforms/StripAssertLike.cpp`
  + `LLHDPasses.td`, `Passes.td`, `CMakeLists.txt` wiring,
  tests `lower-sync-processes.mlir`, `strip-init-drives.mlir`,
  `strip-assert-like.mlir`.
- 1d ImportVerilog/circt-verilog: `lib/Conversion/ImportVerilog/ImportVerilog.cpp`,
  `include/circt/Conversion/ImportVerilog.h`,
  `tools/circt-verilog/*`, `test/circt-verilog/memories.sv`.

## 2. Synth mapping core (6 PRs, in order)

- 2a NPN core: `lib/Support/TruthTable.cpp`,
  `include/circt/Support/TruthTable.h`,
  `lib/Dialect/Synth/Transforms/CutRewriter.{h,cpp}`,
  `TestPriorityCuts.cpp`, `unittests/Support/TruthTableTest.cpp`.
- 2b Wide 7–9 inputs: `WidePatternTables.{h,cpp}` (5.7k lines — PR must
  document generator provenance + load-time verification),
  `tech-mapper-wide-skip.mlir`, `lut-mapper.mlir`, `priority-cuts.mlir`.
- 2c Mapper QoR: `TechMapper.cpp`, `GenericLUTMapper.cpp`,
  `SOPBalancing.cpp`, `tech-mapper*.mlir` (phase/supergate/area-flow/load-aware
  cases belonging to these files only).
- 2d Supergates: `GenSupergates.cpp`, `gen-supergates*.mlir`,
  `tech-mapper-supergates.mlir`, `tech-mapper-ingredient-cuts.mlir`.
- 2e Cleanup passes: `TieCells.cpp`, `DedupInstances.cpp`,
  `SortTopologically.cpp`, `StripLibToExtern.cpp` + matching tests.
- 2f Sequential: `lib/Conversion/ImportLiberty/ImportLiberty.cpp`,
  `include/circt/Conversion/ImportLiberty.h`,
  `include/circt/Dialect/Synth/SynthOps.{td,h}`,
  `lib/Dialect/Synth/SynthOps.cpp`,
  `lib/Dialect/Synth/Transforms/SeqMemMapper.cpp`,
  `SeqTechMapper.cpp`, NLDM part of `DeriveMappingCost.cpp`,
  tests `seq-{mem,tech}-mapper*.mlir`, `liberty-seq.mlir`,
  `sram-macros.mlir`, `Inputs/{numeric,toplevel-cell}.lib`,
  `Inputs/sram-macros.mlir`, plus `ResourceUsageAnalysis.cpp` counting
  (`resource.mlir`) if bundled here.

## 3. Timing-driven, backend-neutral (5 PRs, after 2)

- 3a Timing infra: `include/circt/Dialect/Synth/Timing/`
  minus OpenTimer (`TimingTypes.h`, `TimingResult.h`, `TimingJSON.h`,
  `TimingDesign.h`, `TimingDesignBuilder.h`),
  `lib/Dialect/Synth/Timing/{TimingResult,TimingJSON,TimingDesignBuilder}.cpp`,
  `lib/Dialect/Synth/Timing/CMakeLists.txt`,
  `unittests/Dialect/Synth/{TimingJSON,TimingDesignBuilder}Test.cpp`,
  `unittests/Dialect/Synth/CMakeLists.txt`.
- 3b Probes/feedback: `synth.timing_probe` op in `SynthOps.td` (+ `SynthOps.cpp`,
  `SynthAttributes.td`, `SynthEnums.td` bits owned by probes),
  `include/circt/Dialect/Synth/Analysis/`
  (`TimingFeedback.h`, `TimingPressureAnalysis.h`, `RefinementRegions.h`),
  `lib/Dialect/Synth/Analysis/`
  (`TimingFeedback.cpp`, `TimingPressureAnalysis.cpp`,
  `RefinementRegions.cpp`, `CMakeLists.txt`),
  `lib/Dialect/Synth/Transforms/`
  (`DeriveProbes.cpp`, `DiscoverRefinementRegions.cpp`,
  `ImportTimingFeedback.cpp`, `StripTimingProbes.cpp`),
  `SynthPasses.td` + `SynthesisPipeline.{h,cpp}` probe plumbing,
  tests `probe.mlir`, `probe-early-pipeline.mlir`, `derive-probes.mlir`,
  `discover-*.mlir`, `import-feedback*.mlir`, `strip-probes.mlir`,
  `refine-policies.mlir`, `Inputs/refine-*.json`.
- 3c Bundles: `create_bundle` op, `BundleUtils.h`,
  `ExtractBundles.cpp`, `MapBundles.cpp`,
  tests `bundle.mlir`, `extract-bundles.mlir`, `map-bundles.mlir`,
  `docs/MultiOutputMapping.md`.
- 3d SDC: `EmitSDC.cpp`, `emit-sdc.mlir`, `docs/SdcEmission.md`,
  `circt-synth --sdc-output` flags in `tools/circt-synth/circt-synth.cpp`.
- 3e Opt-in QoR + driver: `BufferHighFanout.cpp`, `SizeCritical.cpp`,
  `BubblePushing.cpp`, `RestructureMuxTrees.cpp` (default-off),
  `tools/circt-synth/{circt-synth.cpp,DatapathRetry.cpp,h}`
  minus OpenTimer judging, `SynthesisPipeline.cpp` remainder,
  tests `buffer-*.mlir`, `bubble-pushing.mlir`, `size-critical.mlir`,
  `datapath-refine*.mlir`, `datapath-retry.mlir`,
  `probe-early-buffer.mlir`, `Inputs/{datapath-retry,probe,refine,sample}.lib`,
  `Inputs/fake-sta*.sh`, `Inputs/buffer-violators.json`,
  `Inputs/size-critical-violated.rpt`,
  trimmed `docs/{TimingDrivenMapping,ClockTreeSynthesis,ORFSBringup}.md`.

Per-PR gate: targeted `llvm-lit` on touched `test/Dialect/Synth/*` and
`test/circt-synth/*`, then full `check-circt` before requesting review.

## Suggested execution

1. Land step 0.
2. Cut `1a`–`1d` from `origin/main`, one PR each.
3. Stack `2a`→`2f`, then `3a`→`3e`; each PR description lists included
  paths from this doc so reviewers can verify scope.
