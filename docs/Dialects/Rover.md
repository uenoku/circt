# Rover Passes

Datapath e-graph optimization over `comb`/`hw`/`datapath`, moved in-tree from
tamagoyaki-case-studies/rover-mlir.

* `rover-insert-graph`: wrap each `hw.module` body in `equivalence.graph`.
* `rover-saturate="patterns-file=... max-iters=4 max-nodes=0"`: equality saturation
  with native `BuildPartialProduct`/`BuildCompress`/`BuildZero` rewrites.
* `rover-extract`: delay-prune then area-greedy extraction, inlines the graph.

`circt-synth --rover --rover-patterns=<pdl-interp> [--rover-iters=4 --rover-nodes=0 --rover-delay]`
runs the three passes before comb lowering; see `lib/Dialect/Rover/rules/`
for the base+datapath pattern sources and `test/circt-synth/rover.mlir`.
