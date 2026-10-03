# Equivalence Dialect

E-graphs as IR, moved in-tree from Tamagoyaki.
`equivalence.graph` holds an unordered region, `equivalence.class` groups
equivalent values, `equivalence.yield` terminates the graph.

[TOC]

Passes: `equivalence-insert-graph`, `ematch-saturate`, `equivalence-select-greedy`,
`equivalence-select-ilp` (needs HiGHS, compiled out by default),
`equivalence-extract`, `equivalence-graph-size`, `equivalence-restore-invariants`.
