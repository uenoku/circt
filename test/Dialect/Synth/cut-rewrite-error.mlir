// RUN: circt-opt --verify-diagnostics --synth-cut-rewrite='db-files=%S/Inputs/cut-rewrite-dot-db.mlir' %s
// expected-error@unknown {{cannot contain symbol operations}}

module {}
