
# Run circt-verilog to geenerate a mlir file, and run circt-synth. 
# And then run circt-opt to run fraig and lec.

#!/bin/bash

# Run circt-verilog to generate a mlir file, and run circt-synth.
# And then run circt-opt to run fraig and lec.

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <verilog_file>"
  exit 1
fi

INPUT_FILE=$1
MLIR_FILE="${INPUT_FILE%.v}.mlir"
SYNTH_FILE="${INPUT_FILE%.v}_synth.mlir"
AFTER_FRAIG_FILE="${INPUT_FILE%.v}_after_fraig.mlir"
BEFORE_FRAIG_FILE="${INPUT_FILE%.v}_before_fraig.mlir"

# Run circt-verilog to generate MLIR
circt-verilog "$INPUT_FILE" -o "$MLIR_FILE"

# Grep hw.module @ to get the top module name
TOP_MODULE=$(grep -oP 'hw\.module\s+@\K\w+' "$MLIR_FILE" | head -n 1)

# Run circt-synth
circt-synth "$MLIR_FILE" -o "$SYNTH_FILE"

circt-translate --export-aiger "$SYNTH_FILE" -o "${SYNTH_FILE%.mlir}_before.aig"

circt-opt "$SYNTH_FILE" --convert-synth-to-comb -o "$BEFORE_FRAIG_FILE"
# Run circt-opt with fraig and lec

circt-opt "$SYNTH_FILE" --synth-functional-reduction -canonicalize -o tmp.mlir --mlir-pass-statistics
circt-translate --export-aiger tmp.mlir -o "${SYNTH_FILE%.mlir}_after.aig"
circt-opt tmp.mlir --convert-synth-to-comb -o "$AFTER_FRAIG_FILE"

yosys-abc -c "cec ${SYNTH_FILE%.mlir}_before.aig ${SYNTH_FILE%.mlir}_after.aig"

circt-lec "$BEFORE_FRAIG_FILE" "$AFTER_FRAIG_FILE" --c1 $TOP_MODULE --c2 $TOP_MODULE
echo "Generated $FINAL_FILE"
