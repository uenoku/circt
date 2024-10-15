#!/bin/bash
# Exit when error occurs.
set -e
SV=$1
MLIR=$2
TOP=$3
# Emit it as json.
A=$(yosys -p "read_verilog -sv $1; hierarchy -top $3; synth_xilinx" | grep -oP "Estimated number of LCs:\s*\K\d+" | tail -n1)

# Replace $3 in the string
#B=$($FIRTOOL_PATH/circt-opt $2 -pass-pipeline="builtin.module(yosys-optimizer{passes=\"hierarchy -top $TOP\",synth_xilinx redirect-log=true})")
B=$($FIRTOOL_PATH/circt-opt $2 -pass-pipeline="builtin.module(yosys-optimizer{passes=\"hierarchy -top $TOP\",synth_xilinx redirect-log=true})" -o /dev/null 2>&1 | grep -oP "Estimated number of LCs:\s*\K\d+" | tail -n1)
echo "{\"sv\": $A, \"integration\": $B}"
