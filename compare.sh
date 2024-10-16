#!/bin/bash
# Exit when error occurs.
set -e
$FIRTOOL_PATH/firtool $1 -o tmp.sv --lowering-options=disallowLocalVariables  --disable-all-randomization

# Emit it as json.
A=$(yosys -p "read_verilog -sv tmp.sv; hierarchy -auto-top;   synth_xilinx" | grep -oP "Estimated number of LCs:\s*\K\d+" | tail -n1)

B=$($FIRTOOL_PATH/firtool $1 -hw-pass-plugin='yosys-optimizer{passes="hierarchy -auto-top",synth_xilinx redirect-log=true}' -ir-hw -o /dev/null 2>&1 | grep -oP "Estimated number of LCs:\s*\K\d+" | tail -n1)
echo "{\"sv\": $A, \"integration\": $B}"
