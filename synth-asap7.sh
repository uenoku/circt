#!/bin/bash
# synth-asap7.sh - self-contained ASAP7 mapping demo (fifo -> LVT_FF 5-lib set).
# Usage:
#   ./synth-asap7.sh [--top fifo] [--work /tmp/e2e-asap7] [--period 10]
#                    [--liberty a.lib,b.lib,...] [--sram-macros m.mlir]
#                    [--no-sta] [extra verilog files...]
# Defaults use circt-sfc-experiments/example/fifo/fifo.v, top=fifo.
# Env overrides: ASAP7_LIBS (comma-separated, no spaces), CIRCT_BIN (build/bin).
set -euo pipefail

TOP=fifo
WORK=/tmp/e2e-asap7
PERIOD=10
SRAM_MACROS=""
RUN_STA=1
LIBERTY_OVERRIDE=""
EXTRA_SRCS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --top) TOP="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --period) PERIOD="$2"; shift 2 ;;
    --liberty) LIBERTY_OVERRIDE="$2"; shift 2 ;;
    --sram-macros) SRAM_MACROS="$2"; shift 2 ;;
    --no-sta) RUN_STA=0; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    --) shift; while [[ $# -gt 0 ]]; do EXTRA_SRCS+=("$1"); shift; done ;;
    -*) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
    *) EXTRA_SRCS+=("$1"); shift ;;
  esac
done

REPO="$(cd "$(dirname "$0")" && pwd)"
BIN="${CIRCT_BIN:-$REPO/build/bin}"
for t in circt-verilog circt-synth circt-opt; do
  [[ -x "$BIN/$t" ]] || { echo "missing $BIN/$t: build first (see status.md)" >&2; exit 1; }
done

# Exact 5-file LVT_FF set (status.md). SLVT is NOT part of the set.
if [[ -n "$LIBERTY_OVERRIDE" ]]; then
  LIBS="$LIBERTY_OVERRIDE"
elif [[ -n "${ASAP7_LIBS:-}" ]]; then
  LIBS="$ASAP7_LIBS"
else
  D="$HOME/Downloads"
  LIBS="$D/asap7sc7p5t_SIMPLE_LVT_FF_nldm_201020.lib,$D/asap7sc7p5t_INVBUF_LVT_FF_nldm_201020.lib,$D/asap7sc7p5t_AO_LVT_FF_nldm_201020.lib,$D/asap7sc7p5t_OA_LVT_FF_nldm_201020.lib,$D/asap7sc7p5t_SEQ_LVT_FF_nldm_220123.lib"
fi
IFS=',' read -ra LIBARR <<< "$LIBS"
for lib in "${LIBARR[@]}"; do
  [[ -f "$lib" ]] || { echo "liberty not found: $lib" >&2; exit 1; }
done
echo "liberty: ${#LIBARR[@]} files"

# Default input: in-tree fifo example.
if [[ ${#EXTRA_SRCS[@]} -eq 0 ]]; then
  FIFO="$REPO/circt-sfc-experiments/example/fifo/fifo.v"
  [[ -f "$FIFO" ]] || { echo "missing $FIFO" >&2; exit 1; }
  SRCS=("$FIFO")
else
  SRCS=("${EXTRA_SRCS[@]}")
fi

mkdir -p "$WORK"
DESIGN="$WORK/design.mlir"; MAPPED="$WORK/mapped.mlir"
LOWERED="$WORK/lowered.mlir"
SDC="$WORK/out.sdc"; NET="$WORK/net.v"

echo "==> [1/5] circt-verilog --top=$TOP"
# --infer-clock-ports=false keeps clocks as i1 (flops use seq.to_clock).
# SeqTechMapper then unwraps to_clock directly; with !seq.clock ports it
# inserts seq.from_clock, which synth-emit-sdc cannot trace through yet
# (all clocks unconstrained -> STA "No paths found").
"$BIN/circt-verilog" --top="$TOP" --infer-clock-ports=false "${SRCS[@]}" -o "$DESIGN"

echo "==> [2/5] circt-synth (seq-map before comb lowering, strip-lib-to-extern)"
SYNTH_ARGS=(--liberty "$LIBS" --strip-lib-to-extern --top "$TOP"
  --sdc-clock-period "$PERIOD" --sdc-output "$SDC")
[[ -n "$SRAM_MACROS" ]] && SYNTH_ARGS+=(--sram-macros "$SRAM_MACROS")
time "$BIN/circt-synth" "${SYNTH_ARGS[@]}" "$DESIGN" -o "$MAPPED"

echo "==> [3/5] leftover + count check"
for pat in 'seq.firmem' 'seq.firreg' 'synth.aig' 'llhd.'; do
  n=$(grep -c "$pat" "$MAPPED" || true)
  echo "  $pat: $n"
done
INST=$(grep -c 'hw.instance' "$MAPPED" || true)
FLOPS=$(grep -cE '@(DFF|DFFR|DFFE|SDF|EDF|DFFHQN|SDFF)' "$MAPPED" || true)
echo "  hw.instance: $INST (flop-ish: $FLOPS)"

echo "==> [4/5] lower-seq-to-sv + export verilog (disallowReplicate mandatory for STA/ORFS)"
# SeqTechMapper leaves !seq.clock ports + seq.from_clock for cell CLK pins;
# ExportVerilog cannot emit those, so lower seq clock ops first.
"$BIN/circt-opt" --lower-seq-to-sv "$MAPPED" -o "$LOWERED"
"$BIN/circt-opt" --test-apply-lowering-options=options=disallowReplicate \
  --export-verilog "$LOWERED" -o /dev/null > "$NET" 2>"$WORK/export.log"
wc -l "$NET"
"$BIN/circt-verilog" --parse-only "$NET" && echo "  parse-only: clean"

if [[ "$RUN_STA" -eq 1 ]] && command -v sta >/dev/null 2>&1; then
  echo "==> [5/5] opensta report_checks"
  {
    for lib in "${LIBARR[@]}"; do echo "read_liberty $lib"; done
    echo "read_verilog $NET"
    echo "link_design $TOP"
    echo "read_sdc $SDC"
    echo "report_checks -path_delay max -digits 6"
  } > "$WORK/run.tcl"
  sta -no_init -exit "$WORK/run.tcl" > "$WORK/sta.log" 2>&1 || { echo "STA failed, see $WORK/sta.log" >&2; exit 1; }
  slack=$(grep -oE '\-?[0-9]+\.[0-9]+   slack' "$WORK/sta.log" | grep -oE '\-?[0-9]+\.[0-9]+' | sort -n | head -1)
  arrival=$(grep -oE '[0-9]+\.[0-9]+   data arrival time' "$WORK/sta.log" | grep -oE '[0-9]+\.[0-9]+' | sort -rn | head -1)
  echo "  worst-slack=${slack:-?}ns arrival=${arrival:-?}ns (period $PERIOD ns; see $WORK/sta.log)"
else
  echo "==> [5/5] STA skipped (no sta binary or --no-sta)"
fi

echo "DONE: mapped=$MAPPED net=$NET sdc=$SDC work=$WORK"
