#!/usr/bin/env python3
"""STA scoreboard: Yosys-clean netlist -> OpenSTA report_checks.

OpenSTA's Verilog parser rejects replication ``{N{...}}`` (even single-line),
which circt's ExportVerilog emits for wide concats (e.g. ``{4{...}}`` in
ChipTop). Yosys parses it fine, so we clean first::

    yosys -p "read_verilog NET; hierarchy -top TOP; write_verilog CLEAN"

then run OpenSTA ``report_checks`` on the clean netlist. Instance/area
numbers come from the authoritative ResourceUsageAnalysis
(``hw.instance`` / ``hw.instance@<cell>``, flattening-equivalent) instead of
Verilog regexes, which undercount shared submodules (e.g. fifo's 2x
addr_gen: grep finds 10194, analysis reports 10277 flattened).

Since CIRCT gained the `disallowReplicate` lowering option
(`comb.replicate` is emitted as an explicit concatenation instead of
`{N{...}}`), the Yosys cleaning step is no longer needed. Export directly:

    circt-opt --test-apply-lowering-options='options=disallowReplicate' \
        --export-verilog synth.mlir > clean.v

and pass --skip-yosys.

Usage:
    python3 tmp/sta_scoreboard.py --lib LIB --net NET --top TOP --clock CLK \
        [--period 10] [--workdir DIR] [--skip-yosys] [--stubs STUBS]

Prints a one-line scoreboard row plus the raw STA tail.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r


def yosys_clean(net, top, out):
    cmd = [
        "yosys",
        "-p",
        f"read_verilog {net}; hierarchy -top {top}; "
        f"write_verilog {out}",
    ]
    r = run(cmd)
    log = r.stdout + r.stderr
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"yosys clean failed:\n{log[-4000:]}")
    # Sanity: replication must be gone for the STA parser.
    with open(out) as f:
        head = f.read(2_000_000)
    m = re.search(r"\{\d+\{", head)
    if m:
        # Only warn: replication may exist past the head sample on huge nets.
        sys.stderr.write("warning: replication pattern still present\n")
    return log


def run_sta(lib, net, top, clock, period, stubs=""):
    libs = [x for x in lib.split(",") if x]
    tcl = "".join(f"read_liberty {l}\n" for l in libs)
    tcl += f"read_verilog {net}\n"
    if stubs:
        tcl += f"read_verilog {stubs}\n"
    tcl += (
        f"link_design {top}\n"
        f"create_clock -name clk -period {period} [get_ports {clock}]\n"
        f"report_checks -path_delay max -digits 4 -no_line_splits\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".tcl", delete=False
    ) as f:
        f.write(tcl)
        tcl_path = f.name
    try:
        r = run(["sta", "-no_init", "-exit", tcl_path])
    finally:
        os.unlink(tcl_path)
    return r.stdout + r.stderr


def lib_time_scale_to_ns(lib):
    """Read `time_unit` from the first liberty file; return (scale, unit)."""
    first = lib.split(",")[0]
    try:
        with open(first, errors="ignore") as f:
            head = f.read(50000)
    except OSError:
        return 1.0, "1ns?"
    m = re.search(r'time_unit\s*:\s*"1([a-z]+)"', head)
    unit = ("1" + m.group(1)) if m else "1ns?"
    scale = {"1ps": 0.001, "1ns": 1.0, "1us": 1000.0}.get(unit, 1.0)
    return scale, unit


def parse_sta(log, time_scale=1.0):
    """Extract worst slack/arrival and endpoints from report_checks."""
    slacks = [
        float(x) * time_scale
        for x in re.findall(r"^\s*([-\d.]+)\s+slack", log, re.M)
    ]
    arrivals = [
        float(x) * time_scale
        for x in re.findall(r"^\s*([-\d.]+)\s+data arrival time", log, re.M)
    ]
    starts = re.findall(r"Startpoint:\s*(\S+)", log)
    ends = re.findall(r"Endpoint:\s*(\S+)", log)
    return {
        "worst_slack": min(slacks) if slacks else None,
        "worst_arrival": max(arrivals) if arrivals else None,
        "num_paths": len(slacks),
        "startpoint": starts[0] if starts else None,
        "endpoint": ends[0] if ends else None,
    }


def resource_counts(mlir, top, circt_opt="./build/bin/circt-opt"):
    """Authoritative flattened instance counts via ResourceUsageAnalysis."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as f:
        js = f.name
    try:
        r = run(
            [
                circt_opt,
                f"--synth-print-resource-usage-analysis=top-module-name={top} "
                f"output-file={js} emit-json=true",
                mlir,
                "-o",
                "/dev/null",
            ]
        )
        if r.returncode != 0:
            return {"error": (r.stdout + r.stderr)[-2000:]}
        d = json.load(open(js))[0]
        total = d["total"]
        cells = {
            k.split("@", 1)[1]: v
            for k, v in total.items()
            if k.startswith("hw.instance@")
        }
        inv = sum(v for k, v in cells.items() if "inv" in k.lower())
        return {
            "instances": total.get("hw.instance"),
            "cells": len(cells),
            "inv": inv,
            "inv_frac": inv / total.get("hw.instance", 1),
            "flops": sum(
                v
                for k, v in cells.items()
                if re.search(r"df(f|t|x|r)|sdf|latch|flop", k, re.I)
            ),
        }
    finally:
        if os.path.exists(js):
            os.unlink(js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--clock", required=True)
    ap.add_argument("--period", default="10")
    ap.add_argument("--stubs", default="")
    ap.add_argument("--mlir", default="")
    ap.add_argument("--workdir", default="/tmp/e2e")
    ap.add_argument("--skip-yosys", action="store_true")
    ap.add_argument("--yosys-log", default="")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    clean = os.path.join(args.workdir, f"{args.top}_clean.v")
    if args.skip_yosys:
        clean = args.net
    else:
        log = yosys_clean(args.net, args.top, clean)
        if args.yosys_log:
            open(args.yosys_log, "w").write(log)
        print(f"[yosys] {args.net} -> {clean}")

    sta_log = run_sta(
        args.lib, clean, args.top, args.clock, args.period, args.stubs
    )
    scale, unit = lib_time_scale_to_ns(args.lib)
    sta = parse_sta(sta_log, scale)
    row = {
        "top": args.top,
        "libs": [os.path.basename(x) for x in args.lib.split(",") if x],
        "time_unit": unit,
        "net": os.path.basename(args.net),
        "period_ns": float(args.period) * scale,
        **sta,
    }
    if args.mlir:
        row["count"] = resource_counts(args.mlir, args.top)
    print(json.dumps(row, indent=2))
    print("---- STA tail ----")
    print("\n".join(sta_log.splitlines()[-25:]))


if __name__ == "__main__":
    main()
