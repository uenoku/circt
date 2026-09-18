#!/usr/bin/env python3
"""Lowering retry sweep: run circt-synth structural variants + STA.

Datapath lowering freedom lives in structural choices (datapath on/off,
timing-aware lowering, mapping strategy, SOP balancing), not in the
lowering-policy label (measured byte-identical on fifo/add32: mapping
normalizes it). This harness retries the axes that actually vary and
prints the area/delay Pareto table, reusing sta_scoreboard as a library.

Usage:
  python3 tmp/lowering_sweep.py --lib LIB --top fifo --mlir-in F \
      --clock clk --period 10 [--out DIR] [--combos a,b,c]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sta_scoreboard as sb

SYNTH = "./build/bin/circt-opt"
CIRCT_SYNTH = "./build/bin/circt-synth"
CIRCT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMBOS = {
    # name: extra circt-synth flags. Mapping strategy dominates counts;
    # lowering axes change structure where they bind.
    "base": [],
    "nodp": ["--disable-datapath"],
    "nota": ["--disable-timing-aware"],
    "sop": ["--enable-sop-balancing"],
    "map-area": ["--synthesis-strategy=area"],
    "nodp+map-area": ["--disable-datapath", "--synthesis-strategy=area"],
    "nota+map-area": ["--disable-timing-aware", "--synthesis-strategy=area"],
    "sop+map-area": ["--enable-sop-balancing", "--synthesis-strategy=area"],
    # Mapper effort axis (cut enumeration kept per root; default 6).
    "cut16": ["--max-cut-size-per-root=16"],
    "cut16+map-area": ["--max-cut-size-per-root=16",
                         "--synthesis-strategy=area"],
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def run_unconstrained(lib, net, top):
    """Unconstrained combinational max-delay report (no clock)."""
    libs = [x for x in lib.split(",") if x]
    tcl = "".join(f"read_liberty {l}\n" for l in libs)
    tcl += f"read_verilog {net}\nlink_design {top}\n"
    tcl += ("report_checks -path_delay max -unconstrained -digits 4 "
            "-no_line_splits\n")
    with sb.tempfile.NamedTemporaryFile(
            "w", suffix=".tcl", delete=False) as f:
        f.write(tcl)
        tcl_path = f.name
    try:
        r = subprocess.run(["sta", "-no_init", "-exit", tcl_path],
                           capture_output=True, text=True)
    finally:
        os.unlink(tcl_path)
    return r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--mlir-in", required=True)
    ap.add_argument("--clock", required=True)
    ap.add_argument("--period", default="10")
    ap.add_argument("--out", default="/tmp/e2e/sweep")
    ap.add_argument("--combos", default=",".join(COMBOS),
                    help="comma-separated subset of: " + ",".join(COMBOS))
    ap.add_argument("--synth-extra", default="",
                    help="extra flags prepended to every synth run")
    ap.add_argument("--arrival-target", default="",
                    help="constraint mode: keep the min-area combo with "
                         "worst arrival_ns <= target (ns)")
    ap.add_argument("--slack-target", default="",
                    help="constraint mode: keep the min-area combo with "
                         "worst slack_ns >= target (ns, needs a clock)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for name in args.combos.split(","):
        flags = COMBOS[name]
        mlir = os.path.join(args.out, f"{args.top}_{name}.mlir")
        verilog = os.path.join(args.out, f"{args.top}_{name}.v")
        cmd = (
            [CIRCT_SYNTH, "--liberty", args.lib, "--strip-lib-to-extern",
             "--top", args.top]
            + args.synth_extra.split()
            + flags
            + [args.mlir_in, "-o", mlir]
        )
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=CIRCT_ROOT)
        if r.returncode != 0 or not os.path.exists(mlir):
            print(f"[{name}] synth FAILED")
            continue
        # --export-verilog emits to stdout (scoreboard convention).
        r = subprocess.run(
            [SYNTH, "--test-apply-lowering-options="
             "options=disallowReplicate",
             "--export-verilog", mlir, "-o", "/dev/null"],
            capture_output=True, text=True, cwd=CIRCT_ROOT,
        )
        if r.returncode != 0:
            print(f"[{name}] export FAILED")
            continue
        with open(verilog, "w") as f:
            f.write(r.stdout)
        clean = os.path.join(args.out, f"{args.top}_{name}_clean.v")
        sb.yosys_clean(verilog, args.top, clean)
        if args.clock.upper() == "NONE":
            # Combinational top: unconstrained max-delay report.
            sta_log = run_unconstrained(args.lib, clean, args.top)
        else:
            sta_log = sb.run_sta(args.lib, clean, args.top, args.clock,
                                 args.period)
        scale, _ = sb.lib_time_scale_to_ns(args.lib)
        sta = sb.parse_sta(sta_log, scale)
        counts = sb.resource_counts(
            mlir, args.top,
            circt_opt=os.path.join(CIRCT_ROOT, "build/bin/circt-opt"))
        row = {"combo": name, "flags": " ".join(flags),
               "instances": counts.get("instances"),
               "flops": counts.get("flops"),
               "arrival_ns": sta["worst_arrival"],
               "slack_ns": sta["worst_slack"]}
        results.append(row)
        print(f"[{name}] inst={row['instances']} arrival={row['arrival_ns']} "
              f"slack={row['slack_ns']}")
    print(json.dumps(results, indent=2))
    with open(os.path.join(args.out, f"{args.top}_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    # Constraint mode: min area satisfying the timing target.
    pick = None
    if args.arrival_target:
        target = float(args.arrival_target)
        sat = [r for r in results
               if r["arrival_ns"] is not None
               and r["arrival_ns"] <= target]
        if sat:
            pick = min(sat, key=lambda r: (r["instances"] is None,
                                           r["instances"]))
            print(f"[constraint] arrival <= {target}ns:"
                  f" pick {pick['combo']} inst={pick['instances']}"
                  f" arrival={pick['arrival_ns']}")
        else:
            print(f"[constraint] arrival <= {target}ns:"
                  f" NOTHING SATISFIES")
    if args.slack_target:
        target = float(args.slack_target)
        sat = [r for r in results
               if r["slack_ns"] is not None and r["slack_ns"] >= target]
        if sat:
            pick = min(sat, key=lambda r: (r["instances"] is None,
                                           r["instances"]))
            print(f"[constraint] slack >= {target}ns:"
                  f" pick {pick['combo']} inst={pick['instances']}"
                  f" slack={pick['slack_ns']}")
        else:
            print(f"[constraint] slack >= {target}ns: NOTHING SATISFIES")


if __name__ == "__main__":
    main()
