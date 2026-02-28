#!/usr/bin/env python3
"""Benchmark CutRewriter vs. ABC LUT mapping."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def find_top_module(mlir_path: Path) -> str:
    text = mlir_path.read_text()
    match = re.search(r"hw\.module\s+@([^\s(]+)", text)
    if not match:
        raise ValueError(f"No hw.module could be inferred from {mlir_path}")
    return match.group(1)


def resolve_tool(name: str, search_dir: Path | None) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    if search_dir:
        candidate = search_dir / name
        if candidate.exists():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f"Could not find executable '{name}'")


def run_command(cmd: list[str], **kwargs) -> tuple[str, str, float]:
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    duration = time.perf_counter() - start
    result.check_returncode()
    return result.stdout, result.stderr, duration


def parse_mlir_timing(stderr: str) -> dict[str, float]:
    """Parse per-pass wall times from MLIR --mlir-timing stderr output.

    Returns a dict mapping pass name -> wall time in seconds.
    """
    timings: dict[str, float] = {}
    in_table = False
    for line in stderr.splitlines():
        if "----Wall Time----" in line:
            in_table = True
            continue
        if not in_table:
            continue
        # Lines look like: "    0.0240 ( 92.6%)  ABCRunner"
        m = re.match(r"\s+(\d+\.\d+)\s+\(\s*[\d.]+%\)\s+(.+)", line)
        if not m:
            continue
        seconds = float(m.group(1))
        name = m.group(2).strip()
        # Skip aggregate/wrapper entries
        if (
            name in ("root", "Total")
            or name.startswith("Pipeline Collection")
            or name.endswith("Pipeline")
        ):
            continue
        timings[name] = seconds
    return timings


def parse_abc_print_stats(stdout: str) -> dict[str, int | None]:
    """Parse ABC print_stats output for LUT depth (lev)."""
    depth = None
    for line in stdout.splitlines():
        # Typical print_stats contains: "... lev = <num>"
        match = re.search(r"\blev\s*=\s*(\d+)", line)
        if match:
            depth = int(match.group(1))
            break
    return {"lut_depth": depth}


def run_benchmark(
    max_lut_size: int,
    max_cuts_per_root: int,
    circt_synth: Path,
    circt_opt: Path,
    abc_path: Path,
    output_dir: Path,
    single_module_mlir: Path,
    top_module: str,
) -> dict[str, Any]:
    """Run a single benchmark configuration."""
    config_dir = output_dir / f"lut{max_lut_size}_cuts{max_cuts_per_root}"
    config_dir.mkdir(parents=True, exist_ok=True)

    # LUT mapper pipeline (run first to avoid thermal/cache effects from ABC)
    pipeline = (
        "builtin.module(hw.module(synth-generic-lut-mapper{"
        f"max-lut-size={max_lut_size} "
        f"max-cuts-per-root={max_cuts_per_root}"
        "}))"
    )
    lut_result_mlir = config_dir / "lut_result.mlir"
    lut_log = config_dir / "lut.log"
    stdout, stderr, _ = run_command(
        [
            str(circt_opt),
            str(single_module_mlir),
            "--pass-pipeline",
            pipeline,
            "--mlir-timing",
            "--mlir-timing-display=list",
            "-o",
            str(lut_result_mlir),
        ]
    )
    lut_pass_timings = parse_mlir_timing(stderr)
    log_content = stdout
    if stderr:
        log_content += "\n=== stderr ===\n" + stderr
    lut_log.write_text(log_content)

    # ABC pipeline (abc -K = max LUT inputs, -C = max cuts per node)
    abc_mlir = config_dir / "abc_result.mlir"
    abc_log = config_dir / "abc.log"
    abc_pipeline = (
        "builtin.module(hw.module("
        "synth-structural-hash, "
        "synth-abc-runner{"
        f"abc-path={abc_path} "
        f'abc-commands="if -K {max_lut_size} -C {max_cuts_per_root}; time; print_stats; strash"'
        "}))"
    )
    stdout, stderr, _ = run_command(
        [
            str(circt_opt),
            str(single_module_mlir),
            "--pass-pipeline",
            abc_pipeline,
            "--mlir-timing",
            "--mlir-timing-display=list",
            "-o",
            str(abc_mlir),
        ]
    )
    abc_pass_timings = parse_mlir_timing(stderr)
    abc_print_stats = parse_abc_print_stats(stdout)
    log_content = stdout
    if stderr:
        log_content += "\n=== stderr ===\n" + stderr
    abc_log.write_text(log_content)

    abc_analysis = run_analysis(circt_opt, "abc", abc_mlir, top_module, config_dir)
    lut_analysis = run_analysis(
        circt_opt, "lut", lut_result_mlir, top_module, config_dir
    )

    return {
        "config": {
            "max_lut_size": max_lut_size,
            "max_cuts_per_root": max_cuts_per_root,
        },
        "artifacts": {
            "abc_mlir": str(abc_mlir),
            "lut_mlir": str(lut_result_mlir),
            "abc_log": str(abc_log),
            "lut_log": str(lut_log),
        },
        "abc": {
            "pass_timings_s": abc_pass_timings,
            "pipeline": abc_pipeline,
            **abc_print_stats,
            **abc_analysis,
        },
        "lut": {
            "pass_timings_s": lut_pass_timings,
            **lut_analysis,
        },
    }


def run_analysis(
    circt_opt: Path, name: str, mlir: Path, top_module: str, out_dir: Path
) -> dict[str, Any]:
    longest_json = out_dir / f"{name}_longest_path.json"
    resource_json = out_dir / f"{name}_resource_usage.json"

    longest_arg = (
        f"top-module-name={top_module} output-file={longest_json} emit-json=true"
        " show-top-k-percent=100"
    )
    longest_cmd = [
        str(circt_opt),
        str(mlir),
        f"--synth-print-longest-path-analysis={longest_arg}",
        "-o",
        os.devnull,
    ]
    run_command(longest_cmd)

    resource_arg = (
        f"top-module-name={top_module} output-file={resource_json} emit-json=true"
    )
    resource_cmd = [
        str(circt_opt),
        str(mlir),
        f"--synth-print-resource-usage-analysis={resource_arg}",
        "-o",
        os.devnull,
    ]
    run_command(resource_cmd)

    long_delay = parse_longest_path(longest_json, top_module)
    resources = parse_resource_usage(resource_json, top_module)
    gate_count = sum(resources.values())
    return {
        "longest_delay": long_delay,
        "gate_count": gate_count,
        "resources": resources,
        "longest_json": str(longest_json),
        "resource_json": str(resource_json),
    }


def parse_longest_path(path: Path, module: str) -> int | None:
    data = json.loads(path.read_text())
    for entry in data:
        if entry.get("module_name") == module:
            top_paths = entry.get("top_paths", [])
            if top_paths:
                return top_paths[0].get("path", {}).get("delay")
    return None


def parse_resource_usage(path: Path, module: str) -> dict[str, int]:
    data = json.loads(path.read_text())
    for entry in data:
        if entry.get("moduleName") == module:
            return entry.get("total", {})
    return {}


def extract_first_module(src: Path, dst: Path) -> None:
    text = src.read_text()
    first = text.find("\n  hw.module @")
    if first == -1:
        dst.write_text(text)
        return
    second = text.find("\n  hw.module @", first + 1)
    if second == -1:
        dst.write_text(text)
        return
    trimmed = text[:second].rstrip()
    if not trimmed.endswith("}"):
        trimmed += "\n"
    trimmed += "\n}\n"
    dst.write_text(trimmed)


def run_perf_profile(
    cmd: list[str],
    out_dir: Path,
    flamegraph_pl: Path,
    label: str,
) -> Path:
    """Run `cmd` under perf record and produce a flamegraph SVG."""
    perf = shutil.which("perf")
    if not perf:
        raise FileNotFoundError("'perf' not found in PATH")

    perf_data = out_dir / f"{label}.perf.data"
    folded = out_dir / f"{label}.folded"
    svg = out_dir / f"{label}.flamegraph.svg"

    subprocess.run(
        [perf, "record", "-g", "-F", "997", "-o", str(perf_data), "--"] + cmd,
        check=True,
    )
    script_result = subprocess.run(
        [perf, "script", "-i", str(perf_data)],
        capture_output=True,
        text=True,
        check=True,
    )

    # stackcollapse-perf.pl if available, else fall back to perf script raw
    stackcollapse = flamegraph_pl.parent / "stackcollapse-perf.pl"
    if stackcollapse.exists():
        collapse_result = subprocess.run(
            ["perl", str(stackcollapse)],
            input=script_result.stdout,
            capture_output=True,
            text=True,
            check=True,
        )
        folded.write_text(collapse_result.stdout)
    else:
        folded.write_text(script_result.stdout)

    flamegraph_result = subprocess.run(
        ["perl", str(flamegraph_pl), "--title", label, str(folded)],
        capture_output=True,
        text=True,
        check=True,
    )
    svg.write_text(flamegraph_result.stdout)
    return svg


def parse_int_list(value: str) -> list[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run circt-synth + circt-opt LUT mapping vs. ABC if-cut benchmark"
    )
    parser.add_argument("--input-mlir", type=Path, help="Pre-synthesis MLIR input")
    parser.add_argument(
        "--post-synth-mlir", type=Path, help="Existing post-synthesis MLIR"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench-results"))
    parser.add_argument(
        "--circt-bin",
        type=Path,
        default=Path("build/bin"),
        help="Directory prefix for circt binaries",
    )
    parser.add_argument(
        "--abc-path",
        default=os.getenv("ABC_PATH", "/home/uenoku/dev/yosys/yosys-abc"),
        help="Path to the ABC executable (defaults to yosys-abc)",
    )
    parser.add_argument(
        "--max-lut-size",
        type=int,
        default=6,
        help="Maximum number of LUT inputs (ABC -K, LUT mapper max-lut-size)",
    )
    parser.add_argument(
        "--max-lut-sizes",
        type=parse_int_list,
        help="Comma-separated list of max-lut-size values to sweep",
    )
    parser.add_argument(
        "--max-cuts-per-root",
        type=int,
        default=8,
        help="Maximum cuts per node (ABC -C, LUT mapper max-cuts-per-root)",
    )
    parser.add_argument(
        "--max-cuts-per-root-list",
        type=parse_int_list,
        help="Comma-separated list of max-cuts-per-root values to sweep",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional output path for a PNG plot",
    )
    parser.add_argument(
        "--profile",
        choices=["abc", "lut"],
        help="Profile the specified tool using perf record + flamegraph for "
        "the first max_lut_size/max_cuts_per_root configuration",
    )
    parser.add_argument(
        "--flamegraph-pl",
        type=Path,
        default=Path(os.getenv("FLAMEGRAPH_PL", "FlameGraph/flamegraph.pl")),
        help="Path to flamegraph.pl (from https://github.com/brendangregg/FlameGraph)",
    )
    args = parser.parse_args()

    # Determine which configurations to run
    max_lut_sizes = args.max_lut_sizes if args.max_lut_sizes else [args.max_lut_size]
    max_cuts_per_root_list = (
        args.max_cuts_per_root_list
        if args.max_cuts_per_root_list
        else [args.max_cuts_per_root]
    )

    if not args.input_mlir and not args.post_synth_mlir:
        parser.error(
            "At least one of --input-mlir or --post-synth-mlir must be provided"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    circt_bin = args.circt_bin.expanduser().resolve()
    circt_synth = resolve_tool("circt-synth", circt_bin)
    circt_opt = resolve_tool("circt-opt", circt_bin)
    abc_path = resolve_tool(args.abc_path, None)

    base_mlir = args.post_synth_mlir or args.input_mlir
    top_module = find_top_module(base_mlir)

    post_synth_mlir = (
        args.post_synth_mlir
        if args.post_synth_mlir
        else output_dir / f"{base_mlir.stem}_post_synth.mlir"
    )

    if not args.post_synth_mlir and args.input_mlir:
        run_command(
            [
                str(circt_synth),
                str(args.input_mlir),
                "--until-before",
                "mapping",
                "-o",
                str(post_synth_mlir),
            ]
        )

    single_module_mlir = output_dir / f"{post_synth_mlir.stem}_single.mlir"
    extract_first_module(post_synth_mlir, single_module_mlir)

    # Run all configurations
    results = []
    for max_lut_size in max_lut_sizes:
        for max_cuts_per_root in max_cuts_per_root_list:
            print(
                f"Running benchmark: max_lut_size={max_lut_size}, "
                f"max_cuts_per_root={max_cuts_per_root}"
            )
            result = run_benchmark(
                max_lut_size=max_lut_size,
                max_cuts_per_root=max_cuts_per_root,
                circt_synth=circt_synth,
                circt_opt=circt_opt,
                abc_path=abc_path,
                output_dir=output_dir,
                single_module_mlir=single_module_mlir,
                top_module=top_module,
            )
            results.append(result)
            abc_timings = result["abc"]["pass_timings_s"]
            lut_timings = result["lut"]["pass_timings_s"]
            abc_str = ", ".join(
                f"{k}: {v * 1000:.1f}ms" for k, v in abc_timings.items()
            )
            lut_str = ", ".join(
                f"{k}: {v * 1000:.1f}ms" for k, v in lut_timings.items()
            )
            abc_depth = result["abc"].get("lut_depth")
            lut_depth = result["lut"].get("longest_delay")
            print(f"  ABC  [{abc_str}]")
            if abc_depth is not None:
                print(f"       LUT depth: {abc_depth}")
            print(f"  LUT  [{lut_str}]")
            if lut_depth is not None:
                print(f"       LUT depth: {lut_depth}")

    # Optional profiling run (first configuration only)
    if args.profile:
        max_lut_size = max_lut_sizes[0]
        max_cuts_per_root = max_cuts_per_root_list[0]
        config_dir = output_dir / f"lut{max_lut_size}_cuts{max_cuts_per_root}"
        config_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\nProfiling {args.profile.upper()} "
            f"(max-lut-size={max_lut_size}, max-cuts-per-root={max_cuts_per_root}) ..."
        )

        if args.profile == "abc":
            profile_pipeline = (
                "builtin.module(hw.module("
                "synth-structural-hash, "
                "synth-abc-runner{"
                f"abc-path={abc_path} "
                f'abc-commands="if -K {max_lut_size} -C {max_cuts_per_root}; time; print_stats; strash"'
                "}))"
            )
            profile_cmd = [
                str(circt_opt),
                str(single_module_mlir),
                "--pass-pipeline",
                profile_pipeline,
                "-o",
                os.devnull,
            ]
        else:
            profile_pipeline = (
                "builtin.module(hw.module(synth-generic-lut-mapper{"
                f"max-lut-size={max_lut_size} "
                f"max-cuts-per-root={max_cuts_per_root}"
                "}))"
            )
            profile_cmd = [
                str(circt_opt),
                str(single_module_mlir),
                "--pass-pipeline",
                profile_pipeline,
                "-o",
                os.devnull,
            ]

        svg = run_perf_profile(
            profile_cmd,
            out_dir=config_dir,
            flamegraph_pl=args.flamegraph_pl.expanduser().resolve(),
            label=f"{args.profile}_lut{max_lut_size}_cuts{max_cuts_per_root}",
        )
        print(f"Flamegraph written to {svg}")

    # Write aggregated results
    master_summary = {
        "total_configs": len(results),
        "max_lut_sizes": max_lut_sizes,
        "max_cuts_per_root_values": max_cuts_per_root_list,
        "results": results,
    }

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(master_summary, indent=2))
    print(f"\nBenchmark finished. Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
