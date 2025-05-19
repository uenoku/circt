#!/usr/bin/env python3

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import subprocess
from pathlib import Path


def export_markdown_to_pdf(markdown_path, output_dir):
    """
    Export markdown file to PDF using pandoc if available.

    Args:
        markdown_path: Path to the markdown file
        output_dir: Directory where the PDF will be saved

    Returns:
        Path to the generated PDF file or None if conversion failed
    """
    try:
        # Check if pandoc is installed
        subprocess.run(
            ["pandoc", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Get the filename without extension
        markdown_filename = os.path.basename(markdown_path)
        pdf_filename = os.path.splitext(markdown_filename)[0] + ".pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)

        # Convert markdown to PDF using pandoc
        cmd = [
            "pandoc",
            markdown_path,
            "-o",
            pdf_path,
            "--pdf-engine=xelatex",
            "-V",
            "geometry:margin=1in",
            "--standalone",
        ]

        print(f"Converting markdown to PDF using command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"PDF report generated: {pdf_path}")
        return pdf_path

    except subprocess.CalledProcessError as e:
        print(f"Error running pandoc: {e}")
        return None
    except FileNotFoundError:
        print("Pandoc not found. Please install pandoc to enable PDF export.")
        print("Installation instructions: https://pandoc.org/installing.html")
        return None


def load_freq_data(filename):
    """Load frequency data from JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    # Return the full data structure
    return data


def create_histogram(
    data,
    log_scale=False,
    bins=None,
    module_name=None,
    path_type=None,
    max_delay=None,
    min_delay=None,
):
    """Create histogram with cumulative distribution overlay."""
    # Extract delay and frequency values
    delays = [item["delay"] for item in data]
    freqs = [item["freq"] for item in data]

    # Apply delay filters if specified
    if min_delay is not None or max_delay is not None:
        filtered_data = []
        for d, f in zip(delays, freqs):
            if (min_delay is None or d >= min_delay) and (
                max_delay is None or d <= max_delay
            ):
                filtered_data.append({"delay": d, "freq": f})

        if filtered_data:
            delays = [item["delay"] for item in filtered_data]
            freqs = [item["freq"] for item in filtered_data]
        else:
            print(
                f"Warning: No data points after filtering by delay range for {module_name} ({path_type})."
            )
            return None, None, None

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine bin edges if not specified
    if bins is None:
        # Use more intelligent binning based on data range
        range_size = max(delays) - min(delays)
        if range_size > 100:
            bin_size = max(1, range_size // 100)  # Aim for ~100 bins
            bins = np.arange(min(delays), max(delays) + bin_size + 1, bin_size) - 0.5
        else:
            bins = np.arange(min(delays), max(delays) + 2) - 0.5

    # Create histogram
    n, bins, patches = ax.hist(
        delays,
        bins=bins,
        weights=freqs,
        alpha=0.75,
        color="steelblue",
        label="Frequency",
    )

    # Calculate percentiles for vertical lines
    total_freq = sum(freqs)
    sorted_data = sorted(zip(delays, freqs), key=lambda x: x[0])
    cum_freqs = []
    running_sum = 0
    percentile_values = [50, 90, 95, 99, 99.9]
    percentiles = {}

    for d, f in sorted_data:
        running_sum += f
        cum_freqs.append(running_sum)

    for p in percentile_values:
        threshold = total_freq * p / 100
        for i, cum_freq in enumerate(cum_freqs):
            if cum_freq >= threshold:
                percentiles[p] = sorted_data[i][0]
                break

    # Add percentile lines
    percentile_colors = {
        50: "green",
        90: "orange",
        95: "red",
        99: "purple",
        99.9: "black",
    }

    # Add vertical lines for percentiles
    for p, delay in percentiles.items():
        ax.axvline(
            x=delay,
            color=percentile_colors[p],
            linestyle="--",
            alpha=0.7,
            label=f"{p}th percentile: {delay}",
        )

    # Add cumulative distribution as a line on the same axis
    # Sort data by delay (ascending order)
    cum_delays = [d for d, f in sorted_data]

    # Create a second y-axis for cumulative distribution
    ax2 = ax.twinx()

    # Normalize cumulative frequencies to percentage
    cum_freqs_percent = [cf / total_freq * 100 for cf in cum_freqs]

    # Plot cumulative line
    ax2.plot(cum_delays, cum_freqs_percent, "r-", linewidth=2, label="Cumulative %")
    ax2.set_ylabel("Cumulative Percentage (%)", color="r")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="r")

    # Set log scale if requested (only for frequency axis, not cumulative)
    if log_scale:
        ax.set_yscale("log")

    # Add labels and title
    ax.set_xlabel("Delay")
    ax.set_ylabel("Frequency")
    title = f"{path_type} Delay Distribution"
    if module_name:
        title += f" for {module_name}"
    if log_scale:
        title += " (Log Scale)"
    ax.set_title(title)

    # Add legends for both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Add grid
    ax.grid(True, alpha=0.3)

    return fig, ax, percentiles


def calculate_statistics(data, module_name, path_type):
    """Calculate statistics for the given data."""
    delays = [item["delay"] for item in data]
    freqs = [item["freq"] for item in data]
    total_freq = sum(freqs)
    weighted_sum = sum(d * f for d, f in zip(delays, freqs))
    avg_delay = weighted_sum / total_freq if total_freq > 0 else 0

    # Calculate percentiles
    sorted_data = sorted(zip(delays, freqs), key=lambda x: x[0])
    cum_freqs = []
    running_sum = 0

    for _, f in sorted_data:
        running_sum += f
        cum_freqs.append(running_sum / total_freq * 100)

    percentiles = {}
    percentile_values = [50, 90, 95, 99, 99.9]

    for p in percentile_values:
        for i, cum_freq in enumerate(cum_freqs):
            if cum_freq >= p:
                percentiles[p] = sorted_data[i][0]
                break

    stats = {
        "module_name": module_name,
        "path_type": path_type,
        "total_freq": total_freq,
        "avg_delay": avg_delay,
        "min_delay": min(delays),
        "max_delay": max(delays),
        "percentiles": percentiles,
    }

    return stats


def print_statistics(stats):
    """Print statistics to console."""
    print(f"Module: {stats['module_name']}")
    print(f"Path Type: {stats['path_type']}")
    print(f"Total path count: {stats['total_freq']}")
    print(f"Average delay: {stats['avg_delay']:.2f}")
    print(f"Min delay: {stats['min_delay']}")
    print(f"Max delay: {stats['max_delay']}")
    print("\nPercentiles:")
    for p, v in stats["percentiles"].items():
        print(f"{p}th percentile: {v}")
    print("\n" + "-" * 50 + "\n")


def generate_summary_report(all_stats, output_dir):
    """Generate a summary report in markdown format."""
    report_path = os.path.join(output_dir, "delay_summary_report.md")

    # Load the original JSON data to access port delay information
    with open(all_stats[0].get("source_file", "result.md.json"), "r") as f:
        full_data = json.load(f)

    with open(report_path, "w") as f:
        f.write("# Delay Distribution Summary Report\n\n")

        # Group stats by module
        modules = {}
        for stats in all_stats:
            module_name = stats["module_name"]
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append(stats)

        # Create summary table for each module
        for module_name, module_stats in modules.items():
            f.write(f"## {module_name}\n\n")
            f.write(
                "| Path Type | Avg Delay | Min | Max | 50th | 90th | 95th | 99th | 99.9th |\n"
            )
            f.write(
                "|-----------|-----------|-----|-----|------|------|------|------|--------|\n"
            )

            for stats in module_stats:
                f.write(
                    f"| {stats['path_type']} | {stats['avg_delay']:.2f} | {stats['min_delay']} | {stats['max_delay']} "
                )
                for p in [50, 90, 95, 99, 99.9]:
                    f.write(f"| {stats['percentiles'].get(p, 'N/A')} ")
                f.write("|\n")

            f.write("\n")

        # Add detailed sections for each module and path type
        f.write("\n## Detailed Statistics\n\n")

        for module_name, module_stats in modules.items():
            f.write(f"### {module_name}\n\n")

            # Find the module data in the original JSON
            module_data = None
            for data in full_data:
                if data.get("moduleName") == module_name:
                    module_data = data
                    break

            for stats in module_stats:
                path_type = stats["path_type"]

                f.write(f"#### {path_type}\n\n")
                f.write(f"- **Total frequency count:** {stats['total_freq']}\n")
                f.write(f"- **Average delay:** {stats['avg_delay']:.2f}\n")
                f.write(f"- **Min delay:** {stats['min_delay']}\n")
                f.write(f"- **Max delay:** {stats['max_delay']}\n")

                f.write("\n**Percentiles:**\n\n")
                for p, v in stats["percentiles"].items():
                    f.write(f"- {p}th percentile: {v}\n")

                # Add port delay information for relevant path types
                if (
                    path_type == "Open Paths to FF"
                    and module_data
                    and "inputPortDelay" in module_data
                ):
                    f.write("\n**Input Port Delays:**\n\n")
                    f.write("| Port Name | Average Max Delay | Max Delay |\n")
                    f.write("|-----------|------------------|----------|\n")

                    for port in module_data["inputPortDelay"]:
                        f.write(
                            f"| {port.get('name', 'N/A')} | {port.get('averageMaxDelay', 'N/A')} | {port.get('maxDelay', 'N/A')} |\n"
                        )

                if (
                    path_type == "Open Paths from Output Ports"
                    and module_data
                    and "outputPortDelay" in module_data
                ):
                    f.write("\n**Output Port Delays:**\n\n")
                    f.write("| Port Name | Average Max Delay | Max Delay |\n")
                    f.write("|-----------|------------------|----------|\n")

                    for port in module_data["outputPortDelay"]:
                        f.write(
                            f"| {port.get('name', 'N/A')} | {port.get('averageMaxDelay', 'N/A')} | {port.get('maxDelay', 'N/A')} |\n"
                        )

                # Use just the filenames without URL encoding - they're in the same directory
                if "plot_paths" in stats:
                    # Regular plot with cumulative overlay
                    if "regular" in stats["plot_paths"]:
                        f.write(f"\n**Regular Distribution:**\n\n")
                        f.write(
                            f"![Delay Distribution](./{stats['plot_paths']['regular']})\n\n"
                        )

                    # Log scale plot with cumulative overlay
                    if "log" in stats["plot_paths"]:
                        f.write(f"**Log Scale Distribution:**\n\n")
                        f.write(
                            f"![Log Scale Delay Distribution](./{stats['plot_paths']['log']})\n\n"
                        )

                f.write("---\n\n")

    print(f"Summary report generated: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize frequency distribution from JSON file"
    )
    parser.add_argument(
        "filename",
        default="result.md.json",
        nargs="?",
        help="JSON file with frequency data",
    )
    parser.add_argument("--bins", type=int, help="Number of bins for histogram")
    parser.add_argument(
        "--output-dir", default="delay_plots", help="Directory to save output files"
    )
    parser.add_argument(
        "--min-delay", type=int, help="Minimum delay to include in visualization"
    )
    parser.add_argument(
        "--max-delay", type=int, help="Maximum delay to include in visualization"
    )
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="Export markdown report to PDF (requires pandoc)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load data
    full_data = load_freq_data(args.filename)

    # Process each module
    all_stats = []

    for module_data in full_data:
        if "moduleName" not in module_data:
            continue

        module_name = module_data.get("moduleName", "Unknown")
        print(f"Processing module: {module_name}")

        # Create a safe module name for filenames
        safe_module_name = (
            module_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        )

        # Process each path type
        path_types = {
            "Closed Paths": "closedPaths",
            "Open Paths to FF": "openPathsToFF",
            "Open Paths from Output Ports": "openPathsFromOutputPorts",
        }

        for display_name, field_name in path_types.items():
            if field_name not in module_data:
                print(f"  Warning: {field_name} not found in module {module_name}")
                continue

            path_data = module_data[field_name]
            if not path_data:
                print(f"  Warning: No data in {field_name} for module {module_name}")
                continue

            print(f"  Processing {display_name}...")

            # Create a safe path type for filenames
            safe_path_type = (
                display_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            )

            # Calculate statistics
            stats = calculate_statistics(path_data, module_name, display_name)
            stats["plot_paths"] = {}  # Initialize plot paths dictionary
            stats["source_file"] = (
                args.filename
            )  # Store source filename for later reference

            # Print statistics to console
            print_statistics(stats)

            # Create and save regular histogram with cumulative overlay
            fig, ax, percentiles = create_histogram(
                path_data,
                log_scale=False,
                bins=args.bins,
                module_name=module_name,
                path_type=display_name,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
            )

            if fig is not None:
                # Use safe filenames without spaces or special characters
                regular_filename = (
                    f"{safe_module_name}_{safe_path_type}_delay_distribution.png"
                )
                output_file = os.path.join(args.output_dir, regular_filename)
                fig.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  Plot saved to {output_file}")

                # Store just the filename, not the full path
                stats["plot_paths"]["regular"] = regular_filename

                # Update percentiles in stats
                stats["percentiles"] = percentiles

                # Create and save log scale histogram with cumulative overlay
                fig, ax, _ = create_histogram(
                    path_data,
                    log_scale=True,
                    bins=args.bins,
                    module_name=module_name,
                    path_type=display_name,
                    min_delay=args.min_delay,
                    max_delay=args.max_delay,
                )

                if fig is not None:
                    log_filename = f"{safe_module_name}_{safe_path_type}_delay_distribution_log.png"
                    output_file = os.path.join(args.output_dir, log_filename)
                    fig.savefig(output_file, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  Log plot saved to {output_file}")
                    stats["log_plot"] = True
                    stats["plot_paths"]["log"] = log_filename

                all_stats.append(stats)

    # Generate summary report
    markdown_path = os.path.join(args.output_dir, "delay_summary_report.md")
    generate_summary_report(all_stats, args.output_dir)

    # Export to PDF if requested
    if args.export_pdf:
        export_markdown_to_pdf(markdown_path, args.output_dir)


if __name__ == "__main__":
    main()
