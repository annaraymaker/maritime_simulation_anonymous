#!/usr/bin/env python3
"""
plot_results.py
===============
Plot time-to-infection versus infection rate for all monitored maritime
chokepoints.

Reads consolidated simulation results from a JSON file
(``results/infection_rate_results.json``) and produces a scatter plot with:
  - One colour per chokepoint
  - A black × marker for the per-rate average
  - A quadratic trend line illustrating diminishing returns at higher rates

Output is saved as a PDF.

Usage
-----
    python plot_results.py [--results results/infection_rate_results.json]
                           [--output  infection_rates.pdf]
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_infection_rates(results: dict, output_path: str = "infection_rates.pdf"):
    """
    Generate and save the infection-rate vs. time-to-infection scatter plot.

    Parameters
    ----------
    results : dict
        Nested dict with structure:
        ``{rate_label: {chokepoint: {time_to_infection_hours: float, ...}, ...}, ...}``
    output_path : str
        Path for the output PDF.
    """
    # Determine chokepoint ordering from the first rate entry
    first_rate = next(iter(results.values()))
    chokepoints = list(first_rate.keys())
    rate_labels = list(results.keys())

    # Build (x_position, y_values_per_chokepoint) arrays
    x_positions = np.array([float(label.replace("%", "")) for label in rate_labels])
    y_matrix = []   # shape: (n_rates, n_chokepoints)

    for label in rate_labels:
        row = []
        for cp in chokepoints:
            entry = results[label].get(cp, {})
            row.append(entry.get("time_to_infection_hours", float("nan")))
        y_matrix.append(row)
    y_matrix = np.array(y_matrix)  # (n_rates, n_chokepoints)

    colors = plt.cm.tab10(np.linspace(0, 1, len(chokepoints)))

    fig, ax = plt.subplots(figsize=(7, 4))

    # Scatter: one colour per chokepoint, one x-position per rate
    for j, cp in enumerate(chokepoints):
        for i, x in enumerate(x_positions):
            ax.scatter(x, y_matrix[i, j], color=colors[j], alpha=0.7, s=50)

    # Per-rate averages (black ×)
    averages = np.nanmean(y_matrix, axis=1)
    for x, avg in zip(x_positions, averages):
        ax.scatter(x, avg, marker="x", color="black", s=100)

    # Quadratic trend through averages
    valid = ~np.isnan(averages)
    coeffs = np.polyfit(x_positions[valid], averages[valid], deg=2)
    x_fit = np.linspace(x_positions.min(), x_positions.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)
    (trend_line,) = ax.plot(x_fit, y_fit, color="gray", linewidth=2, label="Trend")

    # Legend
    legend_elems = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            label=cp.replace("_", " "),
            markerfacecolor=colors[j], markersize=8,
        )
        for j, cp in enumerate(chokepoints)
    ]
    legend_elems.append(
        plt.Line2D([0], [0], marker="x", color="black", label="Average",
                   markersize=8, linestyle="None")
    )
    legend_elems.append(trend_line)

    ax.legend(handles=legend_elems, loc="upper right", ncol=2, fontsize=8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{label}" for label in rate_labels])
    ax.set_xlabel("Infection Rate (%)")
    ax.set_ylabel("Time to Infection (hours)")
    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    print(f"Plot saved to {output_path}")
    plt.show()

    # Print per-rate averages in days
    print("\nPer-rate average time-to-infection:")
    for label, avg in zip(rate_labels, averages):
        print(f"  {label:6s}: {avg / 24:.1f} days ({avg:.1f} hours)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot time-to-infection vs. infection rate for all chokepoints."
    )
    parser.add_argument(
        "--results",
        default="results/infection_rate_results.json",
        help="Path to consolidated JSON results file (default: results/infection_rate_results.json).",
    )
    parser.add_argument(
        "--output",
        default="infection_rates.pdf",
        help="Output PDF path (default: infection_rates.pdf).",
    )
    args = parser.parse_args()

    try:
        with open(args.results) as f:
            results = json.load(f)
    except OSError as exc:
        print(f"Cannot read results file: {exc}")
        return

    plot_infection_rates(results, output_path=args.output)


if __name__ == "__main__":
    main()
