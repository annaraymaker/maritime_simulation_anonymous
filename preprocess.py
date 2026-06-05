#!/usr/bin/env python3
"""
preprocess.py
=============
Sample a reproducible subset of vessels from raw AIS CSV data.

Processes CSV files in chronological order, accumulating new vessel
identifiers (MMSIs) as they appear and selecting a fixed fraction of each
new batch.  Because fractions are applied to the *same* shuffled list, the
1% selection is always a strict subset of the 2.5% selection, which is a
strict subset of the 5% selection, and so on—ensuring consistent vehicle
populations across infection-rate experiments.

Output is one text file per sample rate, each containing one MMSI per line.
These files are consumed by ``data_loader.py``.

Usage
-----
    python preprocess.py --data-dir /path/to/ais_csvs [--out-dir .]

Expected CSV columns (at minimum):
    ``mmsi``, ``latitude``, ``longitude``, ``position_updated_at``
    OR ``position_timestamp`` (see --timestamp-col flag).

The script assumes filenames sort into chronological order.  Two file-naming
conventions are supported (see ``--file-type`` flag):
  - ``standard``: column name is ``position_updated_at``
  - ``historical``: column name is ``position_timestamp`` (renamed on load)
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATES = [0.01, 0.025, 0.05, 0.075, 0.10]
SLOT_MINUTES = 10  # temporal resolution for vessel discovery

# Simulation time window — adjust to match your AIS dataset
FIXED_START_TIME = pd.Timestamp("2024-12-01 00:00:00", tz="UTC")
FIXED_END_TIME   = pd.Timestamp("2025-02-01 00:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# File sorting
# ---------------------------------------------------------------------------

def _sort_key(filename):
    """
    Sort CSV files so that standard AIS files precede historical archives,
    then order numerically by the trailing integer in the filename.
    """
    base = os.path.basename(filename)
    # Assign priority group: files with a "historical" prefix sort after
    # standard files so the primary stream is processed first.
    group = 1 if base.lower().startswith("historical") else 0
    tokens = base.rsplit("_", maxsplit=1)
    num = float("inf")
    if len(tokens) == 2:
        try:
            num = int(tokens[1].split(".")[0])
        except ValueError:
            pass
    return (group, num)


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def assign_time_slots(df, slot_minutes=SLOT_MINUTES):
    """Bin timestamps into fixed-width slots."""
    df["time_slot"] = df["position_updated_at"].dt.floor(f"{slot_minutes}T")
    return df


def _localize_to_utc(ts):
    """Ensure a Timestamp is UTC-aware."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def read_csv_file(file_path, timestamp_col="position_updated_at"):
    """
    Read one AIS CSV file, normalise column names, and localise timestamps.

    Parameters
    ----------
    file_path : str
    timestamp_col : str
        Source column name for the timestamp.  Will be renamed to
        ``position_updated_at`` if different.

    Returns
    -------
    pd.DataFrame  with columns: mmsi, latitude, longitude,
                                position_updated_at, time_slot
    """
    cols = ["latitude", "longitude", "mmsi", timestamp_col]
    df = pd.read_csv(file_path, usecols=cols, encoding="latin1")

    if timestamp_col != "position_updated_at":
        df = df.rename(columns={timestamp_col: "position_updated_at"})

    # Drop rows with missing coordinates
    df = df[df["latitude"].notnull() & df["longitude"].notnull()]

    # Parse and localise timestamps
    df["position_updated_at"] = pd.to_datetime(
        df["position_updated_at"], errors="coerce"
    )
    df = df[df["position_updated_at"].notnull()]
    df["position_updated_at"] = df["position_updated_at"].apply(_localize_to_utc)

    return assign_time_slots(df)


# ---------------------------------------------------------------------------
# Multi-rate vessel sampling
# ---------------------------------------------------------------------------

def process_window(df, seen_boats, sample_rates, cutoff=None):
    """
    Extract new vessels from *df* and sample them at each rate.

    Each time slot in *df* that falls within the simulation window is
    processed in chronological order.  New vessels (not in *seen_boats*) are
    shuffled and sliced at each rate; the slices are nested so that, e.g.,
    the 1% selection ⊆ the 2.5% selection ⊆ the 5% selection.

    Parameters
    ----------
    df : pd.DataFrame
        Combined rows to process (may span multiple files).
    seen_boats : set
        MMSIs already encountered in earlier time slots (mutated in place).
    sample_rates : list of float
        Sampling fractions in ascending order (e.g. [0.01, 0.025, 0.05]).
    cutoff : pd.Timestamp or None
        If given, rows in slots >= *cutoff* are returned as leftover (they
        may belong to the next file's first time slot).

    Returns
    -------
    new_boats_by_rate : dict  {rate: {mmsi: introduction_timeslot}}
    leftover_df : pd.DataFrame  (rows >= cutoff, or empty)
    terminate : bool  (True if simulation window boundary was crossed)
    """
    new_boats_by_rate = {r: {} for r in sample_rates}
    leftover_groups = []

    for ts, group in df.groupby("time_slot"):
        if ts < FIXED_START_TIME or ts >= FIXED_END_TIME:
            return new_boats_by_rate, pd.DataFrame(columns=df.columns), True

        if cutoff is not None and ts >= cutoff:
            leftover_groups.append(group)
            continue

        new_boats = set(group["mmsi"].unique()) - seen_boats
        seen_boats.update(group["mmsi"].unique())

        if new_boats:
            shuffled = list(new_boats)
            np.random.shuffle(shuffled)
            for rate in sorted(sample_rates):
                n_select = round(rate * len(shuffled))
                for mmsi in shuffled[:n_select]:
                    new_boats_by_rate[rate].setdefault(mmsi, ts)
            logging.info(
                "%s: %d new vessels; sampled %d at %.1f%%",
                ts,
                len(new_boats),
                round(max(sample_rates) * len(new_boats)),
                max(sample_rates) * 100,
            )

    leftover_df = (
        pd.concat(leftover_groups, ignore_index=True)
        if leftover_groups
        else pd.DataFrame(columns=df.columns)
    )
    return new_boats_by_rate, leftover_df, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample vessel MMSIs from raw AIS CSV files at multiple rates."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing AIS CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory for output MMSI list files (default: current directory).",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=DEFAULT_SAMPLE_RATES,
        help=(
            f"Sampling fractions to generate (default: {DEFAULT_SAMPLE_RATES}). "
            "Each value must be between 0 and 1."
        ),
    )
    parser.add_argument(
        "--timestamp-col",
        default="position_updated_at",
        help=(
            "CSV column name for the timestamp. "
            "Use 'position_timestamp' for historical-format files "
            "(default: position_updated_at)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    sample_rates = sorted(args.rates)
    output_files = {
        r: os.path.join(args.out_dir, f"{round(r * 100):.4g}pct_selected_boats.txt")
        for r in sample_rates
    }

    all_files = sorted(
        glob.glob(os.path.join(args.data_dir, "*.csv")), key=_sort_key
    )
    if not all_files:
        logging.error("No CSV files found in %s", args.data_dir)
        return
    logging.info("Found %d CSV files.", len(all_files))

    selected_boats = {r: {} for r in sample_rates}
    seen_boats = set()
    buffer_df = pd.DataFrame()

    for i, file_path in enumerate(all_files):
        logging.info("Processing: %s", file_path)
        df_current = read_csv_file(file_path, timestamp_col=args.timestamp_col)
        combined = (
            pd.concat([buffer_df, df_current], ignore_index=True)
            if not buffer_df.empty
            else df_current
        )

        if i < len(all_files) - 1:
            df_next = read_csv_file(all_files[i + 1], timestamp_col=args.timestamp_col)
            cutoff = df_next["time_slot"].min()
        else:
            cutoff = None

        new_by_rate, buffer_df, terminate = process_window(
            combined, seen_boats, sample_rates, cutoff
        )
        for r in sample_rates:
            for mmsi, ts in new_by_rate[r].items():
                selected_boats[r].setdefault(mmsi, ts)

        if terminate:
            break

    # Flush any remaining buffered rows
    if not buffer_df.empty:
        new_by_rate, _, _ = process_window(buffer_df, seen_boats, sample_rates, cutoff=None)
        for r in sample_rates:
            for mmsi, ts in new_by_rate[r].items():
                selected_boats[r].setdefault(mmsi, ts)

    # Write output files
    for r in sample_rates:
        with open(output_files[r], "w") as f:
            for mmsi in selected_boats[r]:
                f.write(f"{mmsi}\n")
        n = len(selected_boats[r])
        logging.info("Rate %.4g%%: %d vessels written to %s", r * 100, n, output_files[r])
        print(f"  {r * 100:.4g}%: {n:,} vessels → {output_files[r]}")


if __name__ == "__main__":
    main()
