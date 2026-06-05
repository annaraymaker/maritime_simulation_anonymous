#!/usr/bin/env python3
"""
glitch_filter.py
================
Remove GPS artifact (teleport) rows from filtered AIS position data.

AIS data frequently contains positional anomalies where a vessel's reported
position jumps unrealistically far between consecutive records—typical of
receiver glitches, satellite multipath, or transcription errors rather than
real movement.

This script detects and removes such rows by computing the implied speed
between consecutive points (using the Haversine formula) and flagging any
row where the approach or departure speed exceeds a configurable threshold.

Usage
-----
    python glitch_filter.py \\
        --input  1pct_filtered_data.pkl \\
        --output 1pct_cleaned_data.pkl \\
        --speed-threshold 200

Pipeline position
-----------------
    preprocess.py → data_loader.py → **glitch_filter.py** → simulate.py
"""

import argparse
import logging

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Per-vessel glitch removal
# ---------------------------------------------------------------------------

def remove_glitches(df: pd.DataFrame, speed_threshold: float = 100.0) -> pd.DataFrame:
    """
    Remove GPS artifact rows from a single vessel's position history.

    A row is flagged if the implied speed *to* the previous point **or**
    *from* the current point to the next exceeds *speed_threshold* km/h.
    This catches isolated outliers that briefly teleport and return.

    Parameters
    ----------
    df : pd.DataFrame
        Position records for one vessel (any columns beyond the four below
        are preserved unchanged).  Required columns:
        ``latitude``, ``longitude``, ``position_updated_at``.
    speed_threshold : float
        Maximum plausible vessel speed in km/h (default: 100).

    Returns
    -------
    pd.DataFrame  (glitch rows removed, helper columns dropped)
    """
    df = df.sort_values("position_updated_at").reset_index(drop=True)
    if len(df) < 2:
        return df

    df["prev_lat"]  = df["latitude"].shift(1)
    df["prev_lon"]  = df["longitude"].shift(1)
    df["prev_time"] = df["position_updated_at"].shift(1)
    df["next_lat"]  = df["latitude"].shift(-1)
    df["next_lon"]  = df["longitude"].shift(-1)
    df["next_time"] = df["position_updated_at"].shift(-1)

    # Time differences in hours
    df["dt_prev"] = (df["position_updated_at"] - df["prev_time"]).dt.total_seconds() / 3600.0
    df["dt_next"] = (df["next_time"] - df["position_updated_at"]).dt.total_seconds() / 3600.0

    # Distances in km
    df["dist_prev"] = haversine(df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"])
    df["dist_next"] = haversine(df["latitude"], df["longitude"], df["next_lat"], df["next_lon"])

    # Implied speeds
    df["speed_prev"] = df["dist_prev"] / df["dt_prev"]
    df["speed_next"] = df["dist_next"] / df["dt_next"]

    # Flag glitches (handle boundary rows separately)
    glitch = (df["speed_prev"] > speed_threshold) | (df["speed_next"] > speed_threshold)
    glitch.iloc[0]  = df["speed_next"].iloc[0]  > speed_threshold
    glitch.iloc[-1] = df["speed_prev"].iloc[-1] > speed_threshold

    n_removed = glitch.sum()
    if n_removed > 0:
        logging.debug("Removed %d glitch rows out of %d.", n_removed, len(df))

    helper_cols = [
        "prev_lat", "prev_lon", "prev_time",
        "next_lat", "next_lon", "next_time",
        "dt_prev", "dt_next", "dist_prev", "dist_next",
        "speed_prev", "speed_next",
    ]
    return df[~glitch].drop(columns=helper_cols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remove GPS teleport artifacts from AIS position data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input pickle file from data_loader.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output pickle file (GPS-cleaned).",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=200.0,
        help="Maximum plausible speed in km/h; rows implying higher speeds are removed (default: 200).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Loading %s ...", args.input)
    try:
        df = pd.read_pickle(args.input)
    except OSError as exc:
        logging.error("Cannot read input file: %s", exc)
        return

    logging.info("Loaded %d rows for %d vessels.", df.shape[0], df["mmsi"].nunique())
    df["position_updated_at"] = pd.to_datetime(df["position_updated_at"])

    cleaned_groups = []
    total_removed = 0

    for mmsi, group in df.groupby("mmsi"):
        before = group.shape[0]
        cleaned = remove_glitches(group, speed_threshold=args.speed_threshold)
        removed = before - cleaned.shape[0]
        if removed > 0:
            logging.info("MMSI %s: removed %d glitch rows.", mmsi, removed)
            total_removed += removed
        cleaned_groups.append(cleaned)

    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)
    logging.info(
        "Cleaning complete. %d rows retained, %d removed.",
        cleaned_df.shape[0],
        total_removed,
    )

    try:
        cleaned_df.to_pickle(args.output)
        logging.info("Cleaned data saved to %s", args.output)
        print(f"Cleaned data saved to {args.output} ({cleaned_df.shape[0]:,} rows)")
    except OSError as exc:
        logging.error("Error saving output: %s", exc)


if __name__ == "__main__":
    main()
