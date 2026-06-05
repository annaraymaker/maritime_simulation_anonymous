#!/usr/bin/env python3
"""
data_loader.py
==============
Filter raw AIS CSV files to a pre-selected set of vessel MMSIs and
serialise the result as a Pandas pickle file.

This is the second step in the preprocessing pipeline:

    1. ``preprocess.py``  → produces ``<rate>pct_selected_boats.txt``
    2. **data_loader.py** → produces ``<rate>pct_filtered_data.pkl``
    3. ``glitch_filter.py`` → produces ``<rate>pct_cleaned_data.pkl``
    4. ``simulate.py``    → runs the infection simulation

Usage
-----
    python data_loader.py \\
        --mmsi-file 1pct_selected_boats.txt \\
        --data-dir  /path/to/ais_csvs \\
        --output    1pct_filtered_data.pkl
"""

import argparse
import glob
import logging
import os

import pandas as pd


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def process_file(file_path, mmsi_set, chunksize=10_000):
    """
    Stream a CSV file in chunks, retaining only rows whose MMSI is in *mmsi_set*.

    Parameters
    ----------
    file_path : str
    mmsi_set : set of str
    chunksize : int

    Returns
    -------
    pd.DataFrame  (may be empty if no matching rows)
    """
    # Detect column-name variant: some historical AIS archives use
    # ``position_timestamp`` instead of ``position_updated_at``.
    basename = os.path.basename(file_path)
    if "historical" in basename.lower():
        usecols = ["latitude", "longitude", "mmsi", "position_timestamp"]
        rename = {"position_timestamp": "position_updated_at"}
    else:
        usecols = ["latitude", "longitude", "mmsi", "position_updated_at"]
        rename = {}

    filtered_chunks = []
    try:
        for chunk in pd.read_csv(
            file_path, usecols=usecols, encoding="latin1", chunksize=chunksize
        ):
            if rename:
                chunk = chunk.rename(columns=rename)
            chunk["mmsi"] = chunk["mmsi"].astype(str)
            hit = chunk[chunk["mmsi"].isin(mmsi_set)]
            if not hit.empty:
                filtered_chunks.append(hit)
    except Exception as exc:
        logging.error("Error reading %s: %s", file_path, exc)

    return pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Filter AIS CSVs to a pre-selected MMSI list and save as pickle."
    )
    parser.add_argument(
        "--mmsi-file",
        required=True,
        help="Text file with one MMSI per line (output of preprocess.py).",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing raw AIS CSV files.",
    )
    parser.add_argument(
        "--output",
        default="filtered_data.pkl",
        help="Output pickle filename (default: filtered_data.pkl).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="CSV streaming chunk size in rows (default: 10000).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load MMSI allowlist
    logging.info("Loading MMSIs from %s ...", args.mmsi_file)
    try:
        with open(args.mmsi_file) as f:
            mmsi_set = {line.strip() for line in f if line.strip()}
        logging.info("Loaded %d MMSIs.", len(mmsi_set))
    except OSError as exc:
        logging.error("Cannot read MMSI file: %s", exc)
        return

    # Discover CSV files
    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    if not csv_files:
        logging.error("No CSV files found in %s", args.data_dir)
        return
    logging.info("Found %d CSV files.", len(csv_files))

    # Filter and combine
    filtered_dfs = []
    for path in csv_files:
        logging.info("Processing: %s", path)
        df = process_file(path, mmsi_set, chunksize=args.chunksize)
        if not df.empty:
            filtered_dfs.append(df)
            logging.info("  → %d matching rows.", df.shape[0])
        else:
            logging.info("  → no matching rows.")

    if not filtered_dfs:
        logging.warning("No matching data found. Nothing written.")
        return

    final_df = pd.concat(filtered_dfs, ignore_index=True)
    logging.info("Final dataset: %d rows.", final_df.shape[0])

    try:
        final_df.to_pickle(args.output)
        logging.info("Saved to %s", args.output)
        print(f"Filtered data saved to {args.output} ({final_df.shape[0]:,} rows)")
    except OSError as exc:
        logging.error("Error saving pickle: %s", exc)


if __name__ == "__main__":
    main()
