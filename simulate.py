#!/usr/bin/env python3
"""
simulate.py
===========
Main entry point for the maritime worm propagation simulation.

The simulation models a proximity-based SIR infection spreading through real
AIS vessel position data.  An external adversary seeds infection in vessels
near a chosen port; the malware then propagates ship-to-ship whenever an
infected vessel comes within RF range (default: 100 nmi / 185 km) of a
susceptible vessel.  Simulation terminates when all monitored maritime
chokepoints have been breached or all time slots are exhausted.

Usage
-----
    python simulate.py \\
        --data     1pct_cleaned_data.pkl \\
        --port     Jebel_Ali \\
        --distance 185.2

Outputs (written to results/):
    <run_name>_<N>b.json   per-target breach metrics
    <run_name>_<N>b.html   interactive vessel-trail map
"""

import argparse
import warnings
import networkx as nx

warnings.filterwarnings("ignore", category=FutureWarning)

from simulation_utils import (
    init_dask_cluster,
    load_data_pickle,
    save_checkpoint,
    load_checkpoint,
    select_initial_infected,
    build_contact_network,
    save_metrics,
    visualize_infected_boats,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Monitored maritime chokepoints
TARGET_LOCATIONS = {
    "Panama_Canal":      {"lat":  9.12,  "lon": -79.73},
    "Cape_of_Good_Hope": {"lat": -34.35, "lon":  18.42},
    "Gibraltar":         {"lat":  35.96, "lon":  -5.55},
    "Bosporus":          {"lat":  41.03, "lon":  29.00},
    "Suez_Canal":        {"lat":  30.56, "lon":  32.34},
    "Strait_of_Hormuz":  {"lat":  26.65, "lon":  56.52},
    "Bab_el_Mandeb":     {"lat":  12.75, "lon":  43.31},
    "Strait_of_Malacca": {"lat":   2.27, "lon": 101.76},
}

# Candidate seeding ports: add entries as needed
SEEDING_PORTS = {
    "Jebel_Ali":         (25.000,  55.080),   # UAE — busiest port in Middle East
    "Los_Angeles":       (33.741, -118.272),   # USA — busiest in the Americas
    "Rotterdam":         (51.924,   4.478),    # Netherlands — busiest in Europe
    "Shanghai":          (31.230,  121.474),   # China — largest by throughput
    "Singapore":         ( 1.352,  103.820),   # Singapore — key Southeast Asia hub
    "Mundra":            (22.970,   69.667),   # India — busiest in South Asia
    "Port_of_Durban":    (-29.859,  31.022),   # South Africa — busiest in Africa
}

# RF proximity range (100 nautical miles ≈ 185.2 km)
DEFAULT_INFECTION_DISTANCE_KM = 185.2

# Time-slot width for position binning
SLOT_MINUTES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_time_slots(df, slot_minutes=SLOT_MINUTES):
    """Floor timestamps to fixed-width slots for efficient grouping."""
    df["time_slot"] = df["position_updated_at"].dt.floor(f"{slot_minutes}T")
    return df


def _default_checkpoint(target_locations, infection_distance_km):
    return {
        "current_step": 0,
        "last_processed_timeslot": None,
        "patient_zeros": None,
        "df_all": None,
        "infection_distance_rad": infection_distance_km / 6371.0,
        "G": nx.Graph(),
        "degrees": None,
        "infection_times": None,
        "remaining_targets": target_locations.copy(),
        "infected_mmsis": set(),
        "target_metrics": {
            k: {"critical_paths": [], "breach_timestamp": None, "collateral_damage": None}
            for k in target_locations
        },
        "simulation_start_time": None,
    }


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation(data_path: str, seed_port: str, infection_distance_km: float):
    """
    Execute the full propagation simulation for one (data, port, distance) scenario.

    Parameters
    ----------
    data_path : str
        Path to preprocessed, GPS-cleaned AIS pickle file.
    seed_port : str
        Key in :data:`SEEDING_PORTS` specifying the infection origin.
    infection_distance_km : float
        Maximum vessel-to-vessel RF range in kilometres.
    """
    if seed_port not in SEEDING_PORTS:
        raise ValueError(
            f"Unknown seed port '{seed_port}'. "
            f"Available: {list(SEEDING_PORTS)}"
        )

    start_lat, start_lon = SEEDING_PORTS[seed_port]
    out_name = f"{seed_port}_{round(infection_distance_km)}km_{data_path.replace('/', '')}"

    # Initialise Dask (inside __main__ guard in main() below)
    print(f"\nScenario: port={seed_port}, range={infection_distance_km} km, data={data_path}")

    checkpoint = load_checkpoint(out_name) or _default_checkpoint(
        TARGET_LOCATIONS, infection_distance_km
    )

    # ------------------------------------------------------------------
    # Step 1: Load AIS data
    # ------------------------------------------------------------------
    if checkpoint["current_step"] < 1:
        print("\n[Step 1] Loading AIS dataset...")
        df_all, start_time = load_data_pickle(data_path)
        df_all = assign_time_slots(df_all)
        checkpoint["df_all"] = df_all
        checkpoint["simulation_start_time"] = start_time
        checkpoint["current_step"] = 1
        save_checkpoint(checkpoint, out_name)
    else:
        print("[Step 1] Already complete.")

    n_vessels = len(checkpoint["df_all"]["mmsi"].unique().compute())
    print(f"  Vessels in dataset: {n_vessels:,}")

    # ------------------------------------------------------------------
    # Step 2: Select patient zeros near seeding port
    # ------------------------------------------------------------------
    if checkpoint["current_step"] < 2:
        print("\n[Step 2] Selecting patient zeros...")
        pzeros = select_initial_infected(
            checkpoint, start_lat, start_lon, infection_distance_km
        )
        checkpoint["patient_zeros"] = pzeros
        checkpoint["infected_mmsis"].update(pzeros)
        checkpoint["current_step"] = 2
        save_checkpoint(checkpoint, out_name)
    else:
        print(f"[Step 2] Already complete. Patient zeros: {checkpoint['patient_zeros']}")

    # ------------------------------------------------------------------
    # Step 3: Run contact-network infection simulation
    # ------------------------------------------------------------------
    if checkpoint["current_step"] < 4:
        print("\n[Step 3] Running infection simulation...")
        checkpoint["G"] = build_contact_network(checkpoint, out_name)
        checkpoint["current_step"] = 4
        save_checkpoint(checkpoint, out_name)
    else:
        print("[Step 3] Already complete.")

    # ------------------------------------------------------------------
    # Step 4: Save per-target metrics
    # ------------------------------------------------------------------
    print("\n[Step 4] Saving metrics...")
    save_metrics(checkpoint, out_name)

    # ------------------------------------------------------------------
    # Step 5: Generate interactive map
    # ------------------------------------------------------------------
    if checkpoint["current_step"] < 6:
        print("\n[Step 5] Generating visualization...")
        visualize_infected_boats(checkpoint, out_name, TARGET_LOCATIONS)
        checkpoint["current_step"] = 6
        save_checkpoint(checkpoint, out_name)
    else:
        print("[Step 5] Already complete (simulation fully finished).")

    print(f"\nTotal infected vessels: {len(checkpoint['infected_mmsis']):,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate maritime malware propagation over AIS position data."
    )
    parser.add_argument(
        "--data",
        default="cleaned_data.pkl",
        help="Path to preprocessed, GPS-cleaned AIS pickle file (default: cleaned_data.pkl)",
    )
    parser.add_argument(
        "--port",
        default="Jebel_Ali",
        help=f"Seeding port name. Options: {list(SEEDING_PORTS)} (default: Jebel_Ali)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=DEFAULT_INFECTION_DISTANCE_KM,
        help=f"RF infection range in km (default: {DEFAULT_INFECTION_DISTANCE_KM})",
    )
    args = parser.parse_args()

    from simulation_utils import init_dask_cluster
    _client, _cluster = init_dask_cluster()

    run_simulation(
        data_path=args.data,
        seed_port=args.port,
        infection_distance_km=args.distance,
    )


if __name__ == "__main__":
    main()
