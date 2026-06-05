#!/usr/bin/env python3
"""
simulation_utils.py
===================
Core library for the maritime worm propagation simulation.

Provides:
  - Dask cluster initialization
  - Checkpoint save/load
  - AIS data loading
  - Haversine distance (Numba-JIT and plain variants)
  - Patient-zero selection
  - Contact-network construction (time-slotted SIR model)
  - Infection metrics output
  - Interactive map visualization
"""

import os
import json
import pickle
import random
import warnings
import multiprocessing

import folium
import networkx as nx
import numpy as np
import pandas as pd
import dask.dataframe as dd
from branca.colormap import LinearColormap
from dask.distributed import Client, LocalCluster
from joblib import Parallel, delayed
from numba import jit
from sklearn.neighbors import BallTree
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Radius used when checking whether a newly infected vessel has breached a
# target maritime chokepoint.
TARGET_PROXIMITY_RADIUS_KM = 200.0

# Sentinel value stored in the graph for the external adversary that seeded
# the initial infection.
ADVERSARY_LABEL = "ADVERSARY"


# ---------------------------------------------------------------------------
# Dask cluster
# ---------------------------------------------------------------------------

def init_dask_cluster():
    """
    Spin up a local Dask cluster with one single-threaded worker per CPU core.

    Returns
    -------
    client : dask.distributed.Client
    cluster : dask.distributed.LocalCluster

    Notes
    -----
    Call this inside ``if __name__ == '__main__':`` to avoid recursive
    process spawning on Windows/macOS.
    """
    num_cores = multiprocessing.cpu_count()
    cluster = LocalCluster(
        n_workers=num_cores,
        threads_per_worker=1,
        memory_limit="auto",
    )
    client = Client(cluster)
    print(f"Dask cluster ready: {client.dashboard_link}")
    return client, cluster


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(data, checkpoint_name, print_update=True):
    """
    Atomically persist *data* to ``checkpoints/<checkpoint_name>.pkl``.

    Writes to a temporary file first, then renames to avoid partial writes
    corrupting existing checkpoints on crash.
    """
    os.makedirs("checkpoints", exist_ok=True)
    cpn = os.path.join("checkpoints", checkpoint_name)
    tmp_path = f"{cpn}_tmp.pkl"
    final_path = f"{cpn}.pkl"

    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(tmp_path, final_path)

    if print_update:
        print(f"Checkpoint saved: {checkpoint_name}")


def load_checkpoint(checkpoint_name):
    """
    Load a checkpoint saved by :func:`save_checkpoint`.

    Returns the checkpoint dict, or ``None`` if no checkpoint exists.
    """
    cpn = os.path.join("checkpoints", f"{checkpoint_name}.pkl")
    if os.path.exists(cpn):
        with open(cpn, "rb") as f:
            data = pickle.load(f)
        print(f"Checkpoint loaded: {checkpoint_name}")
        return data
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_pickle(file_path, npartitions=None):
    """
    Load a preprocessed AIS pickle file into a Dask DataFrame.

    The pickle file must contain at minimum the columns:
    ``mmsi``, ``latitude``, ``longitude``, ``position_updated_at``.

    Parameters
    ----------
    file_path : str
        Path to the ``.pkl`` file.
    npartitions : int, optional
        Number of Dask partitions.  Defaults to 10× CPU count.

    Returns
    -------
    df_all : dask.dataframe.DataFrame   (persisted in distributed memory)
    simulation_start_time : pd.Timestamp (timezone-naive)
    """
    pdf = pd.read_pickle(file_path)
    pdf = (
        pdf
        .sort_values(["mmsi", "position_updated_at"])
        .drop_duplicates(subset=["mmsi", "position_updated_at"], keep="first")
    )
    print(f"Dataset loaded: {pdf.shape[0]:,} rows, {pdf.shape[1]} columns.")

    simulation_start_time = make_tz_naive(pdf["position_updated_at"].min())

    if npartitions is None:
        npartitions = 10 * multiprocessing.cpu_count()

    df_all = dd.from_pandas(pdf, npartitions=npartitions).persist()
    return df_all, simulation_start_time


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

@jit(nopython=True)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Haversine great-circle distance (km).  Numba-JIT compiled.

    Accepts scalars or NumPy arrays (element-wise).
    """
    R = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def haversine_distance_py(lat1, lon1, lat2, lon2):
    """Pure-Python fallback (same formula, no JIT).  Use for scalar calls."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

def make_tz_naive(dt):
    """Strip timezone info from a Timestamp or Series."""
    if isinstance(dt, pd.Timestamp):
        return dt.tz_localize(None) if dt.tz is not None else dt
    if isinstance(dt, pd.Series):
        return dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
    return dt


def make_tz_aware(dt, tz="UTC"):
    """Add UTC timezone to a naive Timestamp or Series."""
    if isinstance(dt, pd.Timestamp):
        return dt.tz_localize(tz) if dt.tz is None else dt
    if isinstance(dt, pd.Series):
        return dt.dt.tz_localize(tz) if dt.dt.tz is None else dt
    return dt


# ---------------------------------------------------------------------------
# Patient-zero selection
# ---------------------------------------------------------------------------

def select_initial_infected(
    checkpoint,
    target_lat,
    target_lon,
    threshold_km,
    fixed_start_time=pd.Timestamp("2024-12-01 00:00:00", tz="UTC"),
    fixed_end_time=pd.Timestamp("2025-02-01 00:00:00", tz="UTC"),
    slot_minutes=10,
):
    """
    Select patient-zero vessels near a seeding location.

    Scans time slots (in chronological order) until one or more vessels are
    found whose first recorded position within that slot is within
    *threshold_km* of (*target_lat*, *target_lon*).

    All position records for selected vessels that precede the discovered
    time slot are removed from ``checkpoint['df_all']`` so that the
    simulation timeline begins at the moment of initial infection.

    Parameters
    ----------
    checkpoint : dict
        Simulation checkpoint containing ``'df_all'`` (Dask DataFrame with
        a ``'time_slot'`` column).
    target_lat, target_lon : float
        Coordinates of the seeding port.
    threshold_km : float
        Maximum distance (km) from the seeding port.
    fixed_start_time, fixed_end_time : pd.Timestamp
        Search window boundaries.
    slot_minutes : int
        Width of each time slot in minutes.

    Returns
    -------
    set of MMSI strings
    """
    df = checkpoint["df_all"]
    timeslots = pd.date_range(
        start=fixed_start_time, end=fixed_end_time, freq=f"{slot_minutes}T"
    )

    selected_boats = None
    selected_timeslot = None

    for ts in timeslots:
        print(ts)
        df_timeslot = df[df["time_slot"] == ts]
        positions = df_timeslot.groupby("mmsi").agg(
            {"latitude": "first", "longitude": "first"}
        )
        positions_pd = positions.compute()
        if positions_pd.empty:
            continue

        distances = haversine_distance(
            positions_pd["latitude"].values,
            positions_pd["longitude"].values,
            target_lat,
            target_lon,
        )
        positions_pd["distance_km"] = distances
        near_boats = positions_pd[positions_pd["distance_km"] <= threshold_km]

        if not near_boats.empty:
            selected_boats = near_boats
            selected_timeslot = ts
            break

    if selected_boats is not None:
        for boat in selected_boats.index:
            checkpoint["df_all"] = checkpoint["df_all"][
                ~(
                    (checkpoint["df_all"]["mmsi"] == boat)
                    & (checkpoint["df_all"]["position_updated_at"] < selected_timeslot)
                )
            ]
        print(f"Patient zeros in slot {selected_timeslot}: {list(selected_boats.index)}")
        return set(selected_boats.index)

    print("No vessel found within threshold in any time slot.")
    return set()


# ---------------------------------------------------------------------------
# Optional: filter vulnerable population
# ---------------------------------------------------------------------------

def filter_vulnerable_population(df, checkpoint, vulnerable_percent=100):
    """
    Randomly down-sample the vessel population to *vulnerable_percent* percent.

    The patient-zero vessel is always retained.  Useful if you want to model
    a scenario where only a fraction of vessels run the vulnerable software.

    Parameters
    ----------
    df : Dask DataFrame
    checkpoint : dict
    vulnerable_percent : float  (0–100)

    Returns
    -------
    Dask DataFrame
    """
    patient_zeros = checkpoint["patient_zeros"]
    unique_boats = df["mmsi"].unique().compute().tolist()
    total_boats = len(unique_boats)
    num_vulnerable = max(int(total_boats * vulnerable_percent / 100), 1)

    for pz in patient_zeros:
        if pz in unique_boats:
            unique_boats.remove(pz)

    sample_count = max(num_vulnerable - len(patient_zeros), 0)
    vulnerable_boats = random.sample(unique_boats, sample_count)
    vulnerable_boats.extend(patient_zeros)

    df_vuln = df[df["mmsi"].isin(vulnerable_boats)]
    print(f"Vulnerable vessels: {len(vulnerable_boats):,} / {total_boats:,} (includes patient zero(s)).")
    return df_vuln


# ---------------------------------------------------------------------------
# Contact-network construction (parallel within time slot)
# ---------------------------------------------------------------------------

def _process_neighbors_chunk(
    start_idx,
    end_idx,
    indices_list,
    mmsi_arr,
    lat_arr,
    lon_arr,
    time_arr,
    G,
    infected_boats_set,
    remaining_targets,
    target_proximity_radius_km=TARGET_PROXIMITY_RADIUS_KM,
):
    """
    Process a contiguous slice of vessels for a single time slot.

    For each infected vessel in [start_idx, end_idx), propagates infection
    to all susceptible neighbors within the BallTree radius and checks
    whether newly infected vessels breach any target chokepoint.

    Returns
    -------
    new_infections : set
    edges : list of (u, v, {contact_time: ...})
    breach_events : list of (mmsi, target_name, timestamp)
    """
    new_infections = set()
    edges = []
    breach_events = []

    for i in range(start_idx, end_idx):
        boat_a = mmsi_arr[i]
        if boat_a not in infected_boats_set:
            continue

        time_a = pd.Timestamp(time_arr[i])
        for j in indices_list[i]:
            boat_b = mmsi_arr[j]
            if G.nodes[boat_b]["state"] == "I":
                continue

            contact_time = max(time_a, pd.Timestamp(time_arr[j]))
            edge = (boat_a, boat_b) if boat_a < boat_b else (boat_b, boat_a)
            edges.append((edge[0], edge[1], {"contact_time": contact_time}))

            G.nodes[boat_b]["state"] = "I"
            G.nodes[boat_b]["infector"] = boat_a
            G.nodes[boat_b]["infection_time"] = contact_time
            new_infections.add(boat_b)

            # Check proximity to monitored chokepoints
            boat_b_lat = lat_arr[j]
            boat_b_lon = lon_arr[j]
            for tgt_name, tgt_pos in remaining_targets.items():
                d = haversine_distance(
                    tgt_pos["lat"], tgt_pos["lon"], boat_b_lat, boat_b_lon
                )
                if d < target_proximity_radius_km:
                    breach_events.append((boat_b, tgt_name, time_arr[j]))

    return new_infections, edges, breach_events


def build_contact_network(checkpoint, out_name):
    """
    Run the full time-slotted SIR infection simulation.

    Processes each 10-minute time slot in ascending order.  Within each slot,
    a BallTree spatial query identifies all pairs of vessels within the RF
    infection range; infected vessels transmit to any susceptible neighbor.
    Neighbor processing is parallelised across CPU cores via Joblib.

    Stops early once all monitored chokepoints have been breached.

    Parameters
    ----------
    checkpoint : dict
        Must contain: ``G``, ``df_all``, ``patient_zeros``,
        ``infection_distance_rad``, ``infected_mmsis``,
        ``remaining_targets``, ``target_metrics``,
        ``simulation_start_time``, ``last_processed_timeslot``.
    out_name : str
        Checkpoint filename prefix.

    Returns
    -------
    networkx.Graph  (infection contact network)
    """
    G = checkpoint["G"]
    df = checkpoint["df_all"]

    # Initialise graph: all vessels susceptible
    unique_boats = df["mmsi"].unique().compute().tolist()
    G.add_nodes_from(
        (boat, {"state": "S", "infector": -1, "infection_time": None})
        for boat in unique_boats
    )

    # Seed patient zeros
    for pz in checkpoint["patient_zeros"]:
        G.nodes[pz]["state"] = "I"
        G.nodes[pz]["infector"] = ADVERSARY_LABEL
        G.nodes[pz]["infection_time"] = checkpoint["simulation_start_time"]
    checkpoint["infected_mmsis"].update(checkpoint["patient_zeros"])
    print(f"Patient zeros: {checkpoint['patient_zeros']}")

    time_slots = df["time_slot"].drop_duplicates().compute().sort_values().tolist()

    for current_time in tqdm(time_slots, desc="Processing time slots"):
        if checkpoint["last_processed_timeslot"] and current_time < checkpoint["last_processed_timeslot"]:
            continue

        slot_data = df[df["time_slot"] == current_time].compute()
        if slot_data.empty:
            continue

        # Build BallTree over vessel positions in this slot
        coords = np.deg2rad(
            np.column_stack((slot_data["latitude"], slot_data["longitude"]))
        )
        tree = BallTree(coords, metric="haversine")
        indices_list = tree.query_radius(coords, r=checkpoint["infection_distance_rad"])

        mmsi_arr = slot_data["mmsi"].values
        lat_arr = slot_data["latitude"].values
        lon_arr = slot_data["longitude"].values
        time_arr = pd.to_datetime(slot_data["position_updated_at"]).values

        infected_set = set(checkpoint["infected_mmsis"])
        remaining = checkpoint["remaining_targets"]

        # Parallel chunk processing
        chunk_size = 5000
        n = len(indices_list)
        tasks = [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]

        results = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(_process_neighbors_chunk)(
                s, e, indices_list,
                mmsi_arr, lat_arr, lon_arr, time_arr,
                G, infected_set, remaining,
            )
            for s, e in tasks
        )

        new_infections, edges_to_add, breaching_boats = set(), [], []
        for inf_set, edge_list, breach_list in results:
            new_infections |= inf_set
            edges_to_add.extend(edge_list)
            breaching_boats.extend(breach_list)

        G.add_edges_from(edges_to_add)
        checkpoint["infected_mmsis"].update(new_infections)

        # Resolve breaches and trace critical paths
        if not checkpoint["remaining_targets"]:
            print("All targets breached — stopping simulation.")
            break

        for boat_b, target_name, b_time in breaching_boats:
            if target_name not in checkpoint["remaining_targets"]:
                continue  # Already recorded

            print(f"\n{target_name} breached!")
            print(f"  At {b_time}, vessel {boat_b} is within {TARGET_PROXIMITY_RADIUS_KM} km.")
            del checkpoint["remaining_targets"][target_name]

            # Trace infection chain back to patient zero
            path = [boat_b]
            infector = boat_b
            while True:
                nxt = G.nodes[infector]["infector"]
                if nxt == ADVERSARY_LABEL:
                    path.append(ADVERSARY_LABEL)
                    break
                if nxt == -1 or nxt not in G:
                    print(f"Warning: broken path at vessel {infector} (infector={nxt}).")
                    break
                path.append(nxt)
                infector = nxt

            checkpoint["target_metrics"][target_name]["critical_paths"] = path
            checkpoint["target_metrics"][target_name]["breach_timestamp"] = b_time
            checkpoint["target_metrics"][target_name]["collateral_damage"] = len(
                checkpoint["infected_mmsis"]
            )
            print(f"  Remaining targets: {list(checkpoint['remaining_targets'])}")

        checkpoint["last_processed_timeslot"] = current_time
        save_checkpoint(checkpoint, out_name, print_update=False)

    return G


# ---------------------------------------------------------------------------
# Metrics output
# ---------------------------------------------------------------------------

def save_metrics(checkpoint, out_name):
    """
    Write per-target breach metrics to a JSON file in ``results/``.

    Metrics include:
    - ``time_to_infection_hours`` — elapsed hours from simulation start
    - ``degrees_of_separation_hops`` — infection chain length
    - ``collateral_damage`` — total infected vessels at time of breach
    """
    os.makedirs("results", exist_ok=True)
    output_data = {}

    for target, metrics in checkpoint["target_metrics"].items():
        print(target)
        ts = metrics["breach_timestamp"]
        if ts is not None:
            path = metrics["critical_paths"]
            elapsed_hours = (
                (ts - checkpoint["simulation_start_time"]).total_seconds() / 3600
            )
            cd = metrics["collateral_damage"]
            output_data[target] = {
                "time_to_infection_hours": round(elapsed_hours, 2),
                "degrees_of_separation_hops": len(path),
                "collateral_damage": cd,
            }
            print(f"  Time-to-infection: {elapsed_hours:.2f} h")
            print(f"  Degrees of separation: {len(path)} hops")
            print(f"  Collateral damage: {cd} vessels")
        else:
            output_data[target] = {"status": "NOT BREACHED"}
            print("  NOT BREACHED")

    out_file = os.path.join(
        "results", f"{out_name}_{len(checkpoint['infected_mmsis'])}b.json"
    )
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Metrics written to {out_file}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_infected_boats(
    checkpoint,
    out_name,
    target_locations,
    target_proximity_radius_km=TARGET_PROXIMITY_RADIUS_KM,
    max_segment_distance_km=300.0,
):
    """
    Generate an interactive Folium map showing infected vessel trails.

    Vessel trails are drawn from the moment of infection onward.  Segments
    longer than *max_segment_distance_km* (e.g. caused by data gaps or
    date-line crossings) are broken.  Target chokepoints appear as circles:
    green if breached, blue otherwise.

    Output is saved as an HTML file in ``results/``.
    """
    os.makedirs("results", exist_ok=True)

    df = checkpoint["df_all"]
    G = checkpoint["G"]
    infected_boats = checkpoint["infected_mmsis"]

    # Collect critical-path vessels from all target metrics
    critical_vessels = set()
    for metrics in checkpoint["target_metrics"].values():
        for mmsi in metrics.get("critical_paths", []):
            if mmsi != ADVERSARY_LABEL:
                critical_vessels.add(mmsi)

    mean_lat = df["latitude"].mean().compute()
    mean_lon = df["longitude"].mean().compute()
    base_map = folium.Map(
        location=[mean_lat, mean_lon], zoom_start=6, tiles="Cartodb Positron"
    )

    df_infected = df[df["mmsi"].isin(infected_boats)]
    pdf_infected = (
        df_infected.compute().sort_values(["mmsi", "position_updated_at"])
    )
    pdf_infected["position_updated_at"] = make_tz_naive(
        pdf_infected["position_updated_at"]
    )

    # Draw non-critical trails first, then critical ones on top
    deferred_critical = []

    for mmsi, group in pdf_infected.groupby("mmsi"):
        infection_time = make_tz_naive(G.nodes[mmsi]["infection_time"])
        infector = G.nodes[mmsi]["infector"]
        if infection_time is None:
            continue

        group = group[group["position_updated_at"] >= infection_time]
        if group.shape[0] < 2:
            continue

        is_critical = mmsi in critical_vessels
        color = "#cc1414" if is_critical else "#000000"

        # Split trail at large spatial gaps or date-line crossings
        points = list(zip(group["latitude"], group["longitude"]))
        segments = []
        current_seg = [points[0]]
        for pt in points[1:]:
            last = current_seg[-1]
            if abs(pt[1] - last[1]) > 180:
                if len(current_seg) >= 2:
                    segments.append(current_seg)
                current_seg = [pt]
                continue
            dist = haversine_distance_py(last[0], last[1], pt[0], pt[1])
            if dist <= max_segment_distance_km:
                current_seg.append(pt)
            else:
                if len(current_seg) >= 2:
                    segments.append(current_seg)
                current_seg = [pt]
        if len(current_seg) >= 2:
            segments.append(current_seg)

        popup_text = f"MMSI: {mmsi} | Infected by: {infector} at {infection_time}"
        for seg in segments:
            line = folium.PolyLine(
                locations=seg, color=color, weight=3, opacity=0.8, popup=popup_text
            )
            if is_critical:
                deferred_critical.append(line)
            else:
                line.add_to(base_map)

    # Add critical trails on top
    for line in deferred_critical:
        line.add_to(base_map)

    # Draw chokepoint circles
    for place, coords in target_locations.items():
        breached = checkpoint["target_metrics"][place]["breach_timestamp"] is not None
        folium.Circle(
            location=[coords["lat"], coords["lon"]],
            radius=target_proximity_radius_km * 1000,
            color="green" if breached else "blue",
            fill=True,
            fill_opacity=0.4,
            popup=place,
        ).add_to(base_map)

    out_file = os.path.join(
        "results", f"{out_name}_{len(checkpoint['infected_mmsis'])}b.html"
    )
    base_map.save(out_file)
    print(f"Map saved to {out_file}")
