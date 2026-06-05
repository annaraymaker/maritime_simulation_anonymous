#!/usr/bin/env python3
"""
visualize_map.py
================
Render a high-quality interactive map from a simulation checkpoint.

Vessel trails are drawn from the moment of infection onward.  Trails that
are part of the critical infection path to any chokepoint are drawn in red
on top of the grey general-population trails so they remain visible.
Target chokepoints appear as filled circles: green (breached) or blue (not
reached).

Usage
-----
    python visualize_map.py \\
        --checkpoint Jebel_Ali_185km_1pct_cleaned_data.pkl \\
        --output     pretty_map.html
"""

import argparse
import math
import warnings

import folium
import pandas as pd

from simulation_utils import load_checkpoint, make_tz_naive, ADVERSARY_LABEL

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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

TARGET_RADIUS_KM  = 200.0   # circle radius on the map
MAX_SEGMENT_KM    = 300.0   # max allowed gap before breaking a trail segment


# ---------------------------------------------------------------------------
# Distance helper (pure Python — no Numba needed here)
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_map(checkpoint_name: str, output_path: str = "pretty_map.html"):
    """
    Load *checkpoint_name* and render an HTML map to *output_path*.

    Parameters
    ----------
    checkpoint_name : str
        Checkpoint key (file at ``checkpoints/<checkpoint_name>.pkl``).
    output_path : str
        Output HTML file path.
    """
    checkpoint = load_checkpoint(checkpoint_name)
    if checkpoint is None:
        print(f"Checkpoint '{checkpoint_name}' not found.")
        return

    df = checkpoint["df_all"]
    G  = checkpoint["G"]
    infected_mmsis = checkpoint["infected_mmsis"]

    # Derive critical-path vessels from all target metrics
    critical_vessels = set()
    for metrics in checkpoint["target_metrics"].values():
        for mmsi in metrics.get("critical_paths", []):
            if mmsi != ADVERSARY_LABEL:
                critical_vessels.add(mmsi)

    # Map centred on mean position of infected vessels
    mean_lat = df["latitude"].mean().compute()
    mean_lon = df["longitude"].mean().compute()
    base_map = folium.Map(
        location=[mean_lat, mean_lon], zoom_start=6, tiles="Cartodb Positron"
    )

    # Compute infected trails
    df_infected = df[df["mmsi"].isin(infected_mmsis)]
    pdf = df_infected.compute().sort_values(["mmsi", "position_updated_at"])
    pdf["position_updated_at"] = pdf["position_updated_at"].apply(
        lambda x: x.tz_localize(None) if hasattr(x, "tzinfo") and x.tzinfo else x
    )

    deferred_red = []  # drawn last so critical trails appear on top

    for mmsi, group in pdf.groupby("mmsi"):
        infection_time = G.nodes[mmsi].get("infection_time")
        if infection_time is None:
            continue
        infection_time = make_tz_naive(infection_time)

        group = group[group["position_updated_at"] >= infection_time]
        if group.shape[0] < 2:
            continue

        is_critical = mmsi in critical_vessels
        color = "#cc1414" if is_critical else "#000000"

        # Segment the trail (break at date-line crossings and large gaps)
        points = list(zip(group["latitude"], group["longitude"]))
        segments, current_seg = [], [points[0]]

        for pt in points[1:]:
            last = current_seg[-1]
            if abs(pt[1] - last[1]) > 180:          # date-line crossing
                if len(current_seg) >= 2:
                    segments.append(current_seg)
                current_seg = [pt]
                continue
            if _haversine(last[0], last[1], pt[0], pt[1]) <= MAX_SEGMENT_KM:
                current_seg.append(pt)
            else:
                if len(current_seg) >= 2:
                    segments.append(current_seg)
                current_seg = [pt]
        if len(current_seg) >= 2:
            segments.append(current_seg)

        infector   = G.nodes[mmsi].get("infector", "Unknown")
        popup_text = f"MMSI: {mmsi} | Infected by: {infector} at {infection_time}"

        for seg in segments:
            line = folium.PolyLine(
                locations=seg, color=color, weight=3,
                opacity=0.8 if not is_critical else 1.0,
                popup=popup_text,
            )
            if is_critical:
                deferred_red.append(line)
            else:
                line.add_to(base_map)

    for line in deferred_red:
        line.add_to(base_map)

    # Chokepoint circles
    target_metrics = checkpoint.get("target_metrics", {})
    for place, coords in TARGET_LOCATIONS.items():
        breached = target_metrics.get(place, {}).get("breach_timestamp") is not None
        folium.Circle(
            location=[coords["lat"], coords["lon"]],
            radius=TARGET_RADIUS_KM * 1000,
            color="green" if breached else "blue",
            fill=True,
            fill_opacity=0.4,
            popup=place,
        ).add_to(base_map)

    base_map.save(output_path)
    print(f"Map saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render an interactive vessel-trail map from a simulation checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint name (file will be read from checkpoints/<name>.pkl).",
    )
    parser.add_argument(
        "--output",
        default="pretty_map.html",
        help="Output HTML path (default: pretty_map.html).",
    )
    args = parser.parse_args()
    build_map(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
