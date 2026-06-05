# Maritime Worm Propagation Simulation

This repository contains the simulation code used to evaluate the real-world
impact of a ship-to-ship self-propagating maritime malware ("worm"), as
described in the accompanying paper.

The simulation models how a proximity-based malware could propagate across
the global fleet by exploiting the mandatory RF broadcast ingestion that all
vessels perform for situational awareness.  It uses real-world AIS (Automatic
Identification System) position data to drive a time-slotted SIR (Susceptible
→ Infected) epidemic model, and reports how quickly an initially small
infection reaches each of eight major maritime chokepoints.

---

## Overview

Infection spreads ship-to-ship whenever an infected vessel comes within the
configurable RF range (default: **100 nmi / 185 km**, the nominal range of
Class A AIS transponders) of a susceptible vessel during the same 10-minute
time slot.  The simulation begins with a set of *patient-zero* vessels found
near a seeding port and terminates once all monitored chokepoints have been
breached or all time slots are exhausted.

**Monitored chokepoints**

| Chokepoint | Coordinates |
|---|---|
| Panama Canal | 9.12°N, 79.73°W |
| Cape of Good Hope | 34.35°S, 18.42°E |
| Strait of Gibraltar | 35.96°N, 5.55°W |
| Bosphorus | 41.03°N, 29.00°E |
| Suez Canal | 30.56°N, 32.34°E |
| Strait of Hormuz | 26.65°N, 56.52°E |
| Bab-el-Mandeb | 12.75°N, 43.31°E |
| Strait of Malacca | 2.27°N, 101.76°E |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
│
├── preprocess.py          # Stage 1 — sample vessel MMSIs at multiple infection rates
├── data_loader.py         # Stage 2 — filter raw CSV data to sampled MMSI list
├── glitch_filter.py       # Stage 3 — remove GPS artifact / teleport rows
├── simulate.py            # Stage 4 — run the SIR infection simulation
│
├── epidemiology.py        # Analytical multi-seed spread-time model
├── plot_results.py        # Plot time-to-infection vs. infection rate
├── visualize_map.py       # Render interactive vessel-trail map from checkpoint
│
├── simulation_utils.py    # Core library (imported by simulate.py / visualize_map.py)
│
├── checkpoints/           # Auto-created; simulation checkpoints written here
└── results/
    └── infection_rate_results.json   # Consolidated simulation output (all rates)
```

---

## Testbed Network Topology

The file `testbed_network.pdf` contains the full network architecture 
diagram of the physical maritime testbed used in this research. It 
illustrates how the testbed components are interconnected across the 
three protocol layers studied in the paper:

- **Serial layer** (NMEA 0183 / IEC 61162-1): AIS transponder, GPS 
  receiver, and other sensor outputs
- **CAN bus layer** (NMEA 2000 / IEC 61162-3): autopilot controller, 
  engine interface, and navigation instruments
- **Ethernet layer** (IEC 61162-450): MFD, SignalK server, Digital 
  Switching Unit, and IP-based gateways

The diagram is provided as a reference for researchers seeking to 
reproduce or extend the testbed, and to illustrate the representative 
coverage of our setup relative to real-world vessel deployments.



## Pipeline

```
Raw AIS CSV files
        │
        ▼
  preprocess.py          Scan all CSV files chronologically; sample N% of
  --data-dir <dir>  ──►  unique vessels per time slot. Produces one
  --rates 0.01 ...       MMSI list file per sample rate.
        │
        ▼  <rate>pct_selected_boats.txt
        │
  data_loader.py         Re-scan CSVs; keep only rows for sampled MMSIs.
  --mmsi-file ...   ──►  Produces a compact pickle for the next stage.
  --data-dir <dir>
        │
        ▼  <rate>pct_filtered_data.pkl
        │
  glitch_filter.py       Remove GPS teleport artifacts (rows implying
  --input ...       ──►  unrealistic vessel speeds). Produces a clean
  --output ...           pickle ready for simulation.
        │
        ▼  <rate>pct_cleaned_data.pkl
        │
  simulate.py            Run the time-slotted SIR simulation starting
  --data ...        ──►  from vessels near the seeding port. Writes
  --port ...             per-target JSON metrics and an HTML map to
  --distance ...         results/.
        │
        ├── results/<run>_<N>b.json    breach times, hop counts, collateral damage
        └── results/<run>_<N>b.html   interactive vessel-trail map
```

---

## Installation

```bash
python -m venv venv && source venv/bin/activate   # recommended
pip install -r requirements.txt
```

Python 3.10+ is required.  The simulation uses Dask Distributed for
parallelism; all workers run locally — no cluster setup is needed.

---

## Usage

### Full pipeline (one infection rate)

```bash
# 1. Sample 1% of vessels from raw AIS data
python preprocess.py \
    --data-dir /path/to/ais_csvs \
    --out-dir  . \
    --rates    0.01

# 2. Build filtered pickle
python data_loader.py \
    --mmsi-file 1pct_selected_boats.txt \
    --data-dir  /path/to/ais_csvs \
    --output    1pct_filtered_data.pkl

# 3. Remove GPS artifacts
python glitch_filter.py \
    --input  1pct_filtered_data.pkl \
    --output 1pct_cleaned_data.pkl

# 4. Run simulation (seeds infection near Jebel Ali)
python simulate.py \
    --data     1pct_cleaned_data.pkl \
    --port     Jebel_Ali \
    --distance 185.2
```

### Run all infection rates at once

```bash
for rate in 0.01 0.025 0.05 0.075 0.10; do
    label=$(python -c "print(str($rate*100).rstrip('0').rstrip('.'))")
    python preprocess.py --data-dir /path/to/ais_csvs --rates $rate
    python data_loader.py \
        --mmsi-file ${label}pct_selected_boats.txt \
        --data-dir  /path/to/ais_csvs \
        --output    ${label}pct_filtered_data.pkl
    python glitch_filter.py \
        --input  ${label}pct_filtered_data.pkl \
        --output ${label}pct_cleaned_data.pkl
    python simulate.py \
        --data  ${label}pct_cleaned_data.pkl \
        --port  Jebel_Ali
done
```

### Visualize results

```bash
# Plot time-to-infection vs. infection rate
python plot_results.py \
    --results results/infection_rate_results.json \
    --output  infection_rates.pdf

# Render vessel-trail map from a saved checkpoint
python visualize_map.py \
    --checkpoint Jebel_Ali_185km_1pct_cleaned_data.pkl \
    --output     pretty_map.html

# Multi-seed analytical model
python epidemiology.py
```

---

## AIS Data Format

The scripts expect CSV files with at minimum the following columns:

| Column | Type | Notes |
|---|---|---|
| `mmsi` | string / int | Vessel identifier |
| `latitude` | float | Decimal degrees |
| `longitude` | float | Decimal degrees |
| `position_updated_at` | datetime string | UTC preferred |

A historical-format variant using `position_timestamp` instead of
`position_updated_at` is also supported via the `--timestamp-col` flag in
`preprocess.py` and is detected automatically by filename prefix in
`data_loader.py`.

---

## Key Parameters

| Script | Parameter | Default | Description |
|---|---|---|---|
| `preprocess.py` | `--rates` | `0.01 0.025 0.05 0.075 0.10` | Vessel sampling fractions |
| `preprocess.py` | `--seed` | `42` | RNG seed for reproducibility |
| `glitch_filter.py` | `--speed-threshold` | `200` km/h | Max plausible vessel speed |
| `simulate.py` | `--port` | `Jebel_Ali` | Seeding port |
| `simulate.py` | `--distance` | `185.2` km | RF proximity range (100 nmi) |
| `simulation_utils.py` | `TARGET_PROXIMITY_RADIUS_KM` | `200` km | Chokepoint breach radius |
| `simulate.py` | `SLOT_MINUTES` | `10` | Time-slot width |

---

## Simulation Output

Each completed simulation run writes two files to `results/`:

**JSON metrics** (e.g., `Jebel_Ali_185km_1pct_cleaned_data.pkl_1024b.json`):
```json
{
    "Suez_Canal": {
        "time_to_infection_hours": 351.13,
        "degrees_of_separation_hops": 7,
        "collateral_damage": 43
    },
    "Panama_Canal": {
        "time_to_infection_hours": 1209.75,
        "degrees_of_separation_hops": 19,
        "collateral_damage": 1024
    },
    ...
}
```

**HTML map**: an interactive Folium map showing vessel trails coloured black
(general fleet) or red (infection chain leading to a chokepoint breach), with
chokepoint circles coloured green (breached) or blue (not reached).

Pre-computed results for all five infection rates are available in
`results/infection_rate_results.json`.

---

## Checkpointing

Simulation progress is saved to `checkpoints/` after every 10-minute time
slot using an atomic rename (write to `_tmp`, then rename) to prevent
corruption on interruption.  To resume an interrupted run, simply re-run
the same `simulate.py` command — it will detect and load the existing
checkpoint automatically.

---

## Epidemiological Model

`epidemiology.py` implements the analytical complement to the simulation.
When an adversary seeds the malware in *n₀* vessels simultaneously, the
expected global spread time follows:

```
T_n = T₁ / n₀^α
```

where T₁ is the single-seed baseline (measured from simulation), α is a
sublinear scaling exponent, and n₀ is the number of simultaneous seeds.
The sublinear exponent reflects geographic bottlenecks (chokepoints) that
limit parallelism.
