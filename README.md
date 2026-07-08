# 🛡️ Smart Home Network Intrusion Detection System

A machine-learning based Network Intrusion Detection System (NIDS) that monitors
flow-level traffic from smart home IoT devices and classifies it as benign or
one of four attack types, with a live Streamlit monitoring dashboard.

## Problem Statement

Consumer IoT devices (cameras, smart bulbs, thermostats, locks) are frequently
compromised due to weak default credentials, open legacy ports (e.g. Telnet),
and lack of built-in security monitoring — as demonstrated by real-world
botnets like Mirai. This project builds a lightweight, ML-based detection
layer that a home router or gateway could run to flag suspicious device
behavior in real time.

## Detected Attack Classes

| Class | Description | Real-world signature modeled |
|---|---|---|
| BENIGN | Normal IoT device traffic | Regular, low-volume, standard ports (443/MQTT-1883/DNS-53) |
| DDoS | Denial of service flood | Very high packet count, short duration, high byte rate |
| Port Scan | Reconnaissance sweep | Near-instant flows across many destination ports, high SYN ratio |
| Brute Force | Credential-stuffing attempts | Repeated short connections to admin ports (22/23/8080) |
| Botnet C2 | Command-and-control beaconing (Mirai-style) | Long-lived, low-volume, periodic traffic to unusual ports |

## Architecture

```
Traffic simulator (flow generator)
        │
Feature extraction (8 flow-level features)
        │
Random Forest Classifier (150 trees, balanced class weights)
        │
SQLite logging (every classified flow + severity)
        │
Streamlit dashboard (live feed, attack distribution, per-device risk)
```

## Features Used

`device, protocol, dst_port, flow_duration_ms, packet_count, byte_rate_bps,
flag_syn_ratio, unique_dst_ips_per_src`

These mirror the flow-level feature schema produced by CICFlowMeter (the tool
behind the CIC-IoT-2023 dataset), so the pipeline is compatible with real
captured traffic if swapped in.

## Model Performance

- **Accuracy: 99.82%** on held-out test set (20% split, stratified)
- Per-class precision/recall/F1 all ≥0.99
- Feature importance analysis shows `flow_duration_ms`, `dst_port`, and
  `flag_syn_ratio` as the strongest predictors — consistent with published
  IDS literature on scan/flood detection signatures

## Tech Stack

Python · scikit-learn (Random Forest) · pandas/numpy · SQLite · Streamlit · Plotly

## Setup & Run

```bash
pip install -r requirements.txt
python src/generate_dataset.py   # builds data/iot_traffic.csv
python src/train_model.py         # trains + evaluates + saves model
streamlit run src/dashboard.py    # launches live dashboard
```

## Project Structure

```
smart-home-ids/
├── data/               # generated traffic dataset
├── models/             # trained RF model + label encoders
├── logs/               # SQLite detection database
├── src/
│   ├── generate_dataset.py   # synthetic flow generator w/ realistic noise
│   ├── train_model.py        # RF training + evaluation
│   ├── db.py                  # SQLite logging layer
│   └── dashboard.py           # Streamlit live monitoring UI
├── requirements.txt
└── README.md
```

## Limitations & Future Work

- Traffic is simulated with realistic statistical distributions rather than
  captured live; swapping in CIC-IoT-2023 or live Scapy capture would be the
  next step toward production readiness.
- No deep learning / sequence modeling (e.g. LSTM on packet sequences) —
  a deliberate tradeoff for interpretability and fast training.
- Single-node detection only; no distributed/federated detection across
  multiple home gateways.

