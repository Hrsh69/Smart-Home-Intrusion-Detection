# 📦 Legacy Prototype — Smart Home NIDS (Initial Proof-of-Concept)

> ⚠️ **This directory is archived.** It is preserved for historical reference only
> and is **not the active codebase**. See [`../smart-home-nids/`](../smart-home-nids/) for the
> production system.

---

## What this was

This was the **initial proof-of-concept** built before the real dataset was available.
It uses **fully simulated traffic** generated from hand-crafted statistical
distributions rather than real packet captures or the CIC IoT-2023 dataset.

| Property | Value |
|---|---|
| Dataset | Synthetic (hand-generated, `generate_dataset.py`) |
| Attack classes | 4 (BENIGN, DDoS, Port Scan, Brute Force, Botnet C2) |
| Features | 8 flow-level features |
| Model | Random Forest, 150 trees |
| Reported accuracy | 99.82% (on synthetic test split — inflated due to simulated data) |

## Files

| File | Purpose |
|---|---|
| `src/generate_dataset.py` | Generates synthetic `iot_traffic.csv` using NumPy distributions |
| `src/train_model.py` | Trains + evaluates RF classifier, saves `models/rf_model.pkl` |
| `src/db.py` | Minimal SQLite logging layer |
| `src/dashboard.py` | Basic Streamlit dashboard (live feed + attack distribution) |

## Why it was superseded

The simulated data made the 99.82% accuracy meaningless — the model was
memorising the very distributions it was trained on. The production system in
`smart-home-nids/` replaces this with:

- Real **CIC IoT-2023** traffic captures (7.8 M rows)
- **9 attack categories** instead of 4 (including Mirai, Spoofing, WebAttack, Malware)
- **18 CICFlowMeter features** instead of 8 synthetic ones
- Live Scapy packet capture + ARP spoof MITM engine
- SHAP explainability, multi-channel alerting, Docker, CI

## Running the prototype (for reference)

```bash
cd <repo-root>
pip install -r requirements.txt    # root-level (minimal deps)
python legacy-prototype/src/generate_dataset.py
python legacy-prototype/src/train_model.py
streamlit run legacy-prototype/src/dashboard.py
```
