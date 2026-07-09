# 🛡️ Smart Home Network Intrusion Detection System

A machine-learning-based NIDS that classifies real home-network traffic
(IoT devices, laptops, phones, printers) into 9 attack categories using a
Random Forest trained on the **CIC IoT-2023** dataset, with a live Streamlit
dashboard, SHAP explainability, multi-channel alerting, and an ARP-spoof
live capture engine.

---

## 👉 The real system lives in [`smart-home-nids/`](smart-home-nids/)

Everything production-quality is there — see its
**[detailed README](smart-home-nids/README.md)** for architecture, setup,
dataset stats, feature list, quick-start, Docker instructions, and CI.

---

## Repository layout

```
.
├── smart-home-nids/        ← Active production system (start here)
│   ├── src/                  ML pipeline + live capture engine
│   ├── dashboard/            Streamlit pages
│   ├── config/               Pipeline & runtime configuration
│   ├── tests/                pytest test suite
│   ├── Dockerfile
│   └── README.md             ← Full documentation
│
└── legacy-prototype/       ← Archived initial proof-of-concept (not active)
    ├── src/                  Synthetic-data generator, minimal dashboard
    └── README.md             ← Why it was superseded
```

> **`legacy-prototype/`** was the original prototype built on simulated
> traffic with 4 attack classes and 8 features. It is preserved for history
> but is **not maintained** and should not be used as a reference.

---

## Quick start

```bash
cd smart-home-nids
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

See [`smart-home-nids/README.md`](smart-home-nids/README.md) for the full
setup guide including data preparation and model training.
