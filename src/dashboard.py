"""
dashboard.py
------------
Streamlit dashboard for the Smart Home IDS.

WHAT IT DOES:
1. Loads the trained Random Forest model + encoders
2. Simulates a live stream of network flows (drawn from a held-out
   generation function, same distributions as training data - this stands
   in for "live packet capture" which we're not doing in 2 days)
3. Classifies each flow AS IT ARRIVES, logs it to SQLite
4. Shows: live alert feed, attack-type distribution, per-device risk,
   confusion-style summary stats

RUN WITH: streamlit run src/dashboard.py
"""

import sys, os, time
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from db import init_db, log_detection, fetch_recent, fetch_all_for_stats, clear_db, LABEL_NAMES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

st.set_page_config(page_title="Smart Home IDS", page_icon="🛡️", layout="wide")

# ---------- Load model artifacts (cached so it's not reloaded every rerun) ----------
@st.cache_resource
def load_artifacts():
    clf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    device_encoder = joblib.load(f"{MODEL_DIR}/device_encoder.pkl")
    protocol_encoder = joblib.load(f"{MODEL_DIR}/protocol_encoder.pkl")
    return clf, device_encoder, protocol_encoder

clf, device_encoder, protocol_encoder = load_artifacts()
init_db()

DEVICES = list(device_encoder.classes_)

# ---------- Simulate one incoming flow (stand-in for live capture) ----------
def simulate_flow():
    """
    Randomly generates ONE realistic flow, mimicking either benign traffic
    or one of the 4 attack types, with a bias toward mostly-benign traffic
    (as real networks are) with occasional attacks - this makes the live
    feed feel authentic rather than an obvious 20%-each toy stream.
    """
    r = np.random.random()
    device = np.random.choice(DEVICES)
    if r < 0.75:  # 75% benign - realistic network baseline
        protocol = np.random.choice(["TCP", "UDP"], p=[0.7, 0.3])
        dst_port = np.random.choice([443, 80, 8883, 53, 1883])
        flow_duration_ms = max(50, np.random.normal(1500, 400))
        packet_count = max(1, np.random.poisson(12))
        byte_rate_bps = max(100, np.random.normal(2000, 600))
        flag_syn_ratio = np.random.uniform(0.0, 0.15)
    elif r < 0.82:  # DDoS
        protocol = np.random.choice(["UDP", "TCP"], p=[0.75, 0.25])
        dst_port = np.random.choice([80, 443])
        flow_duration_ms = max(5, np.random.normal(80, 30))
        packet_count = max(50, np.random.poisson(500))
        byte_rate_bps = max(50000, np.random.normal(500000, 120000))
        flag_syn_ratio = np.random.uniform(0.4, 0.9)
    elif r < 0.89:  # Port scan
        protocol = "TCP"
        dst_port = np.random.randint(1, 65535)
        flow_duration_ms = max(1, np.random.normal(15, 8))
        packet_count = max(1, np.random.poisson(2))
        byte_rate_bps = max(10, np.random.normal(300, 100))
        flag_syn_ratio = np.random.uniform(0.8, 1.0)
    elif r < 0.95:  # Brute force
        protocol = "TCP"
        dst_port = np.random.choice([22, 23, 8080])
        flow_duration_ms = max(20, np.random.normal(300, 100))
        packet_count = max(2, np.random.poisson(6))
        byte_rate_bps = max(100, np.random.normal(1200, 300))
        flag_syn_ratio = np.random.uniform(0.2, 0.4)
    else:  # Botnet C2
        protocol = "TCP"
        dst_port = np.random.choice([6667, 8443, 4444])
        flow_duration_ms = max(500, np.random.normal(5000, 800))
        packet_count = max(1, np.random.poisson(4))
        byte_rate_bps = max(10, np.random.normal(150, 40))
        flag_syn_ratio = np.random.uniform(0.0, 0.1)

    return {
        "device": device, "protocol": protocol, "dst_port": int(dst_port),
        "flow_duration_ms": flow_duration_ms, "packet_count": packet_count,
        "byte_rate_bps": byte_rate_bps, "flag_syn_ratio": flag_syn_ratio,
        "unique_dst_ips_per_src": max(1, np.random.poisson(1.2)),
    }

def classify_flow(flow):
    device_enc = device_encoder.transform([flow["device"]])[0]
    protocol_enc = protocol_encoder.transform([flow["protocol"]])[0]
    X = pd.DataFrame([{
        "device_enc": device_enc, "protocol_enc": protocol_enc,
        "dst_port": flow["dst_port"], "flow_duration_ms": flow["flow_duration_ms"],
        "packet_count": flow["packet_count"], "byte_rate_bps": flow["byte_rate_bps"],
        "flag_syn_ratio": flow["flag_syn_ratio"],
        "unique_dst_ips_per_src": flow["unique_dst_ips_per_src"],
    }])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    confidence = proba[pred]
    return LABEL_NAMES[pred], confidence

# ================= UI =================
st.title("🛡️ Smart Home Network Intrusion Detection System")
st.caption("Real-time ML-based detection of attacks on IoT devices (Random Forest classifier)")

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    running = st.toggle("▶ Simulate live traffic", value=False)
with col_b:
    n_per_tick = st.slider("Flows per refresh", 1, 20, 5)
with col_c:
    if st.button("🗑️ Clear all logs"):
        clear_db()
        st.rerun()

# Feed new flows into the DB if running
if running:
    for _ in range(n_per_tick):
        flow = simulate_flow()
        label, confidence = classify_flow(flow)
        log_detection(
            device=flow["device"], protocol=flow["protocol"], dst_port=flow["dst_port"],
            predicted_label=label, confidence=confidence,
            flow_duration_ms=flow["flow_duration_ms"], packet_count=flow["packet_count"],
            byte_rate_bps=flow["byte_rate_bps"],
        )

all_rows = fetch_all_for_stats()
recent_rows = fetch_recent(limit=100)

# ---------- Top metrics ----------
total = len(all_rows)
attacks = [r for r in all_rows if r["predicted_label"] != "BENIGN"]
critical = [r for r in all_rows if r["severity"] == "critical"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Flows Analyzed", total)
m2.metric("Attacks Detected", len(attacks))
m3.metric("Critical Alerts", len(critical))
m4.metric("Detection Rate", f"{(len(attacks)/total*100):.1f}%" if total else "0%")

st.divider()

# ---------- Charts ----------
if all_rows:
    df_all = pd.DataFrame(all_rows)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Attack Type Distribution")
        counts = df_all["predicted_label"].value_counts().reset_index()
        counts.columns = ["type", "count"]
        fig = px.pie(counts, names="type", values="count", hole=0.4,
                      color="type",
                      color_discrete_map={"BENIGN": "#2ecc71", "DDOS": "#e74c3c",
                                          "PORT_SCAN": "#f39c12", "BRUTE_FORCE": "#9b59b6",
                                          "BOTNET_C2": "#c0392b"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Risk by Device")
        device_attack = df_all[df_all["predicted_label"] != "BENIGN"]["device"].value_counts().reset_index()
        device_attack.columns = ["device", "attack_count"]
        if not device_attack.empty:
            fig2 = px.bar(device_attack, x="device", y="attack_count", color="attack_count",
                           color_continuous_scale="Reds")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No attacks detected yet on any device.")

st.divider()

# ---------- Live alert table ----------
st.subheader("🚨 Recent Detections")
if recent_rows:
    df_recent = pd.DataFrame(recent_rows)[[
        "timestamp", "device", "protocol", "dst_port",
        "predicted_label", "severity", "confidence"
    ]]
    def highlight_severity(row):
        color = {"critical": "background-color:#4a1414", "high": "background-color:#4a3814",
                 "medium": "background-color:#3a3a14", "none": ""}.get(row["severity"], "")
        return [color] * len(row)
    st.dataframe(df_recent.style.apply(highlight_severity, axis=1), use_container_width=True, height=400)
else:
    st.info("No flows logged yet. Toggle 'Simulate live traffic' above to start.")

if running:
    time.sleep(2)
    st.rerun()
