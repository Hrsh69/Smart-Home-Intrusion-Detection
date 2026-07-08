"""🔴 Live Monitoring — Real-time network flow classification."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.styles import (
    ATTACK_COLORS,
    COLORS,
    SEVERITY_COLORS,
    apply_plotly_theme,
    metric_card,
    section_header,
    severity_badge,
)
from src.alerts import AlertManager
from src.database import NIDSDatabase
from src.predict import NIDSPredictor


def render(db: NIDSDatabase, predictor: NIDSPredictor, alert_mgr: AlertManager) -> None:
    """Render the live monitoring page."""

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:1.5rem;">
            <span class="live-indicator"></span>
            <h2 style="margin:0; font-weight:700; color:{COLORS['text_primary']};">
                Live Network Monitor
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input Method Selection ───────────────────────────────────────────
    tab_manual, tab_simulate = st.tabs([
        "✏️ Manual Input",
        "🎲 Simulate Traffic",
    ])

    # ── Manual Input ─────────────────────────────────────────────────────
    with tab_manual:
        _render_manual_input(db, predictor, alert_mgr)

    # ── Simulate ─────────────────────────────────────────────────────────
    with tab_simulate:
        _render_simulation(db, predictor, alert_mgr)

    # ── Live Feed ────────────────────────────────────────────────────────
    section_header("📡 Recent Predictions Feed")
    _render_live_feed(db)


def _render_manual_input(
    db: NIDSDatabase,
    predictor: NIDSPredictor,
    alert_mgr: AlertManager,
) -> None:
    """Handle manual feature input for single prediction."""
    st.markdown(
        f"<p style='color:{COLORS['text_secondary']}'>Enter network flow feature values manually.</p>",
        unsafe_allow_html=True,
    )

    feature_values = {}

    with st.form(key="manual_input_form"):
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            feature_values["device_type"] = st.selectbox(
                "Device Type",
                ["smart_camera", "smart_speaker", "smart_bulb", "thermostat", "door_lock"],
                key="manual_device_type",
                help="The type of IoT device generating the flow."
            )
        with col_cat2:
            feature_values["protocol"] = st.selectbox(
                "Protocol",
                ["TCP", "UDP"],
                key="manual_protocol",
                help="Transport layer protocol used."
            )

        # Number inputs for the rest
        st.markdown("**Flow Metrics**")
        col_num1, col_num2, col_num3 = st.columns(3)
        
        with col_num1:
            feature_values["dst_port"] = st.number_input("Destination Port", value=443, format="%d", key="manual_dst_port", help="The destination port number (e.g., 443 for HTTPS, 53 for DNS).")
            feature_values["byte_rate_bps"] = st.number_input("Byte Rate (bps)", value=15000.0, format="%.2f", key="manual_byte_rate", help="Data transfer rate in bytes per second.")
            
        with col_num2:
            feature_values["flow_duration_ms"] = st.number_input("Flow Duration (ms)", value=1200.0, format="%.2f", key="manual_flow_dur", help="Total duration of the flow in milliseconds.")
            feature_values["flag_syn_ratio"] = st.number_input("SYN Flag Ratio", value=0.1, format="%.4f", key="manual_syn_ratio", help="Ratio of packets with the SYN flag set (useful for detecting SYN floods).")
            
        with col_num3:
            feature_values["packet_count"] = st.number_input("Packet Count", value=50, format="%d", key="manual_pkt_count", help="Total number of packets in the flow.")
            feature_values["unique_dst_ips_per_src"] = st.number_input("Unique Dest IPs / Source", value=1.0, format="%.2f", key="manual_uniq_ips", help="Number of unique destination IP addresses contacted by this source IP.")

        st.markdown("**Tracking Details**")
        col_ip1, col_ip2, col_dev = st.columns(3)
        with col_ip1:
            src_ip = st.text_input("Source IP (optional)", value="", key="manual_src_ip", help="Source IPv4 address.")
        with col_ip2:
            dst_ip = st.text_input("Dest IP (optional)", value="", key="manual_dst_ip", help="Destination IPv4 address.")
        with col_dev:
            device_id = st.text_input("Device ID (optional)", value="", key="manual_device", help="Unique identifier for the device (e.g., MAC address).")

        submit_button = st.form_submit_button("🔍 Classify Flow")

    if submit_button:
        with st.spinner("Classifying..."):
            result = predictor.predict_single(feature_values)

        # Display result
        col_pred, col_conf, col_sev = st.columns(3)
        with col_pred:
            st.markdown(
                metric_card("Prediction", result.label, "🏷️", color=ATTACK_COLORS.get(result.label, COLORS["accent_cyan"])),
                unsafe_allow_html=True,
            )
        with col_conf:
            st.markdown(
                metric_card("Confidence", f"{result.confidence:.1%}", "📊", color=COLORS["accent_cyan"]),
                unsafe_allow_html=True,
            )
        with col_sev:
            st.markdown(
                metric_card("Severity", result.severity, "⚡", color=SEVERITY_COLORS.get(result.severity, COLORS["accent_cyan"])),
                unsafe_allow_html=True,
            )

        # Top 3 probabilities
        st.markdown(f"**Top-3 Predictions:**")
        for cls, prob in result.top_3:
            bar_width = int(prob * 100)
            color = ATTACK_COLORS.get(cls, COLORS["text_secondary"])
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; margin:4px 0;">
                    <span style="width:100px; font-size:0.85rem; color:{COLORS['text_primary']}">{cls}</span>
                    <div style="flex:1; background:{COLORS['bg_secondary']}; border-radius:4px; height:20px; margin:0 10px;">
                        <div style="width:{bar_width}%; background:{color}; height:100%; border-radius:4px;"></div>
                    </div>
                    <span style="width:60px; text-align:right; font-size:0.85rem; color:{COLORS['text_secondary']}">{prob:.1%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Log to database
        det_id = db.insert_detection(
            prediction=result.label,
            confidence=result.confidence,
            severity=result.severity,
            features=feature_values,
            source_ip=src_ip or None,
            dst_ip=dst_ip or None,
            device_id=device_id or None,
        )
        alert_mgr.process_alert(det_id, result.label, result.confidence, result.severity, src_ip or None)
        st.toast(f"✅ Flow classified as {result.label} and saved.")


def _render_simulation(
    db: NIDSDatabase,
    predictor: NIDSPredictor,
    alert_mgr: AlertManager,
) -> None:
    """Simulate network traffic from test data for demo purposes."""
    st.markdown(
        f"<p style='color:{COLORS['text_secondary']}'>Generate synthetic network flows for demonstration.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        n_flows = st.slider("Number of flows to simulate", 10, 500, 50, key="sim_count")
    with col2:
        sim_speed = st.selectbox("Speed", ["Instant", "Fast (0.1s)", "Normal (0.5s)"], key="sim_speed")

    if st.button("🚀 Start Simulation", key="start_sim"):
        _run_simulation(db, predictor, alert_mgr, n_flows, sim_speed)


def _run_simulation(
    db: NIDSDatabase,
    predictor: NIDSPredictor,
    alert_mgr: AlertManager,
    n_flows: int,
    speed: str,
) -> None:
    """Execute the synthetic traffic simulation."""

    delay = {"Instant": 0, "Fast (0.1s)": 0.1, "Normal (0.5s)": 0.5}.get(speed, 0)

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    predictions = []
    
    # Pre-generate synthetic data
    devices = ["door_lock", "smart_bulb", "smart_camera", "smart_speaker", "thermostat"]
    protocols = ["TCP", "UDP"]
    
    for i in range(n_flows):
        # Synthetic mock flow
        features = {
            "device_type": np.random.choice(devices),
            "protocol": np.random.choice(protocols),
            "dst_port": float(np.random.choice([80, 443, 53, 8080, 22, 23])),
            "flow_duration_ms": float(np.random.exponential(1000)),
            "packet_count": float(np.random.poisson(50)),
            "byte_rate_bps": float(np.random.exponential(20000)),
            "flag_syn_ratio": float(np.random.uniform(0, 1)),
            "unique_dst_ips_per_src": float(np.random.poisson(2) + 1),
        }
        
        result = predictor.predict_single(features)

        src_ip = f"192.168.1.{np.random.randint(2, 254)}"
        dev_id = f"{features['device_type']}_{np.random.randint(1, 20):02d}"

        det_id = db.insert_detection(
            prediction=result.label,
            confidence=result.confidence,
            severity=result.severity,
            features=features,
            source_ip=src_ip,
            device_id=dev_id,
        )
        alert_mgr.process_alert(det_id, result.label, result.confidence, result.severity, src_ip)

        predictions.append({
            "prediction": result.label,
            "confidence": result.confidence,
            "severity": result.severity,
            "source_ip": src_ip,
            "device_id": dev_id,
        })

        progress_bar.progress((i + 1) / n_flows)
        status_text.text(f"Processed {i + 1}/{n_flows} flows — Latest: {result.label} ({result.confidence:.1%})")

        if delay > 0:
            time.sleep(delay)

    progress_bar.empty()
    status_text.empty()
    st.toast(f"✅ Simulation complete — {n_flows} flows classified")

    with results_container:
        pred_df = pd.DataFrame(predictions)
        _display_simulation_summary(pred_df)


def _display_simulation_summary(pred_df: pd.DataFrame) -> None:
    """Show summary charts after simulation."""
    col1, col2 = st.columns(2)

    with col1:
        dist = pred_df["prediction"].value_counts()
        colors = [ATTACK_COLORS.get(cat, COLORS["text_secondary"]) for cat in dist.index]

        fig = go.Figure(go.Pie(
            labels=dist.index.tolist(),
            values=dist.values.tolist(),
            marker=dict(colors=colors),
            hole=0.4,
            textinfo="label+percent",
        ))
        apply_plotly_theme(fig)
        fig.update_layout(height=300, title="Attack Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sev_dist = pred_df["severity"].value_counts()
        ordered = ["Critical", "High", "Medium", "Low", "Info"]
        labels = [s for s in ordered if s in sev_dist.index]
        vals = [sev_dist[s] for s in labels]
        colors = [SEVERITY_COLORS.get(s, COLORS["text_secondary"]) for s in labels]

        fig = go.Figure(go.Bar(
            x=labels, y=vals,
            marker=dict(color=colors),
            text=vals, textposition="outside",
            textfont=dict(color=COLORS["text_primary"]),
        ))
        apply_plotly_theme(fig)
        fig.update_layout(height=300, title="Severity Breakdown")
        st.plotly_chart(fig, use_container_width=True)


def _render_live_feed(db: NIDSDatabase) -> None:
    """Show the most recent predictions in a styled table."""
    recent = db.get_recent_detections(limit=25)

    if recent:
        df = pd.DataFrame(recent)
        display_cols = ["timestamp", "prediction", "confidence", "severity", "source_ip", "device_id"]
        available = [c for c in display_cols if c in df.columns]

        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

        st.dataframe(df[available], use_container_width=True, hide_index=True, height=500)
    else:
        st.info("No predictions yet. Use the tabs above to classify network flows.")
