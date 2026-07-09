"""🔴 Live Monitoring — Real-time network flow classification.

Three input modes:
- ✏️  Manual Input   — enter the 18 CIC-IoT-2023 feature values by hand
- 🎲  Simulate       — generate synthetic flows with realistic distributions
- 📡  Live Capture   — real Scapy packet capture (Phase 4, coming soon)
"""

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

# ── Feature metadata for the manual input form ──────────────────────────────
# Groups: (group_label, [(feature_name, display_label, default, fmt, help_text)])
# Feature order matches selected_features.pkl (verified from live artifact).
_FEATURE_GROUPS: list[tuple[str, list[tuple[str, str, float, str, str]]]] = [
    (
        "⏱️ Flow Timing",
        [
            ("flow_duration", "Flow Duration (s)", 1.0, "%.4f",
             "Total duration of the network flow in seconds."),
            ("iat", "Mean Inter-Arrival Time (s)", 0.05, "%.6f",
             "Average time between consecutive packets in the flow."),
            ("rate", "Packet Rate (pkts/s)", 50.0, "%.2f",
             "Number of packets per second in the flow."),
            ("packets_per_second", "Packets per Second", 50.0, "%.2f",
             "Alias/cross-check for packet rate; may differ if computed over sub-windows."),
        ],
    ),
    (
        "📦 Packet Sizes",
        [
            ("tot_size", "Total Payload Size (bytes)", 50000.0, "%.1f",
             "Sum of all payload bytes transferred in the flow."),
            ("tot_sum", "Total Bytes (tot_sum)", 50000.0, "%.1f",
             "Alias for total size; kept as a separate feature by CICFlowMeter."),
            ("avg", "Mean Packet Size (bytes)", 500.0, "%.2f",
             "Average payload size per packet."),
            ("max", "Max Packet Size (bytes)", 1460.0, "%.1f",
             "Size of the largest packet in the flow."),
            ("min", "Min Packet Size (bytes)", 40.0, "%.1f",
             "Size of the smallest packet in the flow."),
            ("variance", "Packet Size Variance", 80000.0, "%.2f",
             "Statistical variance of packet sizes; high values indicate bursty traffic."),
        ],
    ),
    (
        "💾 Byte Rates",
        [
            ("bytes_per_packet", "Bytes per Packet", 500.0, "%.2f",
             "Average bytes transferred per packet."),
        ],
    ),
    (
        "📋 Header Info",
        [
            ("header_length", "Total Header Length (bytes)", 2400.0, "%.1f",
             "Sum of all IP/TCP/UDP header bytes across all packets."),
            ("header_bytes_per_packet", "Header Bytes per Packet", 48.0, "%.2f",
             "Average header overhead per packet."),
        ],
    ),
    (
        "🚩 TCP Flags",
        [
            ("rst_count", "RST Flag Count", 0.0, "%.0f",
             "Number of packets with the TCP RST flag set."),
            ("urg_count", "URG Flag Count", 0.0, "%.0f",
             "Number of packets with the TCP URG flag set."),
            ("rst_ratio", "RST Flag Ratio", 0.0, "%.4f",
             "Fraction of packets carrying a RST flag (rst_count / packet_count)."),
            ("urg_ratio", "URG Flag Ratio", 0.0, "%.4f",
             "Fraction of packets carrying a URG flag (urg_count / packet_count)."),
        ],
    ),
    (
        "🔌 Protocol",
        [
            ("protocol_type", "Protocol Type (numeric)", 6.0, "%.0f",
             "IP protocol number: 6 = TCP, 17 = UDP, 1 = ICMP."),
        ],
    ),
]

# Flat ordered list of all 18 feature names — matches selected_features.pkl exactly.
# Verified: python3 -c "import joblib; print(joblib.load('models/selected_features.pkl'))"
ALL_FEATURES: list[str] = [feat for _, grp in _FEATURE_GROUPS for feat, *_ in grp]


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

    # ── Input Mode Tabs ──────────────────────────────────────────────────
    tab_live, tab_simulate, tab_manual = st.tabs([
        "📡 Live Capture",
        "🎲 Simulate Traffic",
        "✏️ Manual Input",
    ])

    with tab_live:
        _render_live_capture(db, predictor, alert_mgr)

    with tab_simulate:
        _render_simulation(db, predictor, alert_mgr)

    with tab_manual:
        _render_manual_input(db, predictor, alert_mgr)

    # ── Live Feed ────────────────────────────────────────────────────────
    section_header("📡 Recent Predictions Feed")
    _render_live_feed(db)


# ── Manual Input ──────────────────────────────────────────────────────────────

def _render_manual_input(
    db: NIDSDatabase,
    predictor: NIDSPredictor,
    alert_mgr: AlertManager,
) -> None:
    """Manual entry of the 18 CIC-IoT-2023 flow features for a single prediction."""
    st.markdown(
        f"<p style='color:{COLORS['text_secondary']}'>"
        f"Enter the 18 network-flow feature values below, then click "
        f"<strong>Classify Flow</strong>. Missing values are filled with "
        f"training-set medians automatically.</p>",
        unsafe_allow_html=True,
    )

    feature_values: dict[str, float] = {}

    with st.form(key="manual_input_form"):
        # Render each feature group
        for group_label, features in _FEATURE_GROUPS:
            st.markdown(f"**{group_label}**")
            cols = st.columns(min(len(features), 3))
            for idx, (name, label, default, fmt, help_text) in enumerate(features):
                with cols[idx % 3]:
                    feature_values[name] = st.number_input(
                        label,
                        value=default,
                        format=fmt,
                        key=f"manual_{name}",
                        help=help_text,
                    )

        st.markdown("**🔍 Tracking Details** *(optional)*")
        col_src, col_dst, col_dev = st.columns(3)
        with col_src:
            src_ip = st.text_input("Source IP", value="", key="manual_src_ip",
                                   help="Source IPv4 address (for logging only).")
        with col_dst:
            dst_ip = st.text_input("Destination IP", value="", key="manual_dst_ip",
                                   help="Destination IPv4 address (for logging only).")
        with col_dev:
            device_id = st.text_input("Device ID", value="", key="manual_device_id",
                                      help="Unique device identifier (MAC or hostname).")

        submit = st.form_submit_button("🔍 Classify Flow")

    if submit:
        with st.spinner("Classifying…"):
            result = predictor.predict_single(feature_values)

        # Result cards
        col_pred, col_conf, col_sev = st.columns(3)
        with col_pred:
            st.markdown(
                metric_card("Prediction", result.label, "🏷️",
                            color=ATTACK_COLORS.get(result.label, COLORS["accent_cyan"])),
                unsafe_allow_html=True,
            )
        with col_conf:
            st.markdown(
                metric_card("Confidence", f"{result.confidence:.1%}", "📊",
                            color=COLORS["accent_cyan"]),
                unsafe_allow_html=True,
            )
        with col_sev:
            st.markdown(
                metric_card("Severity", result.severity, "⚡",
                            color=SEVERITY_COLORS.get(result.severity, COLORS["accent_cyan"])),
                unsafe_allow_html=True,
            )

        # Top-3 probability bars
        st.markdown("**Top-3 Predictions:**")
        for cls, prob in result.top_3:
            bar_width = int(prob * 100)
            color = ATTACK_COLORS.get(cls, COLORS["text_secondary"])
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; margin:4px 0;">
                    <span style="width:110px; font-size:0.85rem;
                                 color:{COLORS['text_primary']}">{cls}</span>
                    <div style="flex:1; background:{COLORS['bg_secondary']};
                                border-radius:4px; height:20px; margin:0 10px;">
                        <div style="width:{bar_width}%; background:{color};
                                    height:100%; border-radius:4px;"></div>
                    </div>
                    <span style="width:60px; text-align:right; font-size:0.85rem;
                                 color:{COLORS['text_secondary']}">{prob:.1%}</span>
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
            model_version=predictor.model_version,
        )
        alert_mgr.process_alert(
            det_id, result.label, result.confidence, result.severity, src_ip or None
        )
        st.toast(f"✅ Classified as **{result.label}** — saved to DB.")


# ── Simulation ────────────────────────────────────────────────────────────────

def _render_simulation(
    db: NIDSDatabase,
    predictor: NIDSPredictor,
    alert_mgr: AlertManager,
) -> None:
    """Synthetic traffic simulation using realistic statistical distributions."""
    st.markdown(
        f"<p style='color:{COLORS['text_secondary']}'>"
        f"Generate synthetic network flows with realistic CIC-IoT-2023 "
        f"style distributions for demonstration purposes.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        n_flows = st.slider("Number of flows", 10, 500, 50, key="sim_count")
    with col2:
        sim_speed = st.selectbox(
            "Speed", ["Instant", "Fast (0.1s)", "Normal (0.5s)"], key="sim_speed"
        )

    if st.button("🚀 Start Simulation", key="start_sim"):
        _run_simulation(db, predictor, alert_mgr, n_flows, sim_speed)


def _synthetic_flow() -> dict[str, float]:
    """Generate one synthetic flow with realistic CIC-IoT-2023 style values."""
    flow_dur = max(float(np.random.exponential(1.5)), 1e-6)
    pkt_count = max(int(np.random.poisson(60)), 1)
    tot_size = float(np.random.exponential(55_000))
    rst_count = float(np.random.poisson(0.4))
    urg_count = float(np.random.poisson(0.05))

    return {
        # Timing
        "flow_duration": flow_dur,
        "iat": float(np.random.exponential(0.04)),
        "rate": pkt_count / flow_dur,
        "packets_per_second": pkt_count / flow_dur,
        # Sizes
        "tot_size": tot_size,
        "tot_sum": tot_size,                                      # alias
        "avg": float(abs(np.random.normal(520, 220))),
        "max": float(np.random.randint(64, 1461)),
        "min": float(np.random.randint(40, 200)),
        "variance": float(abs(np.random.exponential(95_000))),
        # Byte rates
        "bytes_per_packet": tot_size / pkt_count,
        # Header
        "header_length": float(np.random.randint(200, 5_001)),
        "header_bytes_per_packet": float(abs(np.random.normal(48, 10))),
        # TCP flags
        "rst_count": rst_count,
        "urg_count": urg_count,
        "rst_ratio": rst_count / pkt_count,
        "urg_ratio": urg_count / pkt_count,
        # Protocol: 6=TCP (70%), 17=UDP (25%), 1=ICMP (5%)
        "protocol_type": float(np.random.choice([6, 17, 1], p=[0.70, 0.25, 0.05])),
    }


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

    predictions: list[dict] = []

    for i in range(n_flows):
        features = _synthetic_flow()
        result = predictor.predict_single(features)

        src_ip = f"192.168.1.{np.random.randint(2, 254)}"
        dev_id = f"device_{np.random.randint(1, 30):02d}"

        det_id = db.insert_detection(
            prediction=result.label,
            confidence=result.confidence,
            severity=result.severity,
            features=features,
            source_ip=src_ip,
            device_id=dev_id,
            model_version=predictor.model_version,
        )
        alert_mgr.process_alert(
            det_id, result.label, result.confidence, result.severity, src_ip
        )

        predictions.append({
            "prediction": result.label,
            "confidence": result.confidence,
            "severity": result.severity,
            "source_ip": src_ip,
            "device_id": dev_id,
        })

        progress_bar.progress((i + 1) / n_flows)
        status_text.text(
            f"Processed {i + 1}/{n_flows} flows — "
            f"Latest: {result.label} ({result.confidence:.1%})"
        )

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


def _render_live_capture(
    db: NIDSDatabase,
    predictor: NIDSPredictor,
    alert_mgr: AlertManager,
) -> None:
    """Live Scapy packet capture tab."""
    from src.capture.sniffer import list_interfaces

    # ── Privilege warning ─────────────────────────────────────
    st.markdown(
        f"""
        <div style="padding:0.8rem 1rem; background:rgba(239,68,68,0.12);
                    border-left:3px solid {COLORS['accent_red']};
                    border-radius:6px; margin-bottom:1rem;">
            <strong style="color:{COLORS['accent_red']}">⚠️ Root/Admin Required</strong><br>
            <span style="color:{COLORS['text_secondary']}; font-size:0.85rem;">
            Live capture uses raw sockets. Launch the app with
            <code>sudo streamlit run app.py</code> on macOS/Linux, or grant
            <code>CAP_NET_RAW + CAP_NET_ADMIN</code> to the Python binary (Phase 5).
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Interface selector ─────────────────────────────────
    ifaces = list_interfaces()
    iface_names = [i["name"] for i in ifaces] or ["en0"]
    iface_labels = [
        f"{i['name']} ({i['ip']})" if i.get("ip") else i["name"]
        for i in ifaces
    ] or ["en0"]

    col_iface, col_timeout = st.columns(2)
    with col_iface:
        selected_idx = st.selectbox(
            "Network Interface",
            range(len(iface_labels)),
            format_func=lambda i: iface_labels[i],
            key="live_iface",
        )
        selected_iface = iface_names[selected_idx]

    with col_timeout:
        idle_timeout = st.slider(
            "Flow Idle Timeout (s)", 10, 120, 30,
            key="live_idle_timeout",
            help="Seconds of inactivity before a flow is classified. "
                 "CICFlowMeter default is 120 s.",
        )

    # ── Optional ARP spoofing ─────────────────────────────
    with st.expander("🚀 ARP Spoof (MITM) — optional", expanded=False):
        st.markdown(
            f"<p style='color:{COLORS['text_secondary']}; font-size:0.85rem;'>"
            f"Enable to route traffic from other devices through this machine for capture. "
            f"Leave blank to capture only traffic to/from this machine.</p>",
            unsafe_allow_html=True,
        )
        targets_raw = st.text_input(
            "Target IP(s) — comma-separated",
            value="",
            key="live_targets",
            placeholder="192.168.1.10, 192.168.1.20",
        )
        gateway_ip = st.text_input(
            "Gateway IP",
            value="192.168.1.1",
            key="live_gateway",
        )

    targets = [t.strip() for t in targets_raw.split(",") if t.strip()] if targets_raw else []

    # ── Start / Stop controls ────────────────────────────
    pipeline_key = "live_capture_pipeline"
    col_start, col_stop, col_status = st.columns([1, 1, 2])

    pipeline = st.session_state.get(pipeline_key)
    is_running = pipeline is not None and pipeline.is_running

    with col_start:
        start_disabled = is_running
        if st.button("▶ Start Capture", disabled=start_disabled, key="live_start",
                     type="primary"):
            try:
                from src.capture.pipeline import CapturePipeline
                p = CapturePipeline(
                    iface=selected_iface,
                    predictor=predictor,
                    db=db,
                    alert_mgr=alert_mgr,
                    targets=targets,
                    gateway=gateway_ip if targets else None,
                    idle_timeout=float(idle_timeout),
                )
                p.start()
                st.session_state[pipeline_key] = p
                st.success(f"✅ Capture started on {selected_iface}")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Failed to start capture: {exc}")

    with col_stop:
        stop_disabled = not is_running
        if st.button("⏹ Stop", disabled=stop_disabled, key="live_stop"):
            if pipeline:
                pipeline.stop()
                st.session_state[pipeline_key] = None
                st.info("⏹ Capture stopped.")
                st.rerun()

    with col_status:
        if is_running:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:8px; margin-top:0.5rem;">
                    <span class="live-indicator"></span>
                    <span style="color:{COLORS['accent_green']}; font-weight:600;">
                        Capturing on {pipeline.iface}
                    </span>
                </div>
                <p style="color:{COLORS['text_secondary']}; font-size:0.8rem; margin:4px 0 0 0;">
                    Flows: {pipeline.flows_classified} &nbsp;|&nbsp;
                    Threats: {pipeline.threats_detected}
                </p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color:{COLORS['text_secondary']}; margin-top:0.5rem;'>"
                f"⏸️ Capture stopped</p>",
                unsafe_allow_html=True,
            )

    # ── Live results feed ────────────────────────────────
    if is_running:
        st.markdown("---")
        st.markdown(f"**Live Classified Flows** (auto-refreshes as flows complete)")

        # Drain result_queue and accumulate in session_state
        live_results_key = "live_capture_results"
        if live_results_key not in st.session_state:
            st.session_state[live_results_key] = []

        new_results: list[dict] = []
        while True:
            try:
                rec = pipeline.result_queue.get_nowait()
                new_results.append(rec)
            except Exception:
                break

        if new_results:
            st.session_state[live_results_key] = (
                new_results + st.session_state[live_results_key]
            )[:100]  # keep last 100

        results = st.session_state.get(live_results_key, [])
        if results:
            rows = [
                {
                    "prediction": r["prediction"],
                    "confidence": f"{r['confidence']:.1%}",
                    "severity": r["severity"],
                    "protocol": int(r["features"].get("protocol_type", 0)),
                    "flow_duration": f"{r['features'].get('flow_duration', 0):.2f}s",
                }
                for r in results
            ]
            st.dataframe(
                rows, use_container_width=True, hide_index=True, height=400
            )
        else:
            st.info("🔍 Waiting for flows to complete… "
                    "(flows are emitted after idle timeout or max packet threshold)")

        # Auto-rerun every 3 s while capture is active
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=3000, key="live_autorefresh")
        except ImportError:
            st.caption("⚡ Install streamlit-autorefresh for automatic updates.")


# ── Live Feed ─────────────────────────────────────────────────────────────────

def _render_live_feed(db: NIDSDatabase) -> None:
    """Show the most recent predictions in a styled table."""
    recent = db.get_recent_detections(limit=25)

    if recent:
        df = pd.DataFrame(recent)
        display_cols = ["timestamp", "prediction", "confidence", "severity",
                        "source_ip", "device_id"]
        available = [c for c in display_cols if c in df.columns]

        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )

        st.dataframe(df[available], use_container_width=True, hide_index=True, height=500)
    else:
        st.info("No predictions yet. Use the tabs above to classify network flows.")
