"""📋 Historical Logs — Searchable detection history with filters and export."""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from dashboard.styles import COLORS, metric_card, section_header, severity_badge
from src.database import NIDSDatabase


def render(db: NIDSDatabase) -> None:
    """Render the historical logs page."""

    st.markdown(
        f"""
        <h2 style="font-weight:700; color:{COLORS['text_primary']}; margin-bottom:1.5rem;">
            📋 Historical Detection Logs
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # ── Summary Stats ────────────────────────────────────────────────────
    stats = db.get_dashboard_stats()
    cols = st.columns(4)
    cards = [
        ("Total Records", f"{stats['total_flows']:,}", "📊"),
        ("Threats", f"{stats['threat_flows']:,}", "⚠️"),
        ("Active Alerts", f"{stats['pending_alerts']:,}", "🔔"),
        ("Detection Rate", f"{stats['detection_rate']:.1f}%", "📈"),
    ]
    for col, (label, value, icon) in zip(cols, cards):
        with col:
            st.markdown(metric_card(label, value, icon), unsafe_allow_html=True)

    # ── Search & Filters ─────────────────────────────────────────────────
    section_header("🔍 Search & Filter")

    with st.expander("Filter Options", expanded=True):
        filter_cols = st.columns(5)

        with filter_cols[0]:
            filter_prediction = st.selectbox(
                "Attack Type",
                ["All", "BENIGN", "DDoS", "DoS", "Mirai", "Recon",
                 "Spoofing", "BruteForce", "WebAttack", "Malware"],
                key="log_filter_prediction",
            )

        with filter_cols[1]:
            filter_severity = st.selectbox(
                "Severity",
                ["All", "Critical", "High", "Medium", "Low", "Info"],
                key="log_filter_severity",
            )

        with filter_cols[2]:
            filter_ip = st.text_input("Source IP (contains)", key="log_filter_ip")

        with filter_cols[3]:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=7),
                key="log_filter_start",
            )

        with filter_cols[4]:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                key="log_filter_end",
            )

    col_search, col_limit = st.columns([3, 1])
    with col_limit:
        result_limit = st.number_input("Max Results", 50, 5000, 500, step=50, key="log_limit")

    # ── Execute Search ───────────────────────────────────────────────────
    results = db.search_detections(
        prediction=filter_prediction if filter_prediction != "All" else None,
        severity=filter_severity if filter_severity != "All" else None,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) + "T23:59:59" if end_date else None,
        source_ip=filter_ip if filter_ip else None,
        limit=result_limit,
    )

    # ── Results ──────────────────────────────────────────────────────────
    section_header(f"📄 Results ({len(results):,} records)")

    if results:
        df = pd.DataFrame(results)

        # Format columns for display
        display_cols = ["id", "timestamp", "prediction", "confidence", "severity",
                        "source_ip", "dst_ip", "device_id"]
        available = [c for c in display_cols if c in df.columns]

        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            df[available],
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        # ── Export ───────────────────────────────────────────────────────
        st.markdown("")
        col_export1, col_export2, _ = st.columns([1, 1, 3])

        with col_export1:
            csv_data = df[available].to_csv(index=False)
            st.download_button(
                label="📥 Export as CSV",
                data=csv_data,
                file_name=f"nids_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="export_csv_btn",
            )

        with col_export2:
            json_data = df[available].to_json(orient="records", indent=2)
            st.download_button(
                label="📥 Export as JSON",
                data=json_data,
                file_name=f"nids_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="export_json_btn",
            )
    else:
        st.info("No records match the current filters.")

    # ── Alert Management ─────────────────────────────────────────────────
    section_header("🔔 Pending Alerts")

    alerts = db.get_unacknowledged_alerts()

    if alerts:
        st.markdown(
            f"<p style='color:{COLORS['accent_amber']}'>"
            f"<strong>{len(alerts)}</strong> unacknowledged alerts</p>",
            unsafe_allow_html=True,
        )

        df_alerts = pd.DataFrame(alerts)
        alert_display = ["id", "timestamp", "severity", "message"]
        available_alerts = [c for c in alert_display if c in df_alerts.columns]

        st.dataframe(
            df_alerts[available_alerts],
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        col_ack1, col_ack2, _ = st.columns([1, 1, 3])
        with col_ack1:
            alert_id = st.number_input("Alert ID to acknowledge", min_value=1, key="ack_alert_id")
        with col_ack2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ Acknowledge", key="ack_btn"):
                db.acknowledge_alert(alert_id)
                st.success(f"Alert {alert_id} acknowledged.")
                st.rerun()

        if st.button("✅ Acknowledge All", key="ack_all_btn"):
            for alert in alerts:
                db.acknowledge_alert(alert["id"])
            st.success(f"All {len(alerts)} alerts acknowledged.")
            st.rerun()
    else:
        st.success("✅ No pending alerts.")
