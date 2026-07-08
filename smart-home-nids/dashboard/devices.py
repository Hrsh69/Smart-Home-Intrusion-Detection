"""🖥️ Device Statistics — Device registry, risk scores, flow history."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.styles import (
    COLORS,
    SEVERITY_COLORS,
    apply_plotly_theme,
    metric_card,
    section_header,
)
from src.database import NIDSDatabase


def render(db: NIDSDatabase) -> None:
    """Render the device statistics page."""

    st.markdown(
        f"""
        <h2 style="font-weight:700; color:{COLORS['text_primary']}; margin-bottom:1.5rem;">
            🖥️ Device Statistics & Risk Assessment
        </h2>
        """,
        unsafe_allow_html=True,
    )

    devices = db.get_all_devices()

    if not devices:
        st.info("No devices registered yet.")
        st.markdown(
            "To populate device data and test the dashboard, please go to the **🔴 Live Monitor** tab "
            "and run a traffic simulation, or manually classify a flow."
        )
        return

    # ── Summary Cards ────────────────────────────────────────────────────
    total_devices = len(devices)
    high_risk = sum(1 for d in devices if d["risk_score"] > 50)
    total_flows = sum(d["total_flows"] for d in devices)
    total_malicious = sum(d["malicious_flows"] for d in devices)

    cols = st.columns(4)
    cards = [
        ("Total Devices", str(total_devices), "🖥️", COLORS["accent_cyan"]),
        ("High Risk", str(high_risk), "🔴", COLORS["accent_red"]),
        ("Total Flows", f"{total_flows:,}", "📡", COLORS["accent_purple"]),
        ("Malicious Flows", f"{total_malicious:,}", "⚠️", COLORS["accent_amber"]),
    ]
    for col, (label, value, icon, color) in zip(cols, cards):
        with col:
            st.markdown(metric_card(label, value, icon, color=color), unsafe_allow_html=True)

    # ── Device Risk Ranking ──────────────────────────────────────────────
    section_header("🏆 Device Risk Ranking")

    df_devices = pd.DataFrame(devices)

    # Risk score bar chart
    top_devices = df_devices.sort_values("risk_score", ascending=False).head(20)

    colors = []
    for score in top_devices["risk_score"]:
        if score >= 75:
            colors.append(SEVERITY_COLORS["Critical"])
        elif score >= 50:
            colors.append(SEVERITY_COLORS["High"])
        elif score >= 25:
            colors.append(SEVERITY_COLORS["Medium"])
        else:
            colors.append(SEVERITY_COLORS["Info"])

    fig_risk = go.Figure(go.Bar(
        y=top_devices["name"].tolist(),
        x=top_devices["risk_score"].tolist(),
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in top_devices["risk_score"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_primary"]),
        hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.1f}%<extra></extra>",
    ))
    apply_plotly_theme(fig_risk)
    fig_risk.update_layout(
        height=max(300, len(top_devices) * 35),
        title="Top Devices by Risk Score",
        xaxis=dict(title="Risk Score (%)", range=[0, 110]),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # ── Risk Score Gauges for Top 6 ──────────────────────────────────────
    section_header("📊 Risk Score Gauges")

    top_6 = df_devices.sort_values("risk_score", ascending=False).head(6)
    gauge_cols = st.columns(3)

    for i, (_, device) in enumerate(top_6.iterrows()):
        with gauge_cols[i % 3]:
            score = device["risk_score"]
            gauge_color = SEVERITY_COLORS["Critical"] if score >= 75 else \
                          SEVERITY_COLORS["High"] if score >= 50 else \
                          SEVERITY_COLORS["Medium"] if score >= 25 else \
                          SEVERITY_COLORS["Info"]

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "%", "font": {"size": 28, "color": COLORS["text_primary"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLORS["text_secondary"]},
                    "bar": {"color": gauge_color},
                    "bgcolor": COLORS["bg_secondary"],
                    "bordercolor": COLORS["border"],
                    "steps": [
                        {"range": [0, 25], "color": "rgba(16, 185, 129, 0.08)"},
                        {"range": [25, 50], "color": "rgba(245, 158, 11, 0.08)"},
                        {"range": [50, 75], "color": "rgba(249, 115, 22, 0.08)"},
                        {"range": [75, 100], "color": "rgba(239, 68, 68, 0.08)"},
                    ],
                },
                title={"text": device["name"], "font": {"size": 12, "color": COLORS["text_secondary"]}},
            ))
            apply_plotly_theme(fig_gauge)
            fig_gauge.update_layout(height=220, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Flow Distribution by Device ──────────────────────────────────────
    section_header("📡 Flow Distribution by Device")

    fig_flows = go.Figure()
    fig_flows.add_trace(go.Bar(
        name="Benign",
        x=df_devices["name"],
        y=(df_devices["total_flows"] - df_devices["malicious_flows"]),
        marker=dict(color=COLORS["accent_green"]),
    ))
    fig_flows.add_trace(go.Bar(
        name="Malicious",
        x=df_devices["name"],
        y=df_devices["malicious_flows"],
        marker=dict(color=COLORS["accent_red"]),
    ))
    apply_plotly_theme(fig_flows)
    fig_flows.update_layout(
        barmode="stack",
        height=400,
        title="Benign vs Malicious Flows per Device",
        xaxis=dict(title="Device", tickangle=45),
        yaxis=dict(title="Flow Count"),
    )
    st.plotly_chart(fig_flows, use_container_width=True)

    # ── Full Device Table ────────────────────────────────────────────────
    section_header("📋 Device Registry")
    st.markdown("<p style='color:#94a3b8; font-size:0.85rem;'>You can double-click on a device's <b>name</b> or <b>type</b> to rename it inline.</p>", unsafe_allow_html=True)

    display_cols = ["id", "name", "type", "ip_address", "first_seen", "last_seen",
                    "total_flows", "malicious_flows", "risk_score"]
    available = [c for c in display_cols if c in df_devices.columns]

    current_df = df_devices[available].sort_values("risk_score", ascending=False).reset_index(drop=True)
    
    edited_df = st.data_editor(
        current_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        disabled=["id", "ip_address", "first_seen", "last_seen", "total_flows", "malicious_flows", "risk_score"],
    )

    # Auto-save changes
    if not edited_df.equals(current_df):
        for idx in range(len(current_df)):
            orig_row = current_df.iloc[idx]
            new_row = edited_df.iloc[idx]
            
            if orig_row["name"] != new_row["name"] or orig_row["type"] != new_row["type"]:
                db.update_device_name(new_row["id"], new_row["name"], new_row["type"])
                st.toast(f"✅ Updated device {new_row['id']}")
                # Using st.rerun() to immediately refresh the visual charts if name changed
                st.rerun()
