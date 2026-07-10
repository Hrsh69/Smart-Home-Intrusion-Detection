"""📊 Dashboard Overview — KPI metrics, attack distribution, threat timeline."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from dashboard.styles import (
    ATTACK_COLORS,
    COLORS,
    SEVERITY_COLORS,
    apply_plotly_theme,
    metric_card,
    section_header,
)
from src.database import NIDSDatabase


def render(db: NIDSDatabase) -> None:
    """Render the dashboard overview page."""

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="padding: 0.25rem 0 1.5rem 0;">
            <h1 style="font-size:1.75rem; font-weight:700; color:{COLORS['text_primary']};
                margin:0 0 0.3rem 0; letter-spacing:-0.02em;">
                Dashboard
            </h1>
            <p style="color:{COLORS['text_muted']}; font-size:0.85rem; margin:0;">
                Network intrusion detection overview
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = db.get_dashboard_stats()

    # ── KPI Cards ────────────────────────────────────────────────────────
    cols = st.columns(5)
    kpis = [
        ("Total Flows", f"{stats['total_flows']:,}", "📡", COLORS["accent_cyan"]),
        ("Threats", f"{stats['threat_flows']:,}", "⚠️", COLORS["accent_red"]),
        ("Benign", f"{stats['benign_flows']:,}", "✓", COLORS["accent_green"]),
        ("Devices", f"{stats['device_count']:,}", "◉", COLORS["accent_purple"]),
        ("Alerts", f"{stats['pending_alerts']:,}", "●", COLORS["accent_amber"]),
    ]

    for col, (label, value, icon, color) in zip(cols, kpis):
        with col:
            st.markdown(metric_card(label, value, icon, color=color), unsafe_allow_html=True)

    # ── Detection Rate Gauge + Attack Pie ────────────────────────────────
    col_gauge, col_pie = st.columns([1, 1])

    with col_gauge:
        section_header("Threat Detection Rate")

        rate = stats["detection_rate"]
        gauge_color = COLORS["accent_green"]
        if rate > 50:
            gauge_color = COLORS["accent_red"]
        elif rate > 25:
            gauge_color = COLORS["accent_amber"]

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rate,
                number={"suffix": "%", "font": {"size": 42, "color": COLORS["text_primary"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLORS["text_muted"]},
                    "bar": {"color": gauge_color, "thickness": 0.7},
                    "bgcolor": COLORS["bg_secondary"],
                    "bordercolor": COLORS["border"],
                    "steps": [
                        {"range": [0, 25], "color": "rgba(52, 211, 153, 0.06)"},
                        {"range": [25, 50], "color": "rgba(251, 191, 36, 0.06)"},
                        {"range": [50, 100], "color": "rgba(248, 113, 113, 0.06)"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["accent_red"], "width": 2},
                        "thickness": 0.75,
                        "value": 75,
                    },
                },
                title={"text": "Malicious Traffic", "font": {"color": COLORS["text_secondary"], "size": 13}},
            )
        )
        apply_plotly_theme(fig_gauge)
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Attack Distribution Pie ──────────────────────────────────────────
    with col_pie:
        section_header("Attack Distribution")
        attack_dist = db.get_attack_distribution()

        if attack_dist:
            labels = list(attack_dist.keys())
            values = list(attack_dist.values())
            colors = [ATTACK_COLORS.get(l, COLORS["text_secondary"]) for l in labels]

            fig_pie = go.Figure(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors, line=dict(color=COLORS["bg_primary"], width=1.5)),
                    textinfo="label+percent",
                    textfont=dict(size=11, color=COLORS["text_secondary"]),
                    hole=0.5,
                    hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
                )
            )
            apply_plotly_theme(fig_pie)
            fig_pie.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No detection data yet. Run the Live Monitor to generate predictions.")

    # ── Threat Timeline ──────────────────────────────────────────────────
    section_header("Threat Timeline — Last 24h")
    timeline = db.get_hourly_totals(hours=24)

    if timeline:
        hours = [t["hour"] for t in timeline]
        totals = [t["total"] for t in timeline]
        threats = [t["threats"] for t in timeline]

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=hours, y=totals, name="Total",
            mode="lines",
            line=dict(color=COLORS["accent_cyan"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.05)",
        ))
        fig_timeline.add_trace(go.Scatter(
            x=hours, y=threats, name="Threats",
            mode="lines",
            line=dict(color=COLORS["accent_red"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248, 113, 113, 0.05)",
        ))
        apply_plotly_theme(fig_timeline)
        fig_timeline.update_layout(
            height=280,
            xaxis_title="",
            yaxis_title="",
            hovermode="x unified",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No timeline data yet.")

    # ── Severity Breakdown ───────────────────────────────────────────────
    col_sev, col_model = st.columns([1, 1])

    with col_sev:
        section_header("Severity Breakdown")
        severity_dist = db.get_severity_distribution()

        if severity_dist:
            ordered = ["Critical", "High", "Medium", "Low", "Info"]
            labels = [s for s in ordered if s in severity_dist]
            values = [severity_dist[s] for s in labels]
            colors = [SEVERITY_COLORS.get(s, COLORS["text_secondary"]) for s in labels]

            fig_sev = go.Figure(go.Bar(
                x=labels, y=values,
                marker=dict(color=colors),
                text=values, textposition="outside",
                textfont=dict(color=COLORS["text_secondary"], size=11),
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
            ))
            apply_plotly_theme(fig_sev)
            fig_sev.update_layout(height=280, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_sev, use_container_width=True)

    with col_model:
        _show_model_info()

    # ── Recent Detections Table ──────────────────────────────────────────
    section_header("Recent Detections")
    recent = db.get_recent_detections(limit=15)

    if recent:
        import pandas as pd

        df = pd.DataFrame(recent)
        display_cols = ["timestamp", "prediction", "confidence", "severity", "source_ip", "device_id"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[available],
            use_container_width=True,
            hide_index=True,
            height=380,
        )
    else:
        st.info("No detections logged yet. Go to **🔴 Live Monitor** to start scanning.")


def _show_model_info() -> None:
    """Show model training report if available."""
    report_path = Path(__file__).resolve().parents[1] / "reports" / "training_report.json"
    if not report_path.exists():
        return

    section_header("Model Performance")

    with open(report_path, "r") as f:
        report = json.load(f)

    cols = st.columns(2)
    metrics = [
        ("Accuracy", f"{report.get('accuracy', 0):.2%}", "🎯", COLORS["accent_green"]),
        ("F1 Score", f"{report.get('f1_weighted', 0):.2%}", "📊", COLORS["accent_cyan"]),
        ("ROC AUC", f"{report.get('roc_auc_weighted', 0):.2%}" if report.get("roc_auc_weighted") else "N/A", "📈", COLORS["accent_purple"]),
        ("OOB", f"{report.get('oob_score', 0):.2%}" if report.get("oob_score") else "N/A", "🌲", COLORS["accent_amber"]),
    ]

    for i, (label, value, icon, color) in enumerate(metrics):
        with cols[i % 2]:
            st.markdown(
                metric_card(label, value, icon, color=color),
                unsafe_allow_html=True,
            )
