"""Dark-themed CSS and Plotly layout helpers for the NIDS dashboard.

Provides inject_css() to style the entire Streamlit app with a modern
glassmorphism dark theme, plus consistent Plotly chart theming.
"""

from __future__ import annotations

import streamlit as st

# ── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg_primary": "#0e1117",
    "bg_secondary": "#1a1f2e",
    "bg_card": "rgba(26, 31, 46, 0.85)",
    "border": "rgba(99, 102, 241, 0.25)",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "accent_cyan": "#06b6d4",
    "accent_purple": "#8b5cf6",
    "accent_pink": "#ec4899",
    "accent_green": "#10b981",
    "accent_amber": "#f59e0b",
    "accent_red": "#ef4444",
    "gradient_start": "#06b6d4",
    "gradient_end": "#8b5cf6",
    "severity_critical": "#ef4444",
    "severity_high": "#f97316",
    "severity_medium": "#f59e0b",
    "severity_low": "#06b6d4",
    "severity_info": "#10b981",
}

SEVERITY_COLORS = {
    "Critical": COLORS["severity_critical"],
    "High": COLORS["severity_high"],
    "Medium": COLORS["severity_medium"],
    "Low": COLORS["severity_low"],
    "Info": COLORS["severity_info"],
}

ATTACK_COLORS = {
    "BENIGN": "#10b981",
    "DDoS": "#ef4444",
    "DoS": "#f97316",
    "Mirai": "#dc2626",
    "Recon": "#06b6d4",
    "Spoofing": "#f59e0b",
    "BruteForce": "#8b5cf6",
    "WebAttack": "#ec4899",
    "Malware": "#991b1b",
}


def inject_css() -> None:
    """Inject the full dark-theme CSS into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Global ──────────────────────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: {COLORS["text_primary"]};
        }}

        .stApp {{
            background: linear-gradient(135deg, {COLORS["bg_primary"]} 0%, #111827 50%, #0f172a 100%);
        }}

        /* ── Sidebar ─────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%) !important;
            border-right: 1px solid {COLORS["border"]};
        }}

        section[data-testid="stSidebar"] .stRadio label {{
            color: {COLORS["text_primary"]} !important;
            font-weight: 500;
            padding: 0.5rem 0;
            transition: all 0.2s ease;
        }}

        section[data-testid="stSidebar"] .stRadio label:hover {{
            color: {COLORS["accent_cyan"]} !important;
        }}

        /* ── Metric Cards ────────────────────────────────────────────── */
        .metric-card {{
            background: {COLORS["bg_card"]};
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid {COLORS["border"]};
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {COLORS["gradient_start"]}, {COLORS["gradient_end"]});
        }}

        .metric-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(6, 182, 212, 0.15);
            border-color: {COLORS["accent_cyan"]};
        }}

        .metric-value {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, {COLORS["gradient_start"]}, {COLORS["gradient_end"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
            line-height: 1.2;
        }}

        .metric-label {{
            font-size: 0.85rem;
            color: {COLORS["text_secondary"]};
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-delta {{
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }}

        /* ── Section Headers ─────────────────────────────────────────── */
        .section-header {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {COLORS["text_primary"]};
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {COLORS["border"]};
        }}

        /* ── Severity Badges ─────────────────────────────────────────── */
        .severity-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .severity-critical {{
            background: rgba(239, 68, 68, 0.2);
            color: {COLORS["severity_critical"]};
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}

        .severity-high {{
            background: rgba(249, 115, 22, 0.2);
            color: {COLORS["severity_high"]};
            border: 1px solid rgba(249, 115, 22, 0.4);
        }}

        .severity-medium {{
            background: rgba(245, 158, 11, 0.2);
            color: {COLORS["severity_medium"]};
            border: 1px solid rgba(245, 158, 11, 0.4);
        }}

        .severity-low {{
            background: rgba(6, 182, 212, 0.2);
            color: {COLORS["severity_low"]};
            border: 1px solid rgba(6, 182, 212, 0.4);
        }}

        .severity-info {{
            background: rgba(16, 185, 129, 0.2);
            color: {COLORS["severity_info"]};
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}

        /* ── Tables ──────────────────────────────────────────────────── */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
        }}

        /* ── Buttons ─────────────────────────────────────────────────── */
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS["gradient_start"]}, {COLORS["gradient_end"]}) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(6, 182, 212, 0.3) !important;
        }}

        /* ── Status Container ────────────────────────────────────────── */
        .status-container {{
            background: {COLORS["bg_card"]};
            backdrop-filter: blur(16px);
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 0.5rem 0;
        }}

        /* ── Pulse Animation ─────────────────────────────────────────── */
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}

        .live-indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: {COLORS["accent_green"]};
            animation: pulse 2s ease-in-out infinite;
            margin-right: 8px;
        }}

        /* ── Progress/Gauge ──────────────────────────────────────────── */
        .gauge-container {{
            text-align: center;
            padding: 1rem;
        }}

        /* ── Scrollbar ───────────────────────────────────────────────── */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {COLORS["bg_primary"]};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {COLORS["border"]};
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS["accent_cyan"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str = "", delta: str = "", color: str = "") -> str:
    """Return HTML for a glassmorphism metric card."""
    delta_html = ""
    if delta:
        delta_color = COLORS["accent_green"] if not delta.startswith("-") else COLORS["accent_red"]
        delta_html = f'<div class="metric-delta" style="color: {delta_color}">{delta}</div>'

    value_style = f'style="background: linear-gradient(135deg, {color}, {COLORS["gradient_end"]}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;"' if color else ""

    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value" {value_style}>{value}</div>
        {delta_html}
    </div>
    """


def severity_badge(severity: str) -> str:
    """Return HTML for a colored severity badge."""
    css_class = f"severity-{severity.lower()}"
    return f'<span class="severity-badge {css_class}">{severity}</span>'


def section_header(title: str) -> None:
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ── Plotly Layout ──────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=COLORS["border"],
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="rgba(99, 102, 241, 0.1)",
        zerolinecolor="rgba(99, 102, 241, 0.2)",
    ),
    yaxis=dict(
        gridcolor="rgba(99, 102, 241, 0.1)",
        zerolinecolor="rgba(99, 102, 241, 0.2)",
    ),
)


def apply_plotly_theme(fig) -> None:
    """Apply the dark NIDS theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
