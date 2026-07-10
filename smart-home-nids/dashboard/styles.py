"""Dark-themed CSS and Plotly layout helpers for the NIDS dashboard.

Provides inject_css() to style the entire Streamlit app with a refined
dark theme that avoids the "AI-generated template" look — asymmetric
spacing, muted accents with selective pops, organic border radii, and
restrained gradient use.
"""

from __future__ import annotations

import streamlit as st

# ── Color Palette ────────────────────────────────────────────────────────────
# Muted, intentional tones — avoid the saturated cyan/purple "AI look."
COLORS = {
    "bg_primary": "#0b0f19",
    "bg_secondary": "#131825",
    "bg_card": "rgba(19, 24, 37, 0.92)",
    "bg_card_hover": "rgba(25, 32, 50, 0.95)",
    "border": "rgba(148, 163, 184, 0.12)",
    "border_hover": "rgba(148, 163, 184, 0.25)",
    "text_primary": "#e2e8f0",
    "text_secondary": "#7a8599",
    "text_muted": "#4a5568",
    "accent_cyan": "#38bdf8",
    "accent_purple": "#a78bfa",
    "accent_pink": "#f472b6",
    "accent_green": "#34d399",
    "accent_amber": "#fbbf24",
    "accent_red": "#f87171",
    "accent_blue": "#60a5fa",
    "gradient_start": "#38bdf8",
    "gradient_end": "#a78bfa",
    "severity_critical": "#f87171",
    "severity_high": "#fb923c",
    "severity_medium": "#fbbf24",
    "severity_low": "#38bdf8",
    "severity_info": "#34d399",
}

SEVERITY_COLORS = {
    "Critical": COLORS["severity_critical"],
    "High": COLORS["severity_high"],
    "Medium": COLORS["severity_medium"],
    "Low": COLORS["severity_low"],
    "Info": COLORS["severity_info"],
}

ATTACK_COLORS = {
    "BENIGN": "#34d399",
    "DDoS": "#f87171",
    "DoS": "#fb923c",
    "Mirai": "#ef4444",
    "Recon": "#38bdf8",
    "Spoofing": "#fbbf24",
    "BruteForce": "#a78bfa",
    "WebAttack": "#f472b6",
    "Malware": "#b91c1c",
}


def inject_css() -> None:
    """Inject the full dark-theme CSS into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Reset & Global ──────────────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: {COLORS["text_primary"]};
        }}

        .stApp {{
            background: {COLORS["bg_primary"]};
        }}

        /* ── Sidebar ─────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background: {COLORS["bg_secondary"]} !important;
            border-right: 1px solid {COLORS["border"]};
        }}

        section[data-testid="stSidebar"] .stRadio label {{
            color: {COLORS["text_secondary"]} !important;
            font-weight: 500;
            font-size: 0.9rem;
            padding: 0.45rem 0.75rem;
            border-radius: 8px;
            transition: color 0.15s ease, background 0.15s ease;
            margin-bottom: 2px;
        }}

        section[data-testid="stSidebar"] .stRadio label:hover {{
            color: {COLORS["text_primary"]} !important;
            background: rgba(148, 163, 184, 0.06);
        }}

        /* ── Metric Cards ────────────────────────────────────────────── */
        .metric-card {{
            background: {COLORS["bg_card"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            position: relative;
        }}

        .metric-card:hover {{
            border-color: {COLORS["border_hover"]};
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }}

        .metric-value {{
            font-size: 1.9rem;
            font-weight: 700;
            color: {COLORS["text_primary"]};
            margin-bottom: 0.2rem;
            line-height: 1.2;
            font-feature-settings: "tnum";
            font-variant-numeric: tabular-nums;
        }}

        .metric-label {{
            font-size: 0.78rem;
            color: {COLORS["text_secondary"]};
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }}

        .metric-delta {{
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 0.2rem;
        }}

        .metric-icon {{
            float: right;
            font-size: 1.3rem;
            opacity: 0.5;
            margin-top: -2px;
        }}

        /* ── Section Headers ─────────────────────────────────────────── */
        .section-header {{
            font-size: 1.15rem;
            font-weight: 600;
            color: {COLORS["text_primary"]};
            margin: 1.75rem 0 0.75rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid {COLORS["border"]};
            letter-spacing: -0.01em;
        }}

        /* ── Severity Badges ─────────────────────────────────────────── */
        .severity-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}

        .severity-critical {{
            background: rgba(248, 113, 113, 0.15);
            color: {COLORS["severity_critical"]};
        }}

        .severity-high {{
            background: rgba(251, 146, 60, 0.15);
            color: {COLORS["severity_high"]};
        }}

        .severity-medium {{
            background: rgba(251, 191, 36, 0.15);
            color: {COLORS["severity_medium"]};
        }}

        .severity-low {{
            background: rgba(56, 189, 248, 0.15);
            color: {COLORS["severity_low"]};
        }}

        .severity-info {{
            background: rgba(52, 211, 153, 0.15);
            color: {COLORS["severity_info"]};
        }}

        /* ── Tables ──────────────────────────────────────────────────── */
        .stDataFrame {{
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ── Buttons ─────────────────────────────────────────────────── */
        .stButton > button {{
            background: {COLORS["bg_card"]} !important;
            color: {COLORS["text_primary"]} !important;
            border: 1px solid {COLORS["border"]} !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.45rem 1.25rem !important;
            transition: border-color 0.15s ease, background 0.15s ease !important;
            font-size: 0.85rem !important;
        }}

        .stButton > button:hover {{
            border-color: {COLORS["accent_cyan"]} !important;
            background: rgba(56, 189, 248, 0.08) !important;
        }}

        .stButton > button[kind="primary"] {{
            background: {COLORS["accent_cyan"]} !important;
            color: {COLORS["bg_primary"]} !important;
            border: none !important;
            font-weight: 700 !important;
        }}

        .stButton > button[kind="primary"]:hover {{
            background: #2da8e0 !important;
        }}

        /* ── Status Container ────────────────────────────────────────── */
        .status-container {{
            background: {COLORS["bg_card"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
            margin: 0.4rem 0;
        }}

        /* ── Live Indicator ──────────────────────────────────────────── */
        @keyframes pulse {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.4; transform: scale(0.85); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}

        .live-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {COLORS["accent_green"]};
            animation: pulse 2s ease-in-out infinite;
            margin-right: 6px;
        }}

        /* ── Info / Status Cards ─────────────────────────────────────── */
        .nids-info-banner {{
            padding: 0.7rem 1rem;
            background: rgba(56, 189, 248, 0.06);
            border-left: 3px solid {COLORS["accent_cyan"]};
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
            color: {COLORS["text_secondary"]};
            margin-bottom: 1rem;
        }}

        .nids-warn-banner {{
            padding: 0.7rem 1rem;
            background: rgba(248, 113, 113, 0.06);
            border-left: 3px solid {COLORS["accent_red"]};
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
            color: {COLORS["text_secondary"]};
            margin-bottom: 1rem;
        }}

        /* ── Mono / Code Text ────────────────────────────────────────── */
        code, .stCode, .nids-mono {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        }}

        /* ── Scrollbar ───────────────────────────────────────────────── */
        ::-webkit-scrollbar {{
            width: 5px;
            height: 5px;
        }}
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {COLORS["border"]};
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS["text_muted"]};
        }}

        /* ── Tabs styling ────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
            font-weight: 500;
        }}

        /* ── Hide Streamlit chrome ───────────────────────────────────── */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str = "", delta: str = "", color: str = "") -> str:
    """Return HTML for a metric card. Color tints the value text only."""
    delta_html = ""
    if delta:
        delta_color = COLORS["accent_green"] if not delta.startswith("-") else COLORS["accent_red"]
        delta_html = f'<div class="metric-delta" style="color: {delta_color}">{delta}</div>'

    value_style = f'style="color: {color};"' if color else ""
    icon_html = f'<span class="metric-icon">{icon}</span>' if icon else ""

    return f"""
    <div class="metric-card">
        {icon_html}
        <div class="metric-label">{label}</div>
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
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"], size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=11, color=COLORS["text_secondary"]),
    ),
    xaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.06)",
        zerolinecolor="rgba(148, 163, 184, 0.1)",
        tickfont=dict(color=COLORS["text_secondary"]),
    ),
    yaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.06)",
        zerolinecolor="rgba(148, 163, 184, 0.1)",
        tickfont=dict(color=COLORS["text_secondary"]),
    ),
)


def apply_plotly_theme(fig) -> None:
    """Apply the dark NIDS theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
