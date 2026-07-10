"""Smart Home NIDS — Streamlit Dashboard Entry Point.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.styles import COLORS, inject_css

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Custom CSS ────────────────────────────────────────────────────────
inject_css()


# ── Initialise Shared Resources (cached) ─────────────────────────────────────

@st.cache_resource
def get_database():
    """Initialise the SQLite database (shared across sessions)."""
    from src.database import NIDSDatabase
    return NIDSDatabase()


@st.cache_resource
def get_predictor():
    """Load the NIDS predictor (shared across sessions)."""
    from src.predict import NIDSPredictor
    predictor = NIDSPredictor()
    try:
        predictor.load()
    except FileNotFoundError:
        return None
    return predictor


@st.cache_resource
def get_alert_manager():
    """Initialise the alert manager."""
    from src.alerts import AlertManager
    db = get_database()
    return AlertManager(db=db)


# ── Sidebar Navigation ──────────────────────────────────────────────────────

def render_sidebar() -> str:
    """Render the sidebar and return the selected page."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 1.25rem 0.5rem 0.75rem 0.5rem;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:1.6rem;">🛡️</span>
                    <div>
                        <div style="font-size:1.1rem; font-weight:700; color:{COLORS['text_primary']};
                                    letter-spacing:-0.02em;">
                            Smart Home NIDS
                        </div>
                        <div style="font-size:0.7rem; color:{COLORS['text_muted']}; margin-top:1px;">
                            Network Intrusion Detection
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "📊 Dashboard",
                "🔴 Live Monitor",
                "📈 Attack Analysis",
                "🖥️ Device Stats",
                "📋 History & Logs",
                "⚙️ Settings",
            ],
            key="nav_radio",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Model status indicator
        predictor = get_predictor()
        if predictor is not None:
            st.markdown(
                f"""
                <div style="padding:0.7rem 0.85rem; background:{COLORS['bg_card']};
                     border-radius:8px; border:1px solid {COLORS['border']};">
                    <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                        <span class="live-indicator"></span>
                        <span style="font-size:0.75rem; color:{COLORS['accent_green']}; font-weight:600;">
                            Model Online
                        </span>
                    </div>
                    <div style="font-size:0.68rem; color:{COLORS['text_muted']}; font-family:'JetBrains Mono',monospace;">
                        {predictor.model_version}
                    </div>
                    <div style="font-size:0.65rem; color:{COLORS['text_muted']}; margin-top:2px;">
                        {len(predictor.feature_names)} features · {len(predictor.class_names)} classes
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="padding:0.7rem 0.85rem; background:{COLORS['bg_card']};
                     border-radius:8px; border:1px solid rgba(248,113,113,0.2);">
                    <div style="display:flex; align-items:center; gap:6px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:{COLORS['accent_red']};display:inline-block;"></span>
                        <span style="font-size:0.75rem; color:{COLORS['accent_red']}; font-weight:600;">
                            Model Not Trained
                        </span>
                    </div>
                    <div style="font-size:0.65rem; color:{COLORS['text_muted']}; margin-top:4px;">
                        Go to ⚙️ Settings to train
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""<div style="position:fixed; bottom:12px; left:12px;">
                <span style="font-size:0.6rem; color:{COLORS['text_muted']};">
                    NIDS v1.0
                </span>
            </div>""",
            unsafe_allow_html=True,
        )

    return page


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main application entry point."""
    db = get_database()
    predictor = get_predictor()
    alert_mgr = get_alert_manager()

    page = render_sidebar()

    if page == "📊 Dashboard":
        from dashboard.home import render
        render(db)

    elif page == "🔴 Live Monitor":
        if predictor is None:
            st.error("⚠️ Model not trained. Go to **⚙️ Settings** to train the model first.")
            st.info("Train the model to enable live monitoring and predictions.")
        else:
            from dashboard.monitoring import render
            render(db, predictor, alert_mgr)

    elif page == "📈 Attack Analysis":
        from dashboard.analysis import render
        render()

    elif page == "🖥️ Device Stats":
        from dashboard.devices import render
        render(db)

    elif page == "📋 History & Logs":
        from dashboard.logs import render
        render(db)

    elif page == "⚙️ Settings":
        from dashboard.settings_page import render
        render(db)


if __name__ == "__main__":
    main()
