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
            <div style="text-align:center; padding:1.5rem 0;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">🛡️</div>
                <h2 style="margin:0; font-size:1.3rem; font-weight:800;
                    background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">
                    Smart Home NIDS
                </h2>
                <p style="color:{COLORS['text_secondary']}; font-size:0.75rem; margin:0.25rem 0 0 0;">
                    Network Intrusion Detection
                </p>
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
                <div style="padding:0.8rem; background:{COLORS['bg_card']};
                     border-radius:10px; border:1px solid {COLORS['border']};">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;">
                        <span class="live-indicator"></span>
                        <span style="font-size:0.8rem; color:{COLORS['accent_green']}; font-weight:600;">
                            Model Online
                        </span>
                    </div>
                    <p style="font-size:0.7rem; color:{COLORS['text_secondary']}; margin:0;">
                        RF Classifier • {len(predictor.feature_names)} features • {len(predictor.class_names)} classes
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="padding:0.8rem; background:{COLORS['bg_card']};
                     border-radius:10px; border:1px solid rgba(239,68,68,0.3);">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="width:10px;height:10px;border-radius:50%;background:{COLORS['accent_red']};display:inline-block;"></span>
                        <span style="font-size:0.8rem; color:{COLORS['accent_red']}; font-weight:600;">
                            Model Not Trained
                        </span>
                    </div>
                    <p style="font-size:0.7rem; color:{COLORS['text_secondary']}; margin:0.25rem 0 0 0;">
                        Go to ⚙️ Settings to train
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            f"<p style='text-align:center; color:{COLORS['text_secondary']}; font-size:0.65rem;'>"
            f"Smart Home NIDS v1.0<br>B.Tech Cybersecurity Project</p>",
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
