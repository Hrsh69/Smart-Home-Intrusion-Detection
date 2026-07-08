"""⚙️ Settings — Model training, alert config, database management."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from dashboard.styles import COLORS, metric_card, section_header
from src.database import NIDSDatabase


def render(db: NIDSDatabase) -> None:
    """Render the settings page."""

    st.markdown(
        f"""
        <h2 style="font-weight:700; color:{COLORS['text_primary']}; margin-bottom:1.5rem;">
            ⚙️ Settings & Configuration
        </h2>
        """,
        unsafe_allow_html=True,
    )

    tab_model, tab_alerts, tab_db, tab_about = st.tabs([
        "🤖 Model Training",
        "🔔 Alert Configuration",
        "🗄️ Database Management",
        "ℹ️ About",
    ])

    with tab_model:
        _render_model_training()

    with tab_alerts:
        _render_alert_config()

    with tab_db:
        _render_database_management(db)

    with tab_about:
        _render_about()


def _render_model_training() -> None:
    """Model training controls."""
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "rf_model.pkl"
    report_path = project_root / "reports" / "training_report.json"

    section_header("🏋️ Model Training")

    # Current model status
    if model_path.exists():
        st.success("✅ Trained model found: `rf_model.pkl`")
        if report_path.exists():
            with open(report_path, "r") as f:
                report = json.load(f)
            cols = st.columns(3)
            with cols[0]:
                st.metric("Accuracy", f"{report.get('accuracy', 0):.2%}")
            with cols[1]:
                st.metric("F1 (weighted)", f"{report.get('f1_weighted', 0):.2%}")
            with cols[2]:
                st.metric("Training Time", f"{report.get('training_time_seconds', 0):.1f}s")
    else:
        st.warning("⚠️ No trained model found. Train one below.")

    st.markdown("---")

    # Training controls
    st.markdown(f"**Train/Re-train the Random Forest model:**")
    st.markdown(
        f"<p style='color:{COLORS['text_secondary']}; font-size:0.9rem;'>"
        f"This will train a new Random Forest classifier using the preprocessed dataset "
        f"with balanced class weights. The process may take 1-3 minutes.</p>",
        unsafe_allow_html=True,
    )

    if st.button("🚀 Start Training", key="train_model_btn"):
        with st.spinner("Training Random Forest model..."):
            try:
                import sys
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))

                from src.train_model import run_training

                metrics = run_training()

                st.success("✅ Model trained successfully!")
                st.balloons()

                cols = st.columns(4)
                with cols[0]:
                    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                with cols[1]:
                    st.metric("F1 (weighted)", f"{metrics['f1_weighted']:.2%}")
                with cols[2]:
                    st.metric("F1 (macro)", f"{metrics['f1_macro']:.2%}")
                with cols[3]:
                    train_time = metrics.get("training_time_seconds", 0)
                    st.metric("Time", f"{train_time:.1f}s")

                # Show confusion matrix plot
                cm_path = project_root / "plots" / "confusion_matrix.png"
                if cm_path.exists():
                    st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)

            except Exception as exc:
                st.error(f"Training failed: {exc}")
                import traceback
                st.code(traceback.format_exc())


def _render_alert_config() -> None:
    """Alert channel configuration display."""
    section_header("🔔 Alert Channels")

    st.markdown(
        f"<p style='color:{COLORS['text_secondary']}; font-size:0.9rem;'>"
        f"Configure alert channels via the <code>.env</code> file. "
        f"Copy <code>.env.example</code> to <code>.env</code> and fill in your values.</p>",
        unsafe_allow_html=True,
    )

    from config.settings import SETTINGS

    # Status display
    channels = [
        ("Desktop Notifications", SETTINGS.DESKTOP_ALERTS_ENABLED, "🖥️",
         "Enabled by default via `plyer`. No config needed."),
        ("Email Alerts (SMTP)", SETTINGS.EMAIL_ALERTS_ENABLED, "📧",
         f"Host: `{SETTINGS.SMTP_HOST}:{SETTINGS.SMTP_PORT}` → `{SETTINGS.ALERT_EMAIL_TO or 'Not set'}`"),
        ("Telegram Webhook", SETTINGS.TELEGRAM_ALERTS_ENABLED, "📱",
         f"Bot Token: `{'✅ Set' if SETTINGS.TELEGRAM_BOT_TOKEN else '❌ Not set'}`"),
    ]

    for name, enabled, icon, detail in channels:
        status = "🟢 Enabled" if enabled else "🔴 Disabled"
        st.markdown(
            f"""
            <div class="status-container" style="margin-bottom:0.8rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:{COLORS['text_primary']}">{icon} {name}</span>
                    <span style="font-size:0.85rem;">{status}</span>
                </div>
                <p style="color:{COLORS['text_secondary']}; font-size:0.8rem; margin:0.25rem 0 0 0;">{detail}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(f"**Severity Routing:**")
    routing = {
        "Critical": "Desktop + Email + Telegram",
        "High": "Desktop + Email",
        "Medium": "Desktop only",
        "Low": "Log only",
        "Info": "No alert",
    }
    for sev, channels_str in routing.items():
        st.markdown(f"- **{sev}** → {channels_str}")


def _render_database_management(db: NIDSDatabase) -> None:
    """Database management controls."""
    section_header("🗄️ Database Management")

    stats = db.get_dashboard_stats()

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Detections", f"{stats['total_flows']:,}")
    with cols[1]:
        st.metric("Registered Devices", f"{stats['device_count']:,}")
    with cols[2]:
        st.metric("Pending Alerts", f"{stats['pending_alerts']:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Export Data:**")
        if st.button("📥 Export All Detections to CSV", key="export_all_csv"):
            from datetime import datetime
            from pathlib import Path

            export_path = Path(__file__).resolve().parents[1] / "reports" / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            count = db.export_detections_csv(str(export_path))
            if count > 0:
                st.success(f"Exported {count:,} records to `{export_path.name}`")
            else:
                st.info("No records to export.")

    with col2:
        st.markdown("**Danger Zone:**")
        if st.button("🗑️ Clear All Data", key="clear_db_btn", type="secondary"):
            st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.warning("⚠️ This will delete ALL detections, devices, alerts, and statistics.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, Clear Everything", key="confirm_clear_yes"):
                    db.clear_all()
                    st.session_state["confirm_clear"] = False
                    st.success("Database cleared.")
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", key="confirm_clear_no"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()


def _render_about() -> None:
    """About section with project info."""
    section_header("ℹ️ About Smart Home NIDS")

    st.markdown(
        f"""
        <div class="status-container">
            <h3 style="color:{COLORS['accent_cyan']}; margin-top:0;">Smart Home Network Intrusion Detection System</h3>
            <p style="color:{COLORS['text_secondary']};">
                A machine-learning-based intrusion detection system designed to detect malicious
                network traffic generated by IoT devices in smart home environments.
            </p>
            <hr style="border-color:{COLORS['border']}">
            <p style="color:{COLORS['text_secondary']}; font-size:0.85rem;">
                <strong>Dataset:</strong> CIC IoT-2023 (Canadian Institute for Cybersecurity)<br>
                <strong>Model:</strong> Random Forest with balanced class weights<br>
                <strong>Features:</strong> 18 network flow features<br>
                <strong>Classes:</strong> BENIGN, DDoS, DoS, Mirai, Recon, Spoofing, BruteForce, WebAttack, Malware<br>
                <strong>Scaler:</strong> RobustScaler (handles outlier-heavy traffic data)<br>
            </p>
            <hr style="border-color:{COLORS['border']}">
            <p style="color:{COLORS['text_secondary']}; font-size:0.8rem;">
                Built as a B.Tech Cybersecurity final-year project.<br>
                Stack: Python · scikit-learn · Streamlit · SQLite · Plotly · SHAP
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
