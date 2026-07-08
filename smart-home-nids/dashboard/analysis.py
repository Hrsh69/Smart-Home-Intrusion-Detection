"""📈 Attack Analysis — Feature importance, confusion matrix, SHAP, metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from dashboard.styles import (
    ATTACK_COLORS,
    COLORS,
    apply_plotly_theme,
    metric_card,
    section_header,
)


def render() -> None:
    """Render the attack analysis page."""

    st.markdown(
        f"""
        <h2 style="font-weight:700; color:{COLORS['text_primary']}; margin-bottom:1.5rem;">
            📈 Attack Analysis & Model Insights
        </h2>
        """,
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).resolve().parents[1]
    report_path = project_root / "reports" / "training_report.json"

    if not report_path.exists():
        st.warning("⚠️ No training report found. Please train a model first via **⚙️ Settings**.")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    # ── Model Performance Summary ────────────────────────────────────────
    section_header("🎯 Model Performance")

    cols = st.columns(5)
    perf_metrics = [
        ("Accuracy", report.get("accuracy", 0), "🎯"),
        ("F1 (weighted)", report.get("f1_weighted", 0), "📊"),
        ("F1 (macro)", report.get("f1_macro", 0), "📉"),
        ("Precision", report.get("precision_weighted", 0), "🔬"),
        ("Recall", report.get("recall_weighted", 0), "🔎"),
    ]
    for col, (label, val, icon) in zip(cols, perf_metrics):
        with col:
            st.markdown(
                metric_card(label, f"{val:.2%}", icon, color=COLORS["accent_green"]),
                unsafe_allow_html=True,
            )

    # ── Per-Class Metrics Table ──────────────────────────────────────────
    section_header("📋 Per-Class Metrics")

    class_report = report.get("per_class_report", {})
    class_names = report.get("class_names", [])

    if class_report and class_names:
        rows = []
        for cls in class_names:
            if cls in class_report:
                r = class_report[cls]
                rows.append({
                    "Class": cls,
                    "Precision": round(r.get("precision", 0), 4),
                    "Recall": round(r.get("recall", 0), 4),
                    "F1-Score": round(r.get("f1-score", 0), 4),
                    "Support": int(r.get("support", 0)),
                })

        df_metrics = pd.DataFrame(rows)

        # Horizontal bar chart of F1 scores
        fig_f1 = go.Figure(go.Bar(
            y=df_metrics["Class"],
            x=df_metrics["F1-Score"],
            orientation="h",
            marker=dict(
                color=[ATTACK_COLORS.get(c, COLORS["accent_cyan"]) for c in df_metrics["Class"]],
            ),
            text=[f"{v:.2%}" for v in df_metrics["F1-Score"]],
            textposition="outside",
            textfont=dict(color=COLORS["text_primary"], size=11),
        ))
        apply_plotly_theme(fig_f1)
        fig_f1.update_layout(
            height=400,
            title="F1-Score by Attack Category",
            xaxis=dict(title="F1-Score", range=[0, 1.1]),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # ── Confusion Matrix ─────────────────────────────────────────────────
    section_header("🔢 Confusion Matrix")

    cm = report.get("confusion_matrix")
    if cm and class_names:
        cm_array = np.array(cm)

        # Normalise row-wise for percentage display
        cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        col_raw, col_norm = st.columns(2)

        with col_raw:
            st.markdown(f"**Absolute Counts**")
            fig_cm = go.Figure(go.Heatmap(
                z=cm_array,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                text=[[str(v) for v in row] for row in cm_array],
                texttemplate="%{text}",
                textfont=dict(size=10),
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            apply_plotly_theme(fig_cm)
            fig_cm.update_layout(
                height=450,
                xaxis=dict(title="Predicted", tickangle=45),
                yaxis=dict(title="True", autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_norm:
            st.markdown(f"**Normalised (%)**")
            fig_norm = go.Figure(go.Heatmap(
                z=cm_norm,
                x=class_names,
                y=class_names,
                colorscale="Viridis",
                text=[[f"{v:.1%}" for v in row] for row in cm_norm],
                texttemplate="%{text}",
                textfont=dict(size=10),
                zmin=0, zmax=1,
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Rate: %{z:.1%}<extra></extra>",
            ))
            apply_plotly_theme(fig_norm)
            fig_norm.update_layout(
                height=450,
                xaxis=dict(title="Predicted", tickangle=45),
                yaxis=dict(title="True", autorange="reversed"),
            )
            st.plotly_chart(fig_norm, use_container_width=True)

    # ── Feature Importance ───────────────────────────────────────────────
    section_header("🌲 Feature Importance (Random Forest)")

    feat_imp = report.get("feature_importance", {})
    if feat_imp:
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_feats)

        fig_imp = go.Figure(go.Bar(
            y=list(reversed(names)),
            x=list(reversed(values)),
            orientation="h",
            marker=dict(
                color=list(reversed(values)),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Importance"),
            ),
            text=[f"{v:.4f}" for v in reversed(values)],
            textposition="outside",
            textfont=dict(color=COLORS["text_primary"], size=10),
        ))
        apply_plotly_theme(fig_imp)
        fig_imp.update_layout(
            height=500,
            title="Feature Importance Ranking",
            xaxis=dict(title="Gini Importance"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Correlation Heatmap ──────────────────────────────────────────────
    section_header("🔗 Feature Correlation Heatmap")
    _render_correlation_heatmap(project_root)

    # ── SHAP Explainability ──────────────────────────────────────────────
    section_header("🔬 SHAP Explainability (Optional)")
    _render_shap_section(project_root)

    # ── Training Info ────────────────────────────────────────────────────
    section_header("ℹ️ Training Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", f"{report.get('train_samples', 'N/A'):,}")
        st.metric("Test Samples", f"{report.get('test_samples', 'N/A'):,}")
    with col2:
        st.metric("Features", report.get("n_features", "N/A"))
        st.metric("Estimators", report.get("n_estimators", "N/A"))
    with col3:
        train_time = report.get("training_time_seconds", 0)
        st.metric("Training Time", f"{train_time:.1f}s" if train_time else "N/A")
        st.metric("OOB Score", f"{report.get('oob_score', 0):.2%}" if report.get("oob_score") else "N/A")


def _render_correlation_heatmap(project_root: Path) -> None:
    """Render correlation heatmap from processed test data."""
    test_path = project_root / "data" / "processed" / "processed_test.csv"
    if not test_path.exists():
        st.info("Test data not available for correlation analysis.")
        return

    try:
        df = pd.read_csv(test_path)
        feature_cols = [c for c in df.columns if c != "attack_category"]

        if len(feature_cols) > 2:
            corr = df[feature_cols].corr()

            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=feature_cols,
                y=feature_cols,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=8),
                hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
            ))
            apply_plotly_theme(fig)
            fig.update_layout(
                height=550,
                xaxis=dict(tickangle=45),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Error generating correlation heatmap: {exc}")


def _render_shap_section(project_root: Path) -> None:
    """Render SHAP explanations on demand."""
    try:
        import shap  # noqa: F401
        shap_available = True
    except ImportError:
        shap_available = False

    if not shap_available:
        st.info("💡 Install `shap` for model explainability: `pip install shap`")
        return

    st.markdown(
        f"<p style='color:{COLORS['text_secondary']}'>SHAP explanations show how each feature contributed to a prediction.</p>",
        unsafe_allow_html=True,
    )

    if st.button("🔬 Generate Global SHAP Explanation", key="shap_global"):
        with st.spinner("Computing SHAP values (this may take 1-2 minutes)..."):
            try:
                import joblib
                from src.explainability import NIDSExplainer

                model_path = project_root / "models" / "rf_model.pkl"
                le_path = project_root / "models" / "label_encoder.pkl"
                features_path = project_root / "models" / "selected_features.pkl"
                test_path = project_root / "data" / "processed" / "processed_test.csv"

                model = joblib.load(model_path)
                label_encoder = joblib.load(le_path)
                feature_names = joblib.load(features_path)

                test_df = pd.read_csv(test_path)
                feature_cols = [c for c in test_df.columns if c != "attack_category"]
                X_sample = test_df[feature_cols].sample(n=min(200, len(test_df)), random_state=42)

                explainer = NIDSExplainer(model, feature_names, list(label_encoder.classes_))
                fig = explainer.explain_global(X_sample, max_display=15)

                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.warning("SHAP explanation could not be generated.")

            except Exception as exc:
                st.error(f"SHAP error: {exc}")
