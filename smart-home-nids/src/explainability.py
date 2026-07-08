"""SHAP-based model explainability for the NIDS.

Provides on-demand local and global explanations using SHAP TreeExplainer
(optimised for Random Forest). Computed lazily since SHAP is expensive.

Usage:
    from src.explainability import NIDSExplainer
    explainer = NIDSExplainer(model, feature_names, class_names)
    fig = explainer.explain_single(X_row, predicted_class_idx)
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("smart_home_nids.explainability")


class NIDSExplainer:
    """SHAP explainer wrapper for the NIDS Random Forest model."""

    def __init__(
        self,
        model,
        feature_names: list[str],
        class_names: list[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self._explainer = None

    def _get_explainer(self):
        """Lazily initialise the SHAP TreeExplainer."""
        if self._explainer is None:
            try:
                import shap
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP TreeExplainer initialised.")
            except ImportError:
                logger.warning("shap not installed — explainability unavailable.")
                raise ImportError(
                    "SHAP is required for explainability. Install with: pip install shap"
                )
        return self._explainer

    def explain_single(
        self,
        X_row: pd.DataFrame | np.ndarray,
        predicted_class_idx: int,
        max_display: int = 12,
    ) -> Optional[plt.Figure]:
        """Generate a SHAP waterfall plot for a single prediction.

        Args:
            X_row: Single-row DataFrame or 1-D array of feature values.
            predicted_class_idx: Index of the predicted class (for multi-output).
            max_display: Maximum features to show.

        Returns:
            Matplotlib Figure, or None if SHAP is unavailable.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not available.")
            return None

        explainer = self._get_explainer()

        if isinstance(X_row, pd.DataFrame):
            X_arr = X_row.values
        else:
            X_arr = np.array(X_row).reshape(1, -1)

        shap_values = explainer.shap_values(X_arr)

        # For multi-class RF, shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class_idx][0]
            base = explainer.expected_value[predicted_class_idx]
        else:
            sv = shap_values[0]
            base = explainer.expected_value

        explanation = shap.Explanation(
            values=sv,
            base_values=base,
            data=X_arr[0],
            feature_names=self.feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.sca(ax)
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        predicted_class = self.class_names[predicted_class_idx]
        ax.set_title(f"SHAP Explanation — Predicted: {predicted_class}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    def explain_global(
        self,
        X_sample: pd.DataFrame,
        max_display: int = 15,
    ) -> Optional[plt.Figure]:
        """Generate a SHAP summary (beeswarm) plot for global feature importance.

        Args:
            X_sample: A sample of test data (100–500 rows recommended).
            max_display: Max features to display.

        Returns:
            Matplotlib Figure, or None if SHAP is unavailable.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not available.")
            return None

        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X_sample.values)

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)

        # For multi-class, shap_values is list — use mean absolute
        if isinstance(shap_values, list):
            # Stack all classes and take mean |SHAP|
            stacked = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap.summary_plot(
                stacked,
                X_sample,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False,
                plot_type="bar",
            )
        else:
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False,
            )

        ax.set_title("SHAP Global Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def get_feature_contributions(
        self,
        X_row: pd.DataFrame | np.ndarray,
        predicted_class_idx: int,
    ) -> dict[str, float]:
        """Return feature → SHAP value mapping for a single prediction.

        Useful for displaying in dashboard tables without a plot.
        """
        explainer = self._get_explainer()

        if isinstance(X_row, pd.DataFrame):
            X_arr = X_row.values
        else:
            X_arr = np.array(X_row).reshape(1, -1)

        shap_values = explainer.shap_values(X_arr)

        if isinstance(shap_values, list):
            sv = shap_values[predicted_class_idx][0]
        else:
            sv = shap_values[0]

        return dict(zip(self.feature_names, [round(float(v), 4) for v in sv]))
