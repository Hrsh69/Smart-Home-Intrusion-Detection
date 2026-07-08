from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def plot_class_distribution(y: pd.Series, out_path: Path, title: str) -> None:
    vc = y.value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    vc.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance_bar(df: pd.DataFrame, value_col: str, out_path: Path, title: str, top_k: int) -> None:
    top = df.head(top_k).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path, title: str, max_features: int = 40) -> None:
    # Keep the plot readable by limiting features
    cols = df.columns[:max_features]
    corr = df[cols].corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, interpolation="nearest", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    plt.yticks(range(len(cols)), cols, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_preprocessing_report_pdf(
    out_pdf: Path,
    summary: Dict[str, Any],
    cleaning_report: Dict[str, Any],
    feature_selection_summary: Dict[str, Any],
    balancing_summary: Dict[str, Any],
    notes: Optional[list[str]] = None,
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    notes = notes or []

    with PdfPages(out_pdf) as pdf:
        # Page 1: Overview
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.suptitle("Smart Home NIDS — Preprocessing Report (CIC IoT-2023)", fontsize=16, y=0.98)
        ax = fig.add_subplot(111)
        ax.axis("off")

        text = []
        text.append("Summary")
        for k, v in summary.items():
            text.append(f"- {k}: {v}")
        text.append("")
        text.append("Cleaning report")
        for k, v in cleaning_report.items():
            if k in ("missing_values_before", "missing_values_after"):
                continue
            text.append(f"- {k}: {v}")
        text.append("")
        text.append("Feature selection")
        for k, v in feature_selection_summary.items():
            text.append(f"- {k}: {v}")
        text.append("")
        text.append("Balancing")
        for k, v in balancing_summary.items():
            text.append(f"- {k}: {v}")
        if notes:
            text.append("")
            text.append("Notes")
            for n in notes:
                text.append(f"- {n}")

        ax.text(0.02, 0.98, "\n".join(text), va="top", fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


def build_preprocessing_report_html(
    out_html: Path,
    summary: Dict[str, Any],
    cleaning_report: Dict[str, Any],
    feature_selection_summary: Dict[str, Any],
    notes: Optional[list[str]] = None,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    notes = notes or []

    def _dict_to_list(title: str, data: Dict[str, Any]) -> str:
        items = "\n".join(f"<li><strong>{k}</strong>: {v}</li>" for k, v in data.items())
        return f"<h2>{title}</h2><ul>{items}</ul>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Smart Home NIDS Preprocessing Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }}
    h1, h2 {{ color: #1f3b5b; }}
    code {{ background: #f3f4f6; padding: 0.1rem 0.3rem; }}
    ul {{ margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <h1>Smart Home NIDS Preprocessing Report</h1>
  {_dict_to_list("Summary", summary)}
  {_dict_to_list("Cleaning Report", cleaning_report)}
  {_dict_to_list("Feature Selection", feature_selection_summary)}
  <h2>Notes</h2>
  <ul>{"".join(f"<li>{note}</li>" for note in notes)}</ul>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")

