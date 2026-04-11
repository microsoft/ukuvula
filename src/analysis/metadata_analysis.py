#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Generate analytical figures, tables, and summary insights from the audiovisual metadata.

Each row in the input CSV represents a single recording. This script produces:
  1. Summary statistics (JSON + Markdown)
  2. Distribution plots (duration, file size, energy, spectral features)
  3. Relationship plots (scatter & correlation heatmap)
  4. Tables (top longest recordings, potential outliers, format aggregates)
  5. Coverage plots (cumulative hours by top N recordings)

Outputs are written under:
  figs/metadata_analysis/  (PNG figures)
  results/metadata_analysis/ (tables + report)

Usage:
  python metadata_analysis.py \
    --input_csv results/audiovisual_metadata_full.csv \
    --output_prefix 20251017 \
    --max_top 30 \
    --save_html true

Requirements: pandas, numpy, matplotlib, seaborn.
"""

from __future__ import annotations
import os
import sys
import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Global plotting configuration for publication-quality figures
sns.set_theme(context="talk", style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 110,
})


def _style_axes(ax: plt.Axes):
    """Apply consistent presentation styling: hide top/right spines, lighten grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Lighten left/bottom
    ax.spines['left'].set_alpha(0.7)
    ax.spines['bottom'].set_alpha(0.7)
    # Grid already active via seaborn style; ensure subtle
    ax.grid(alpha=0.25)
    return ax

FIG_DIR = "figs/metadata_analysis"
TABLE_DIR = "results/metadata_analysis"


@dataclass
class MetadataSummary:
    total_recordings: int
    total_hours: float
    mean_duration_s: float
    median_duration_s: float
    max_duration_s: float
    min_duration_s: float
    duration_std_s: float
    total_size_mb: float
    mean_size_mb: float
    size_std_mb: float
    formats: Dict[str, int]
    suspicious_short_count: int
    suspicious_short_threshold_s: int
    outlier_count_high_duration: int
    outlier_duration_threshold_s: float
    average_rms_energy: float
    average_zero_crossing_rate: float
    average_spectral_centroid: float
    corr_matrix: Dict[str, Dict[str, float]]
    gini_duration: float


def load_metadata(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    # Basic normalization
    expected_cols = {
        "file_name", "file_size_mb", "duration", "unique_speakers", "data_shape", "sample_rate",
        "channels", "rms_energy", "zero_crossing_rate", "spectral_centroid"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Extract extension and treat .mp4 as .mp3
    df["extension"] = df["file_name"].str.extract(r"(\.[A-Za-z0-9]+)$", expand=False).str.lower().fillna("")
    # Convert .mp4 to .mp3 for analysis purposes
    df["extension"] = df["extension"].replace(".mp4", ".mp3")
    # Duration hours
    df["duration_hours"] = df["duration"] / 3600.0
    # Size per hour (compression proxy)
    df["size_mb_per_hour"] = df.apply(lambda r: r.file_size_mb / (r.duration_hours if r.duration_hours > 0 else np.nan), axis=1)
    # Flag suspiciously short (<60s) recordings
    df["is_short"] = df["duration"] < 60
    return df


def compute_summary(df: pd.DataFrame) -> MetadataSummary:
    numeric = df[[
        "duration", "file_size_mb", "rms_energy", "zero_crossing_rate", "spectral_centroid"
    ]].dropna()
    corr = numeric.corr().round(3).to_dict()

    duration_mean = df["duration"].mean()
    duration_std = df["duration"].std()
    high_outlier_thresh = duration_mean + 2 * duration_std
    outlier_count = int((df["duration"] > high_outlier_thresh).sum())
    short_count = int((df["duration"] < 60).sum())

    # Gini coefficient for duration concentration
    durations = df["duration"].values.astype(float)
    if len(durations) == 0:
        gini = float("nan")
    else:
        sorted_d = np.sort(durations)
        n = len(sorted_d)
        cum_d = np.cumsum(sorted_d)
        gini = (2.0 * np.sum((np.arange(1, n + 1)) * sorted_d) / (n * cum_d[-1]) - (n + 1) / n)

    summary = MetadataSummary(
        total_recordings=int(len(df)),
        total_hours=float(df["duration"].sum() / 3600.0),
        mean_duration_s=float(duration_mean),
        median_duration_s=float(df["duration"].median()),
        max_duration_s=float(df["duration"].max()),
        min_duration_s=float(df["duration"].min()),
        duration_std_s=float(duration_std),
        total_size_mb=float(df["file_size_mb"].sum()),
        mean_size_mb=float(df["file_size_mb"].mean()),
        size_std_mb=float(df["file_size_mb"].std()),
        formats=df["extension"].value_counts().to_dict(),
        suspicious_short_count=short_count,
        suspicious_short_threshold_s=60,
        outlier_count_high_duration=outlier_count,
        outlier_duration_threshold_s=float(high_outlier_thresh),
        average_rms_energy=float(df["rms_energy"].mean()),
        average_zero_crossing_rate=float(df["zero_crossing_rate"].mean()),
        average_spectral_centroid=float(df["spectral_centroid"].mean()),
        corr_matrix=corr,
        gini_duration=float(round(gini, 4)),
    )
    return summary


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def plot_distribution(df: pd.DataFrame, column: str, title: str, filename: str, bins: int = 50):
    plt.figure(figsize=(9, 5))
    sns.histplot(df[column], bins=bins, kde=True, color="#1f77b4")
    
    # Custom title and xlabel based on column type
    if column == "duration":
        total_hours = df[column].sum() / 3600.0
        plt.title(f"Duration Distribution (Total = {total_hours:.1f} Hrs)")
        plt.xlabel("Duration (Seconds)")
    elif column == "file_size_mb":
        total_gb = df[column].sum() / 1024.0
        plt.title(f"File Size Distribution (Total = {total_gb:.1f} GB)")
        plt.xlabel("File Size (MB)")
    elif column == "rms_energy":
        plt.title(title)
        plt.xlabel("RMS Energy")
    else:
        plt.title(title)
        plt.xlabel(column.replace("_", " ").title())
    
    plt.ylabel("Count")
    _style_axes(plt.gca())
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None, title: str, filename: str):
    plt.figure(figsize=(8.5, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="tab10", alpha=0.7, edgecolor="none")
    plt.title(title)
    
    # Custom axis labels based on column names
    if x == "duration":
        plt.xlabel("Duration (Seconds)")
    elif x == "file_size_mb":
        plt.xlabel("File Size (MB)")
    elif x == "rms_energy":
        plt.xlabel("RMS Energy")
    else:
        plt.xlabel(x.replace("_", " ").title())
    
    if y == "duration":
        plt.ylabel("Duration (Seconds)")
    elif y == "file_size_mb":
        plt.ylabel("File Size (MB)")
    elif y == "rms_energy":
        plt.ylabel("RMS Energy")
    else:
        plt.ylabel(y.replace("_", " ").title())
    
    if hue:
        # Smaller font size for .wav and .mp3 legend labels
        plt.legend(title=hue, fontsize=10)
    _style_axes(plt.gca())
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_correlation_heatmap(df: pd.DataFrame, filename: str):
    plt.figure(figsize=(7.5, 6.5))
    numeric = df[["duration", "file_size_mb", "rms_energy", "zero_crossing_rate", "spectral_centroid", "size_mb_per_hour"]]
    corr = numeric.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 9})
    plt.title("Feature Correlation Heatmap")
    _style_axes(plt.gca())
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_cumulative_duration(df: pd.DataFrame, filename: str, max_top: int = 50):
    ordered = df.sort_values("duration", ascending=False).head(max_top).copy()
    ordered["cumulative_hours"] = ordered["duration"].cumsum() / 3600.0
    plt.figure(figsize=(9.5, 5.5))
    sns.lineplot(data=ordered, x=np.arange(1, len(ordered) + 1), y="cumulative_hours", marker="o")
    plt.title(f"Cumulative Hours Covered by Top {len(ordered)} Longest Recordings")
    plt.xlabel("Top-N Longest Recordings")
    plt.ylabel("Cumulative Hours")
    plt.grid(alpha=0.3)
    _style_axes(plt.gca())
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_lorenz_curve(df: pd.DataFrame, filename: str):
    durations = np.sort(df["duration"].values.astype(float))
    n = len(durations)
    cum_duration = np.cumsum(durations)
    cum_duration_share = cum_duration / cum_duration[-1] if n else np.array([])
    population_share = np.arange(1, n + 1) / n if n else np.array([])
    plt.figure(figsize=(7, 6))
    plt.plot(np.concatenate([[0], population_share]), np.concatenate([[0], cum_duration_share]), label="Lorenz Curve", color="#2ca02c", linewidth=2.2)
    plt.plot([0, 1], [0, 1], color="#444", linestyle="--", label="Line of Equality")
    plt.title("Lorenz Curve for Duration Concentration")
    plt.xlabel("Cumulative Share of Recordings")
    plt.ylabel("Cumulative Share of Total Duration")
    plt.legend()
    plt.grid(alpha=0.3)
    _style_axes(plt.gca())
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def generate_tables(df: pd.DataFrame, summary: MetadataSummary, output_prefix: str, max_top: int):
    # Top longest recordings
    top_long = df.sort_values("duration", ascending=False).head(max_top).copy()
    top_long["duration_hours"] = top_long["duration"] / 3600.0
    top_path = os.path.join(TABLE_DIR, f"top_{max_top}_longest_{output_prefix}.csv")
    top_long.to_csv(top_path, index=False)

    # Format aggregates
    agg = df.groupby("extension").agg(
        recordings=("file_name", "count"),
        total_size_mb=("file_size_mb", "sum"),
        avg_size_mb=("file_size_mb", "mean"),
        total_hours=("duration", lambda x: x.sum() / 3600.0),
        avg_duration_min=("duration", lambda x: x.mean() / 60.0),
        avg_rms_energy=("rms_energy", "mean"),
    ).reset_index().sort_values("recordings", ascending=False)
    if len(agg):
        total_hours_all = agg["total_hours"].sum()
        agg["percent_total_hours"] = (agg["total_hours"] / total_hours_all * 100).round(2)
    agg_path = os.path.join(TABLE_DIR, f"format_aggregates_{output_prefix}.csv")
    agg.to_csv(agg_path, index=False)

    # Outliers high duration
    outlier_mask = df["duration"] > summary.outlier_duration_threshold_s
    outliers = df[outlier_mask].copy()
    outliers["duration_hours"] = outliers["duration"] / 3600.0
    out_path = os.path.join(TABLE_DIR, f"high_duration_outliers_{output_prefix}.csv")
    outliers.to_csv(out_path, index=False)

    # Short recordings
    short_df = df[df["is_short"]].copy()
    short_path = os.path.join(TABLE_DIR, f"short_recordings_{output_prefix}.csv")
    short_df.to_csv(short_path, index=False)

    return {
        "top_longest": top_path,
        "format_aggregates": agg_path,
        "high_duration_outliers": out_path,
        "short_recordings": short_path,
    }


def write_summary(summary: MetadataSummary, figure_paths: Dict[str, str], table_paths: Dict[str, str], output_prefix: str, save_html: bool):
    summary_json_path = os.path.join(TABLE_DIR, f"summary_{output_prefix}.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    md_path = os.path.join(TABLE_DIR, f"report_{output_prefix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Audiovisual Metadata Analysis Report\n\n")
        f.write(f"**Total recordings:** {summary.total_recordings}\n\n")
        f.write(f"**Total hours (approx):** {summary.total_hours:.2f}\n\n")
        f.write("## Duration Statistics\n")
        f.write(f"- Mean: {summary.mean_duration_s/60:.2f} min\n")
        f.write(f"- Median: {summary.median_duration_s/60:.2f} min\n")
        f.write(f"- Max: {summary.max_duration_s/3600:.2f} h\n")
        f.write(f"- Min: {summary.min_duration_s:.1f} s\n")
        f.write(f"- Std Dev: {summary.duration_std_s/60:.2f} min\n")
        f.write(f"- High duration outlier threshold (> mean + 2*std): {summary.outlier_duration_threshold_s/3600:.2f} h\n")
        f.write(f"- Outliers above threshold: {summary.outlier_count_high_duration}\n\n")
        f.write("## Duration Concentration\n")
        f.write(f"- Gini coefficient (duration): {summary.gini_duration:.4f}\n\n")
        f.write("## File Size\n")
        f.write(f"- Total size: {summary.total_size_mb:.1f} MB\n")
        f.write(f"- Mean size: {summary.mean_size_mb:.2f} MB\n")
        f.write(f"- Size std dev: {summary.size_std_mb:.2f} MB\n\n")
        f.write("## Formats\n")
        for ext, count in summary.formats.items():
            f.write(f"- {ext or '[no extension]'}: {count}\n")
        f.write("\n## Percent of Total Hours by Format\n")
        format_table = table_paths.get("format_aggregates")
        try:
            agg_df = pd.read_csv(format_table)
            if "percent_total_hours" in agg_df.columns:
                for _, r in agg_df.iterrows():
                    f.write(f"- {r['extension'] or '[no extension]'}: {r['percent_total_hours']:.2f}% of total hours\n")
        except Exception:
            f.write("(Format aggregate table not readable for percent hours)\n")
        f.write("\n## Spectral / Energy Averages\n")
        f.write(f"- RMS Energy (mean): {summary.average_rms_energy:.4f}\n")
        f.write(f"- Zero Crossing Rate (mean): {summary.average_zero_crossing_rate:.4f}\n")
        f.write(f"- Spectral Centroid (mean): {summary.average_spectral_centroid:.2f}\n\n")
        f.write("## Short Recordings\n")
        f.write(f"- Threshold (< {summary.suspicious_short_threshold_s}s): {summary.suspicious_short_count} recordings\n\n")
        f.write("## Correlations (selected)\n")
        for row_key, row_vals in summary.corr_matrix.items():
            f.write(f"- {row_key}: {row_vals}\n")
        f.write("\n## Generated Figures\n")
        for label, path in figure_paths.items():
            f.write(f"- {label}: {path}\n")
        f.write("\n## Generated Tables\n")
        for label, path in table_paths.items():
            f.write(f"- {label}: {path}\n")

    if save_html:
        html_path = os.path.join(TABLE_DIR, f"report_{output_prefix}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><title>Metadata Analysis Report</title><style>body{font-family:Arial;margin:20px;} img{max-width:100%;height:auto;}</style></head><body>\n")
            f.write(f"<h1>Audiovisual Metadata Analysis Report</h1>\n")
            f.write(f"<p>Total recordings: <b>{summary.total_recordings}</b><br>Total hours: <b>{summary.total_hours:.2f}</b></p>\n")
            f.write("<h2>Figures</h2>\n<ul>")
            for label, path in figure_paths.items():
                rel = os.path.relpath(path)
                f.write(f"<li><h3>{label}</h3><img src='../{rel}' alt='{label}'/></li>\n")
            f.write("</ul><h2>Tables</h2><ul>")
            for label, path in table_paths.items():
                rel = os.path.relpath(path)
                f.write(f"<li>{label}: {rel}</li>\n")
            f.write("</ul><h2>Summary JSON</h2><pre>" + json.dumps(asdict(summary), indent=2) + "</pre>")
            f.write("</body></html>")
    return md_path


def generate_all(df: pd.DataFrame, output_prefix: str, max_top: int, save_html: bool) -> Dict[str, Any]:
    ensure_dirs()
    df = enrich_df(df)
    summary = compute_summary(df)

    figure_paths = {}
    figure_paths["duration_distribution"] = plot_distribution(df, "duration", "Duration Distribution (Seconds)", f"duration_dist_{output_prefix}.png")
    figure_paths["file_size_mb_distribution"] = plot_distribution(df, "file_size_mb", "File Size Distribution (MB)", f"file_size_dist_{output_prefix}.png")
    figure_paths["rms_energy_distribution"] = plot_distribution(df, "rms_energy", "RMS Energy Distribution", f"rms_energy_dist_{output_prefix}.png")
    figure_paths["zero_crossing_rate_distribution"] = plot_distribution(df, "zero_crossing_rate", "Zero Crossing Rate Distribution", f"zcr_dist_{output_prefix}.png")
    figure_paths["spectral_centroid_distribution"] = plot_distribution(df, "spectral_centroid", "Spectral Centroid Distribution", f"spectral_centroid_dist_{output_prefix}.png")
    figure_paths["size_vs_duration"] = plot_scatter(df, "duration", "file_size_mb", "extension", "Duration vs File Size", f"duration_vs_size_{output_prefix}.png")
    figure_paths["energy_vs_centroid"] = plot_scatter(df, "rms_energy", "spectral_centroid", "extension", "RMS Energy vs Spectral Centroid", f"energy_vs_centroid_{output_prefix}.png")
    figure_paths["zero_cross_vs_centroid"] = plot_scatter(df, "zero_crossing_rate", "spectral_centroid", "extension", "Zero Crossing Rate vs Spectral Centroid", f"zcr_vs_centroid_{output_prefix}.png")
    figure_paths["correlation_heatmap"] = plot_correlation_heatmap(df, f"correlation_heatmap_{output_prefix}.png")
    figure_paths["cumulative_hours_topN"] = plot_cumulative_duration(df, f"cumulative_hours_{output_prefix}.png", max_top=max_top)
    figure_paths["lorenz_curve_duration"] = plot_lorenz_curve(df, f"lorenz_duration_{output_prefix}.png")

    table_paths = generate_tables(df, summary, output_prefix, max_top)
    report_path = write_summary(summary, figure_paths, table_paths, output_prefix, save_html)

    return {
        "summary": asdict(summary),
        "figures": figure_paths,
        "tables": table_paths,
        "report_md": report_path,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Generate figures and analytical summaries for audiovisual metadata CSV.")
    p.add_argument("--input_csv", required=True, help="Path to metadata CSV (one recording per row).")
    p.add_argument("--output_prefix", default="analysis", help="Prefix tag appended to output filenames.")
    p.add_argument("--max_top", type=int, default=30, help="Top N longest recordings for tables & cumulative plot.")
    p.add_argument("--save_html", type=lambda x: x.lower() in {"true","1","yes"}, default=True, help="Also emit an HTML report.")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_metadata(args.input_csv)
    result = generate_all(df, args.output_prefix, args.max_top, args.save_html)
    print(f"✅ Metadata analysis complete. Summary hours: {result['summary']['total_hours']:.2f}. Report: {result['report_md']}")


if __name__ == "__main__":
    main()
