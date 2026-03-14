#!/usr/bin/env python3
"""
STAT 482 — Report 1 EDA Analysis
=================================
Fetches 2025 MLB Statcast data via pybaseball, cleans it per the report
criteria, and generates all five EDA figures plus summary statistics.

Figures produced (saved to ../figures/):
  1. fig_whiff_by_pitch_type.png
  2. fig_physical_distributions.png
  3. fig_sequence_heatmap.png
  4. fig_batter_whiff_dist.png
  5. fig_correlation_matrix.png

Usage:
    python3 code/eda_analysis.py        (from the STAT482 project root)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pybaseball import statcast

# ── paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
FIG_DIR      = os.path.join(PROJECT_ROOT, "figures")
CACHE_FILE   = os.path.join(DATA_DIR, "statcast_2025.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = sns.color_palette("colorblind")

# ── pitch-type labels (mapping Statcast codes → readable names) ───
PITCH_LABELS = {
    "FF": "4-Seam FB",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle-Curve",
    "CH": "Changeup",
    "FS": "Splitter",
}
KEEP_PITCH_TYPES = set(PITCH_LABELS.keys())

# Whiff descriptions
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}

# Non-competitive descriptions to remove
NON_COMPETITIVE = {
    "pitchout", "intent_ball", "intentional_ball",
    "pitchout_hit_into_play",
}


# ════════════════════════════════════════════════════════════════════
#  1. DATA FETCH
# ════════════════════════════════════════════════════════════════════
def fetch_data() -> pd.DataFrame:
    """Fetch 2025 Statcast data; cache to CSV for subsequent runs."""
    if os.path.exists(CACHE_FILE):
        print(f"[INFO] Loading cached data from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, low_memory=False)
    else:
        print("[INFO] Fetching 2025 Statcast data from Baseball Savant...")
        print("       (This may take several minutes.)")
        # Fetch in month-long chunks to avoid timeouts
        chunks = []
        date_ranges = [
            ("2025-03-27", "2025-04-30"),
            ("2025-05-01", "2025-05-31"),
            ("2025-06-01", "2025-06-30"),
            ("2025-07-01", "2025-07-31"),
            ("2025-08-01", "2025-08-31"),
            ("2025-09-01", "2025-09-28"),
        ]
        for start, end in date_ranges:
            print(f"       Fetching {start} to {end} ...")
            chunk = statcast(start_dt=start, end_dt=end)
            chunks.append(chunk)
            print(f"       → {len(chunk):,} pitches")
        df = pd.concat(chunks, ignore_index=True)
        print(f"[INFO] Saving {len(df):,} raw pitches to {CACHE_FILE}")
        df.to_csv(CACHE_FILE, index=False)
    return df


# ════════════════════════════════════════════════════════════════════
#  2. DATA CLEANING
# ════════════════════════════════════════════════════════════════════
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning filters described in Report 1 §2.4."""
    n_raw = len(raw)
    print(f"\n[CLEAN] Starting with {n_raw:,} raw pitches")

    df = raw.copy()

    # 1. Remove non-competitive events
    df = df[~df["description"].isin(NON_COMPETITIVE)]
    # Also filter by pitch type if there's a dedicated column
    if "pitch_type" in df.columns:
        df = df[~df["pitch_type"].isin(["PO", "IN"])]  # pitchout / intentional
    print(f"  After removing non-competitive: {len(df):,}")

    # 2. Drop missing physical measurements
    phys_cols = ["release_speed", "pfx_x", "pfx_z", "release_extension"]
    df = df.dropna(subset=phys_cols)
    print(f"  After dropping missing physics:  {len(df):,}")

    # 3. Remove unrealistic release speeds
    df = df[(df["release_speed"] >= 65) & (df["release_speed"] <= 105)]
    print(f"  After speed filter [65–105]:     {len(df):,}")

    # 4. Keep only the 9 pitch types
    df = df[df["pitch_type"].isin(KEEP_PITCH_TYPES)]
    print(f"  After keeping 9 pitch types:     {len(df):,}")

    # Create readable labels
    df["pitch_label"] = df["pitch_type"].map(PITCH_LABELS)

    # 5. Convert pfx_z and pfx_x from feet to inches
    df["ivb_inches"] = df["pfx_z"] * 12
    df["hb_inches"]  = df["pfx_x"] * 12

    # 6. Binary whiff column
    df["whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)

    # 6. Lag variables within each plate appearance
    #    Group by game + at-bat number
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    grp = df.groupby(["game_pk", "at_bat_number"])
    df["prev_pitch_type"]  = grp["pitch_type"].shift(1).fillna("None")
    df["prev_pitch_label"] = df["prev_pitch_type"].map(
        lambda x: PITCH_LABELS.get(x, "None")
    )
    df["prev_zone"]        = grp["zone"].shift(1).fillna(-1).astype(int)

    # Simplified previous outcome categories
    def classify_outcome(desc):
        if pd.isna(desc):
            return "None"
        if "ball" in desc and "foul" not in desc:
            return "Ball"
        if "called_strike" in desc:
            return "Called Strike"
        if "swinging_strike" in desc:
            return "Swinging Strike"
        if "foul" in desc:
            return "Foul"
        if "hit_into_play" in desc:
            return "In Play"
        return "Other"

    df["prev_outcome"] = grp["description"].shift(1).apply(classify_outcome)

    n_clean = len(df)
    print(f"\n[CLEAN] Final cleaned dataset: {n_clean:,} pitches")

    return df, n_raw


# ════════════════════════════════════════════════════════════════════
#  3. SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════════════
def print_summary(df: pd.DataFrame, n_raw: int):
    """Print LaTeX-ready summary statistics."""
    overall_whiff = df["whiff"].mean() * 100

    # Batters with >= 200 pitches seen
    batter_whiffs = (
        df.groupby("batter")
        .agg(n_pitches=("whiff", "size"), whiff_rate=("whiff", "mean"))
    )
    batter_200 = batter_whiffs[batter_whiffs["n_pitches"] >= 200]
    batter_200_rate = batter_200["whiff_rate"] * 100

    n_pitchers = df["pitcher"].nunique()

    print("\n" + "=" * 60)
    print("  LATEX MACRO VALUES  (copy to Report1.tex lines 22-28)")
    print("=" * 60)
    print(f"  \\totalRaw     = {n_raw:,}")
    print(f"  \\totalPitches = {len(df):,}")
    print(f"  \\overallWhiff = {overall_whiff:.1f}")
    print(f"  \\numBatters   = {len(batter_200)}")
    print(f"  \\numPitchers  = {n_pitchers}")
    print(f"  \\batterLow    = {batter_200_rate.min():.1f}")
    print(f"  \\batterHigh   = {batter_200_rate.max():.1f}")
    print("=" * 60)

    # Summary statistics table (pfx in inches)
    phys = {
        "Release Speed (mph)":          df["release_speed"],
        "Induced Vertical Break (in.)": df["ivb_inches"],
        "Horizontal Break (in.)":       df["hb_inches"],
        "Release Extension (ft)":       df["release_extension"],
    }
    print("\n  SUMMARY STATISTICS TABLE  (lines 170-184)")
    print("-" * 60)
    for name, col in phys.items():
        print(f"  {name:35s}  {col.mean():7.1f}  {col.std():5.1f}  "
              f"{col.min():7.1f}  {col.max():6.1f}")
    print("-" * 60)

    # Whiff by pitch type table
    whiff_tbl = (
        df.groupby("pitch_label")
        .agg(total=("whiff", "size"), whiffs=("whiff", "sum"))
    )
    whiff_tbl["rate"] = whiff_tbl["whiffs"] / whiff_tbl["total"] * 100
    whiff_tbl = whiff_tbl.sort_values("rate", ascending=False)
    print("\n  WHIFF BY PITCH TYPE TABLE  (lines 201-220)")
    print("-" * 60)
    for label, row in whiff_tbl.iterrows():
        print(f"  {label:15s}  {int(row['total']):>8,}  {int(row['whiffs']):>7,}  "
              f"{row['rate']:5.1f}%")
    print("-" * 60)

    return batter_200_rate, overall_whiff


# ════════════════════════════════════════════════════════════════════
#  4. FIGURE GENERATION
# ════════════════════════════════════════════════════════════════════

def fig_whiff_by_pitch_type(df: pd.DataFrame):
    """Figure 1 — bar chart of whiff rate by pitch type."""
    rates = (
        df.groupby("pitch_label")["whiff"]
        .mean()
        .sort_values(ascending=False) * 100
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(rates.index, rates.values, color=PALETTE[:len(rates)],
                  edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Whiff Rate (%)")
    ax.set_xlabel("Pitch Type")
    ax.set_title("Whiff Rate by Pitch Type — 2025 MLB Regular Season")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    # Add value labels on bars
    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_whiff_by_pitch_type.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved {path}")


def fig_physical_distributions(df: pd.DataFrame):
    """Figure 2 — 2×2 density plots of physical characteristics."""
    cols  = ["release_speed", "ivb_inches", "hb_inches", "release_extension"]
    names = ["Release Speed (mph)", "Induced Vertical Break (in.)",
             "Horizontal Break (in.)", "Release Extension (ft)"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col, name in zip(axes.flat, cols, names):
        for label, color in [("No Whiff", PALETTE[0]), ("Whiff", PALETTE[1])]:
            subset = df[df["whiff"] == (1 if label == "Whiff" else 0)]
            ax.hist(subset[col], bins=80, density=True, alpha=0.5,
                    label=label, color=color)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    fig.suptitle("Physical Pitch Characteristics — Whiff vs. No Whiff",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_physical_distributions.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved {path}")


def fig_sequence_heatmap(df: pd.DataFrame):
    """Figure 3 — heatmap of whiff rate by (previous → current) pitch type."""
    # Exclude first pitches (prev = "None")
    seq = df[df["prev_pitch_label"] != "None"].copy()

    # Compute whiff rate per cell
    ct = seq.groupby(["prev_pitch_label", "pitch_label"]).agg(
        n=("whiff", "size"), whiffs=("whiff", "sum")
    ).reset_index()
    ct["rate"] = ct["whiffs"] / ct["n"] * 100

    # Filter cells with fewer than 100 observations
    ct = ct[ct["n"] >= 100]

    # Pivot
    pivot = ct.pivot(index="prev_pitch_label", columns="pitch_label", values="rate")

    # Order by overall whiff rate
    order = (
        df.groupby("pitch_label")["whiff"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    # Keep only labels that exist in our data
    order = [o for o in order if o in pivot.columns]
    row_order = [o for o in order if o in pivot.index]
    pivot = pivot.reindex(index=row_order, columns=order)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Whiff Rate (%)"})
    ax.set_xlabel("Current Pitch Type")
    ax.set_ylabel("Previous Pitch Type")
    ax.set_title("Whiff Rate by Pitch Sequence — 2025 MLB")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_sequence_heatmap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved {path}")


def fig_batter_whiff_dist(batter_rates: pd.Series, overall_whiff: float):
    """Figure 4 — histogram of per-batter whiff rates."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(batter_rates, bins=40, color=PALETTE[0], edgecolor="white",
            linewidth=0.8, alpha=0.85)
    ax.axvline(overall_whiff, color="red", linestyle="--", linewidth=1.5,
               label=f"League avg = {overall_whiff:.1f}%")
    ax.set_xlabel("Batter Whiff Rate (%)")
    ax.set_ylabel("Number of Batters")
    ax.set_title("Distribution of Batter-Level Whiff Rates (≥ 200 pitches seen)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_batter_whiff_dist.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved {path}")


def fig_correlation_matrix(df: pd.DataFrame):
    """Figure 5 — correlation matrix of continuous predictors + whiff."""
    cols = ["release_speed", "ivb_inches", "hb_inches", "release_extension", "whiff"]
    labels = ["Speed", "IVB", "HB", "Extension", "Whiff"]
    corr = df[cols].corr()
    corr.index   = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                square=True)
    ax.set_title("Correlation Matrix — Pitch Characteristics & Whiff Outcome")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_correlation_matrix.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved {path}")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    raw = fetch_data()
    df, n_raw = clean_data(raw)
    batter_rates, overall_whiff = print_summary(df, n_raw)

    print("\n[INFO] Generating figures...")
    fig_whiff_by_pitch_type(df)
    fig_physical_distributions(df)
    fig_sequence_heatmap(df)
    fig_batter_whiff_dist(batter_rates, overall_whiff)
    fig_correlation_matrix(df)
    print("\n[DONE] All figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
