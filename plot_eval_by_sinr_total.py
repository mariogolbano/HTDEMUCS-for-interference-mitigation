#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One combined chart: ALL datasets × ALL models on the SAME figure (dual y-axes).
Style scheme for legibility:
  - Color encodes MODEL (arch): hybrid, time, spec
  - Marker encodes DATASET (e.g., 64qam_wifi, 256qam_wifi, ...)
  - Line style encodes METRIC:
        ΔSIR (dB) -> solid line (LEFT axis)
        NMSE (dB) -> dashed line (RIGHT axis)

Reads:
  ROOT/{dataset}/sinr_{XdB}/eval/{arch}/metrics_by_sinr.csv
Writes:
  OUT_DIR/combined_all_models_datasets_sir_nmse_vs_sinr.png

Example:
  python plot_eval_by_sinr_all_in_one.py \
    --root DATASET_FINAL/rf_datasets_h5/test/sinr_db \
    --out-dir DATASET_FINAL/rf_datasets_h5/test/evals_graphs \
    --datasets 64qam_wifi,256qam_wifi \
    --archs hybrid,time,spec \
    --dpi 160 --save-pdf \
    --dsir-ylim 0,60 --nmse-ylim -20,0
"""
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DATASET_FINAL/rf_datasets_h5/test/sinr_db",
                    help="Root with {dataset}/sinr_*dB/...")
    ap.add_argument("--out-dir", default="DATASET_FINAL/rf_datasets_h5/test/evals_graphs",
                    help="Output folder for figures")
    ap.add_argument("--datasets", default="",
                    help="Comma-separated datasets (if empty: auto-detect all *_* folders)")
    ap.add_argument("--archs", default="hybrid,time,spec",
                    help="Comma-separated model names/architectures")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--save-pdf", action="store_true")
    ap.add_argument("--nmse-ylim", default="", help="Right axis limits, e.g. '-20,0'")
    ap.add_argument("--dsir-ylim", default="", help="Left axis limits, e.g. '0,60'")
    return ap.parse_args()

def list_datasets(root, explicit):
    if explicit:
        return [d.strip() for d in explicit.split(",") if d.strip()]
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and "_" in name:
            out.append(name)
    return sorted(out)

def sinr_from_dirname(name: str):
    m = re.match(r"^sinr_([mp]?\d+(?:\.\d+)?)dB$", name)
    if not m: return None
    s = m.group(1).replace("m", "-").replace("p", "+")
    try:
        return float(s)
    except:
        return None

def collect_metrics(root, dataset, arch):
    """
    Return DataFrame with columns: sinr, delta_sir_db_mean, nmse_db_mean, count
    """
    base = os.path.join(root, dataset)
    rows = []
    if not os.path.isdir(base):
        return pd.DataFrame()
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if not os.path.isdir(p): 
            continue
        sinr = sinr_from_dirname(name)
        if sinr is None:
            continue
        csv_path = os.path.join(p, "eval", arch, "metrics_by_sinr.csv")
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        r = df.iloc[0]
        rows.append({
            "sinr": float(sinr),
            "delta_sir_db_mean": float(r.get("delta_sir_db_mean", np.nan)),
            "nmse_db_mean": float(r.get("nmse_db_mean", np.nan)),
            "count": int(r.get("count", 0)),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sinr").reset_index(drop=True)

def parse_ylim(text):
    if not text:
        return None
    try:
        a, b = text.split(",")
        return (float(a.strip()), float(b.strip()))
    except Exception:
        return None

def pretty_label(name: str):
    # Simple prettifier: "64qam_wifi" -> "64qam + wifi"
    return name.replace("_", " + ")

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = list_datasets(args.root, args.datasets)
    archs    = [a.strip() for a in args.archs.split(",") if a.strip()]

    if not datasets:
        print("[WARN] No datasets found. Check --root or --datasets.")
        return
    if not archs:
        print("[WARN] No architectures provided.")
        return

    # Colors per model (consistent & distinct)
    arch_colors = {
        "hybrid": "tab:blue",
        "time":   "tab:green",
        "spec":   "tab:red",
    }
    # Markers per dataset (cycle if more than provided)
    ds_markers_cycle = ["o", "s", "^", "D", "v", "P", "*"]
    ds_marker = {ds: ds_markers_cycle[i % len(ds_markers_cycle)] for i, ds in enumerate(datasets)}

    # Metric styles
    style_dsir = dict(linestyle="-",  linewidth=1.9)  # solid
    style_nmse = dict(linestyle="--", linewidth=1.9)  # dashed

    nmse_ylim = parse_ylim(args.nmse_ylim)
    dsir_ylim = parse_ylim(args.dsir_ylim)

    # Collect all data first
    data = {(ds, arch): collect_metrics(args.root, ds, arch) for ds in datasets for arch in archs}

    # Build figure
    fig, ax_left = plt.subplots(figsize=(11.5, 7))
    ax_right = ax_left.twinx()

    # Plot every dataset×arch pair
    for ds in datasets:
        for arch in archs:
            df = data[(ds, arch)]
            if df.empty:
                print(f"[WARN] Missing data for {ds}/{arch}")
                continue
            x = df["sinr"].values
            y_d = df["delta_sir_db_mean"].values
            y_n = df["nmse_db_mean"].values
            color = arch_colors.get(arch, "black")
            marker = ds_marker[ds]

            # ΔSIR (left): solid + dataset marker
            ax_left.plot(
                x, y_d,
                color=color,
                marker=marker, markersize=5,
                label=f"{pretty_label(ds)} · {arch} ΔSIR",
                **style_dsir
            )
            # NMSE (right): dashed + same dataset marker
            ax_right.plot(
                x, y_n,
                color=color,
                marker=marker, markersize=5,
                label=f"{pretty_label(ds)} · {arch} NMSE",
                **style_nmse
            )

    # Labels & limits
    ax_left.set_xlabel("SINR (dB)")
    ax_left.set_ylabel("ΔSIR (dB)")
    ax_right.set_ylabel("NMSE (dB)")
    if dsir_ylim: ax_left.set_ylim(*dsir_ylim)
    if nmse_ylim: ax_right.set_ylim(*nmse_ylim)

    ax_right.set_ylim(-9, 5)   # ← límites fijos para la loss
    ax_left.set_ylim(25, 50)   # ← límites fijos para la loss

    ax_left.grid(True, alpha=0.3)
    fig.suptitle("All models & datasets — ΔSIR (left) and NMSE (right) vs SINR", fontsize=12)

    # Legends: separate, compact and readable
    # Legend for models (colors)
    model_handles = [Patch(facecolor=arch_colors.get(a, "black"), edgecolor='none', label=a) for a in archs]
    leg_models = ax_left.legend(handles=model_handles, title="Models (color)", loc="upper left")
    ax_left.add_artist(leg_models)

    # Legend for datasets (markers)
    ds_handles = [Line2D([0], [0], color="black", marker=ds_marker[d], linestyle="",
                         markersize=6, label=pretty_label(d)) for d in datasets]
    leg_dsets = ax_left.legend(handles=ds_handles, title="Datasets (marker)", loc="upper center")
    ax_left.add_artist(leg_dsets)

    # Legend for metrics (line style)
    metric_handles = [
        Line2D([0], [0], color="black", **style_dsir, marker="", label="ΔSIR (left)"),
        Line2D([0], [0], color="black", **style_nmse, marker="", label="NMSE (right)"),
    ]
    ax_left.legend(handles=metric_handles, title="Metrics (style)", loc="upper right")

    out_png = os.path.join(args.out_dir, "combined_all_models_datasets_sir_nmse_vs_sinr.png")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=args.dpi)
    if args.save_pdf:
        out_pdf = os.path.join(args.out_dir, "combined_all_models_datasets_sir_nmse_vs_sinr.pdf")
        fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[OK] {out_png}")

if __name__ == "__main__":
    main()
