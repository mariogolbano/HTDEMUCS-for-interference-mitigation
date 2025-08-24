#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make 2 charts (one per DATASET), comparing the 3 MODELS on the same plot.
Both metrics on the SAME chart (dual y-axes):
  - ΔSIR (dB): solid line with circles  (LEFT axis)
  - NMSE (dB): dashed line with triangles (RIGHT axis)

Color encodes MODEL (arch): hybrid, time, spec.

Reads:
  ROOT/{dataset}/sinr_{XdB}/eval/{arch}/metrics_by_sinr.csv
Writes:
  OUT_DIR/{dataset}_models_sir_nmse_vs_sinr.png

Example:
  python plot_eval_by_sinr_models_per_dataset.py \
    --root DATASET_FINAL/rf_datasets_h5/test/sinr_db \
    --out-dir DATASET_FINAL/rf_datasets_h5/test/evals_graphs \
    --datasets 64qam_wifi,256qam_wifi \
    --archs hybrid,time,spec \
    --dpi 160 --save-pdf
"""
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DATASET_FINAL/rf_datasets_h5/test/sinr_db",
                    help="Root with {dataset}/sinr_*dB/...")
    ap.add_argument("--out-dir", default="DATASET_FINAL/rf_datasets_h5/test/evals_graphs",
                    help="Output folder for figures")
    ap.add_argument("--datasets", default="",
                    help="Comma-separated datasets (e.g., 64qam_wifi,256qam_wifi). If empty: auto-detect.")
    ap.add_argument("--archs", default="hybrid,time,spec",
                    help="Comma-separated architectures (models) to compare")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--save-pdf", action="store_true")
    # optional fixed axis limits
    ap.add_argument("--nmse-ylim", default="", help="e.g., '-20,0' for NMSE axis")
    ap.add_argument("--dsir-ylim", default="", help="e.g., '0,60' for ΔSIR axis")
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

def plot_one_dataset(root, dataset, archs, colors, out_dir, nmse_ylim=None, dsir_ylim=None, dpi=160, save_pdf=False):
    # gather data
    data = {arch: collect_metrics(root, dataset, arch) for arch in archs}
    if all(df.empty for df in data.values()):
        print(f"[WARN] No data available for dataset={dataset}")
        return

    fig, ax_left = plt.subplots(figsize=(10, 4.8))
    ax_right = ax_left.twinx()

    # styles per metric
    metric_styles = {
        "dsir": dict(linestyle="-",  marker="o", linewidth=1.8),
        "nmse": dict(linestyle="--", marker="^", linewidth=1.8),
    }

    handles = []
    for arch in archs:
        df = data[arch]
        if df.empty:
            print(f"[WARN] Missing {arch} for {dataset}")
            continue
        x = df["sinr"].values
        y_d = df["delta_sir_db_mean"].values
        y_n = df["nmse_db_mean"].values
        c = colors.get(arch, "black")

        # ΔSIR (left)
        l1, = ax_left.plot(x, y_d, color=c, label=f"{arch} ΔSIR", **metric_styles["dsir"])
        # NMSE (right)
        l2, = ax_right.plot(x, y_n, color=c, label=f"{arch} NMSE", **metric_styles["nmse"])
        handles.extend([l1, l2])

    ax_left.set_xlabel("SINR (dB)")
    ax_left.set_ylabel("ΔSIR (dB)")
    ax_right.set_ylabel("NMSE (dB)")

    ax_right.set_ylim(-9, 7)   # ← límites fijos para la loss
    ax_left.set_ylim(25, 50)   # ← límites fijos para la loss

    if dsir_ylim: ax_left.set_ylim(*dsir_ylim)
    if nmse_ylim: ax_right.set_ylim(*nmse_ylim)

    ax_left.grid(True, alpha=0.3)
    fig.suptitle(f"{dataset} — Models comparison: ΔSIR & NMSE vs SINR", fontsize=12)

    # Legends: colors=arch, styles=metrics
    arch_patches = [Patch(facecolor=colors[a], edgecolor='none', label=a) for a in archs if a in colors]
    leg_a = ax_left.legend(handles=arch_patches, title="Models", loc="upper left")
    ax_left.add_artist(leg_a)

    style_elems = [
        Line2D([0], [0], color="black", linestyle=metric_styles["dsir"]["linestyle"],
               marker=metric_styles["dsir"]["marker"], label="ΔSIR (left)"),
        Line2D([0], [0], color="black", linestyle=metric_styles["nmse"]["linestyle"],
               marker=metric_styles["nmse"]["marker"], label="NMSE (right)"),
    ]
    ax_left.legend(handles=style_elems, title="Metrics", loc="upper right")

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{dataset}_models_sir_nmse_vs_sinr.png")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=dpi)
    if save_pdf:
        out_pdf = os.path.join(out_dir, f"{dataset}_models_sir_nmse_vs_sinr.pdf")
        fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[OK] {out_png}")

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = list_datasets(args.root, args.datasets)
    archs    = [a.strip() for a in args.archs.split(",") if a.strip()]
    if not datasets:
        print("[WARN] No datasets found. Check --root or --datasets.")
        return

    # choose distinct, coherent colors per model
    colors = {
        "hybrid": "tab:blue",
        "time":   "tab:green",
        "spec":   "tab:red",
    }

    nmse_ylim = parse_ylim(args.nmse_ylim)
    dsir_ylim = parse_ylim(args.dsir_ylim)

    for ds in datasets:
        plot_one_dataset(args.root, ds, archs, colors, args.out_dir,
                         nmse_ylim=nmse_ylim, dsir_ylim=dsir_ylim,
                         dpi=args.dpi, save_pdf=args.save_pdf)

if __name__ == "__main__":
    # lazy imports for legend helpers
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    main()
