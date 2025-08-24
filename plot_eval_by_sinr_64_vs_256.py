#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make 3 charts (one per architecture), overlaying BOTH datasets on the same plot.
Color encodes DATASET, line style encodes METRIC:
  - ΔSIR (dB): solid line with circles
  - NMSE (dB): dashed line with triangles  (right y-axis)

Reads:
  ROOT/{dataset}/sinr_{XdB}/eval/{arch}/metrics_by_sinr.csv
Writes:
  OUT_DIR/combined_{arch}_sir_nmse_vs_sinr.png

Example:
  python plot_eval_by_sinr_combined.py \
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
    ap.add_argument("--datasets", default="64qam_wifi,256qam_wifi",
                    help="Comma-separated datasets to overlay (exactly two recommended)")
    ap.add_argument("--archs", default="hybrid,time,spec",
                    help="Comma-separated architectures (one figure per arch)")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--save-pdf", action="store_true")
    # Optional fixed axis limits (leave empty to auto)
    ap.add_argument("--nmse-ylim", default="", help="e.g., '-20,0' for NMSE axis")
    ap.add_argument("--dsir-ylim", default="", help="e.g., '0,60' for ΔSIR axis")
    return ap.parse_args()

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

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    archs    = [a.strip() for a in args.archs.split(",") if a.strip()]
    if len(datasets) < 2:
        print("[WARN] Fewer than two datasets provided; overlay will still work with what is available.")

    # Assign two distinct colors to datasets (extendable if more)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    colors = {ds: palette[i % len(palette)] for i, ds in enumerate(datasets)}

    nmse_ylim = parse_ylim(args.nmse_ylim)
    dsir_ylim = parse_ylim(args.dsir_ylim)

    for arch in archs:
        # Collect data for each dataset
        data = {ds: collect_metrics(args.root, ds, arch) for ds in datasets}

        fig, ax_left = plt.subplots(figsize=(10, 4.8))
        ax_right = ax_left.twinx()

        # Plot: for each dataset, ΔSIR on left (solid + circles), NMSE on right (dashed + triangles)
        handles_lines = []
        for ds in datasets:
            df = data.get(ds, pd.DataFrame())
            if df.empty:
                print(f"[WARN] No data for {ds} / {arch}")
                continue
            x = df["sinr"].values
            y_dsir = df["delta_sir_db_mean"].values
            y_nmse = df["nmse_db_mean"].values
            c = colors[ds]

            l_dsir, = ax_left.plot(
                x, y_dsir, color=c, marker="o", linewidth=1.8,
                linestyle="-", label=f"{ds} ΔSIR"
            )
            l_nmse, = ax_right.plot(
                x, y_nmse, color=c, marker="^", linewidth=1.8,
                linestyle="--", label=f"{ds} NMSE"
            )
            handles_lines.extend([l_dsir, l_nmse])

        # Axis labels
        ax_left.set_xlabel("SINR (dB)")
        ax_left.set_ylabel("ΔSIR (dB)")
        ax_right.set_ylabel("NMSE (dB)")

        # Optional fixed limits
        if dsir_ylim: ax_left.set_ylim(*dsir_ylim)
        if nmse_ylim: ax_right.set_ylim(*nmse_ylim)

        # Grids and title
        ax_left.grid(True, alpha=0.3)
        fig.suptitle(f"{arch} — ΔSIR & NMSE vs SINR (two datasets)", fontsize=12)

        # Legends: dataset colors + metric styles
        # Legend A: datasets (color patches)
        ds_patches = [Patch(facecolor=colors[ds], edgecolor='none', label=ds) for ds in datasets]
        leg_a = ax_left.legend(handles=ds_patches, title="Datasets", loc="upper left")
        ax_left.add_artist(leg_a)

        # Legend B: metric styles
        style_elems = [
            Line2D([0], [0], color="black", linestyle="-",  marker="o", label="ΔSIR (left)"),
            Line2D([0], [0], color="black", linestyle="--", marker="^", label="NMSE (right)"),
        ]
        ax_left.legend(handles=style_elems, title="Metrics", loc="upper right")

        ax_right.set_ylim(-9, 6)   # ← límites fijos para la loss
        ax_left.set_ylim(25, 50)   # ← límites fijos para la loss

        # Save
        out_png = os.path.join(args.out_dir, f"combined_{arch}_sir_nmse_vs_sinr.png")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(out_png, dpi=args.dpi)
        if args.save_pdf:
            out_pdf = os.path.join(args.out_dir, f"combined_{arch}_sir_nmse_vs_sinr.pdf")
            fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[OK] {out_png}")

if __name__ == "__main__":
    main()
