#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One figure with TWO side-by-side charts:
  - Left: ΔSIR (dB) vs SINR
  - Right: NMSE (dB) vs SINR
Color encodes MODEL (arch): hybrid, time, spec
Marker encodes DATASET: e.g., 64qam_wifi, 256qam_wifi

Legend is placed ABOVE the plots.

Example:
  python plot_eval_by_sinr_all_in_two_panels.py \
    --root DATASET_FINAL/rf_datasets_h5/test/sinr_db \
    --out-dir DATASET_FINAL/rf_datasets_h5/test/evals_graphs \
    --datasets 64qam_wifi,256qam_wifi \
    --archs hybrid,time,spec \
    --dpi 160 --save-pdf \
    --dsir-ylim 0,60 --nmse-ylim -20,0
"""
import os, re, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DATASET_FINAL/rf_datasets_h5/test/sinr_db")
    ap.add_argument("--out-dir", default="DATASET_FINAL/rf_datasets_h5/test/evals_graphs")
    ap.add_argument("--datasets", default="", help="Comma-separated; if empty, auto-detect *_* folders")
    ap.add_argument("--archs", default="hybrid,time,spec")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--save-pdf", action="store_true")
    ap.add_argument("--nmse-ylim", default="", help="e.g. '-20,0'")
    ap.add_argument("--dsir-ylim", default="", help="e.g. '0,60'")
    return ap.parse_args()

def list_datasets(root, explicit):
    if explicit:
        return [d.strip() for d in explicit.split(",") if d.strip()]
    return sorted([n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)) and "_" in n])

def sinr_from_dirname(name):
    m = re.match(r"^sinr_([mp]?\d+(?:\.\d+)?)dB$", name)
    if not m: return None
    s = m.group(1).replace("m", "-").replace("p", "+")
    try: return float(s)
    except: return None

def collect_metrics(root, dataset, arch):
    base = os.path.join(root, dataset)
    rows = []
    if not os.path.isdir(base): return pd.DataFrame()
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if not os.path.isdir(p): continue
        sinr = sinr_from_dirname(name)
        if sinr is None: continue
        csv_path = os.path.join(p, "eval", arch, "metrics_by_sinr.csv")
        if not os.path.isfile(csv_path): continue
        df = pd.read_csv(csv_path)
        if df.empty: continue
        r = df.iloc[0]
        rows.append({
            "sinr": float(sinr),
            "delta_sir_db_mean": float(r.get("delta_sir_db_mean", np.nan)),
            "nmse_db_mean": float(r.get("nmse_db_mean", np.nan)),
        })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sinr").reset_index(drop=True)

def parse_ylim(text):
    if not text: return None
    try:
        a, b = text.split(","); return (float(a), float(b))
    except: return None

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = list_datasets(args.root, args.datasets)
    archs    = [a.strip() for a in args.archs.split(",") if a.strip()]
    if not datasets:
        print("[WARN] No datasets found."); return

    # Colors per model, markers per dataset
    arch_colors = {"hybrid":"tab:blue", "time":"tab:green", "spec":"tab:red"}
    ds_markers_cycle = ["o","s","^","D","v","P","*"]
    ds_marker = {ds: ds_markers_cycle[i % len(ds_markers_cycle)] for i, ds in enumerate(datasets)}

    # Collect all
    data = {(ds, arch): collect_metrics(args.root, ds, arch) for ds in datasets for arch in archs}

    # FIGURE: width auto-ish, height fixed to 7 as requested
    fig, (ax_dsir, ax_nmse) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    # Plot ΔSIR (solid) and NMSE (dashed) for every dataset×arch
    for ds in datasets:
        for arch in archs:
            df = data[(ds, arch)]
            if df.empty:
                print(f"[WARN] Missing data for {ds}/{arch}")
                continue
            x = df["sinr"].values
            y_d = df["delta_sir_db_mean"].values
            y_n = df["nmse_db_mean"].values
            color  = arch_colors.get(arch, "black")
            marker = ds_marker[ds]

            ax_dsir.plot(x, y_d, color=color, marker=marker, markersize=5,
                         linestyle="-", linewidth=1.9, label=f"{arch} · {ds}")
            ax_nmse.plot(x, y_n, color=color, marker=marker, markersize=5,
                         linestyle="--", linewidth=1.9, label=f"{arch} · {ds}")

    # Axes formatting
    ax_dsir.set_xlabel("SINR (dB)"); ax_nmse.set_xlabel("SINR (dB)")
    ax_dsir.set_ylabel("ΔSIR (dB)"); ax_nmse.set_ylabel("NMSE (dB)")
    ax_dsir.grid(True, alpha=0.3);   ax_nmse.grid(True, alpha=0.3)

    dsir_ylim = parse_ylim(args.dsir_ylim); nmse_ylim = parse_ylim(args.nmse_ylim)
    if dsir_ylim: ax_dsir.set_ylim(*dsir_ylim)
    if nmse_ylim: ax_nmse.set_ylim(*nmse_ylim)

    # Single legend ABOVE plots: show models (colors) and datasets (markers) clearly
    model_handles = [Patch(facecolor=arch_colors.get(a,"black"), edgecolor='none', label=a) for a in archs]
    ds_handles = [Line2D([0],[0], color="black", marker=ds_marker[d], linestyle="",
                         markersize=6, label=d.replace("_", " + ")) for d in datasets]
    legend = fig.legend(handles=(model_handles + ds_handles),
                        loc="upper center", ncol=max(3, len(archs)+len(datasets)),
                        bbox_to_anchor=(0.5, 1.02), frameon=False, title="Models (colors) & Datasets (markers)")

    fig.suptitle("ΔSIR and NMSE vs SINR — all models & datasets", y=1.08, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = os.path.join(args.out_dir, "combined_two_panels_sir_nmse_vs_sinr.png")
    fig.savefig(out_png, dpi=args.dpi)
    if args.save_pdf:
        out_pdf = os.path.join(args.out_dir, "combined_two_panels_sir_nmse_vs_sinr.pdf")
        fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[OK] {out_png}")

if __name__ == "__main__":
    main()
