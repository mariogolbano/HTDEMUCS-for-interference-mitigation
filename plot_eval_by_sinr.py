#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ΔSIR and NMSE vs SINR on the SAME chart (dual y-axes), for each dataset × architecture.

Reads:
  DATASET_FINAL/rf_datasets_h5/test/sinr_db/{dataset}/sinr_{XdB}/eval/{arch}/metrics_by_sinr.csv

Outputs one PNG (and optional PDF) per dataset×arch to:
  DATASET_FINAL/rf_datasets_h5/test/evals_graphs/{dataset}_{arch}_sir_nmse_vs_sinr.png

Usage:
  python plot_eval_by_sinr_dual.py \
    --root DATASET_FINAL/rf_datasets_h5/test/sinr_db \
    --out-dir DATASET_FINAL/rf_datasets_h5/test/evals_graphs \
    --datasets 64qam_wifi,256qam_wifi \
    --archs hybrid,time,spec \
    --dpi 160 \
    --save-pdf
"""
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DATASET_FINAL/rf_datasets_h5/test/sinr_db",
                    help="Root with {dataset}/sinr_*dB/...")
    ap.add_argument("--out-dir", default="DATASET_FINAL/rf_datasets_h5/test/evals_graphs",
                    help="Output folder for figures")
    ap.add_argument("--datasets", default="",
                    help="Comma-separated (e.g., 64qam_wifi,256qam_wifi). If empty: auto-detect.")
    ap.add_argument("--archs", default="hybrid,time,spec",
                    help="Comma-separated architectures to plot")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--save-pdf", action="store_true")
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
    # expects "sinr_{XdB}"
    m = re.match(r"^sinr_([mp]?\d+(?:\.\d+)?)dB$", name)
    if not m: return None
    s = m.group(1).replace("m", "-").replace("p", "+")
    try:
        return float(s)
    except:
        return None

def collect_metrics_for_dataset_arch(root, dataset, arch):
    """
    Scan all sinr_*dB folders and read metrics_by_sinr.csv (one-row summary).
    Returns DataFrame with: sinr, nmse_db_mean, delta_sir_db_mean, count
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
        r = df.iloc[0].to_dict()
        rows.append({
            "sinr": float(sinr),
            "nmse_db_mean": float(r.get("nmse_db_mean", np.nan)),
            "delta_sir_db_mean": float(r.get("delta_sir_db_mean", np.nan)),
            "count": int(r.get("count", 0))
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("sinr").reset_index(drop=True)
    return out

def plot_dual(df, dataset, arch, out_dir, dpi=160, save_pdf=False):
    if df.empty:
        print(f"[WARN] No data for {dataset} / {arch}")
        return

    x = df["sinr"].values
    y_dsir = df["delta_sir_db_mean"].values
    y_nmse = df["nmse_db_mean"].values

    fig, ax_left = plt.subplots(figsize=(10, 4.8))
    # Left axis: ΔSIR (dB)
    l1, = ax_left.plot(x, y_dsir, marker="o", linewidth=1.8, label="ΔSIR (dB)")
    ax_left.set_xlabel("SINR (dB)")
    ax_left.set_ylabel("ΔSIR (dB)")
    ax_left.grid(True, alpha=0.3)

    # Right axis: NMSE (dB)
    ax_right = ax_left.twinx()
    l2, = ax_right.plot(x, y_nmse, marker="^",color="tab:orange", linewidth=1.8, label="NMSE (dB)")
    ax_right.set_ylabel("NMSE (dB)")

    ax_right.set_ylim(-9, 6)   # ← límites fijos para la loss
    ax_left.set_ylim(25, 50)   # ← límites fijos para la loss

    # Combined legend
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax_left.legend(lines, labels, loc="best")

    # Short descriptive English title
    fig.suptitle(f"{dataset} · {arch} — ΔSIR & NMSE vs SINR", fontsize=12)

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{dataset}_{arch}_sir_nmse_vs_sinr.png")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(png, dpi=dpi)
    if save_pdf:
        pdf = os.path.join(out_dir, f"{dataset}_{arch}_sir_nmse_vs_sinr.pdf")
        fig.savefig(pdf)
    plt.close(fig)
    print(f"[OK] {png}")

def main():
    args = parse_args()
    datasets = list_datasets(args.root, args.datasets)
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    if not datasets:
        print("[WARN] No datasets found. Check --root or --datasets.")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    for ds in datasets:
        for arch in archs:
            df = collect_metrics_for_dataset_arch(args.root, ds, arch)
            plot_dual(df, ds, arch, args.out_dir, dpi=args.dpi, save_pdf=args.save_pdf)

if __name__ == "__main__":
    main()
