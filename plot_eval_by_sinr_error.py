#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera 6 gráficas (2 datasets x 3 arquitecturas) con la evolución por SINR de:
- delta_sir_db_mean
- error_final_mean
- nmse_db_mean

Lee la estructura:
DATASET_FINAL/rf_datasets_h5/test/sinr_db/{dataset}/sinr_{XdB}/eval/{arch}/metrics_by_sinr.csv

Salida:
DATASET_FINAL/rf_datasets_h5/test/evals_graphs/{dataset}_{arch}_metrics_vs_sinr.png

Uso:
  python plot_eval_by_sinr.py \
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
                    help="Carpeta raíz con {dataset}/sinr_*dB/...")
    ap.add_argument("--out-dir", default="DATASET_FINAL/rf_datasets_h5/test/evals_graphs",
                    help="Carpeta de salida para las figuras")
    ap.add_argument("--datasets", default="", help="Lista coma-separada (p.ej. 64qam_wifi,256qam_wifi). "
                                                   "Si se omite, se detecta automáticamente (todas las carpetas *_*).")
    ap.add_argument("--archs", default="hybrid,time,spec", help="Arquitecturas a procesar (coma-separadas)")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--save-pdf", action="store_true")
    return ap.parse_args()

def list_datasets(root, explicit):
    if explicit:
        return [d.strip() for d in explicit.split(",") if d.strip()]
    # auto-detecta subcarpetas tipo *_*
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and "_" in name:
            out.append(name)
    return sorted(out)

def sinr_from_dirname(name: str):
    # Espera "sinr_{XdB}" -> extrae X como float/int
    m = re.match(r"^sinr_([mp]?\d+)(?:dB)?$", name)  # admite m10dB (neg) o 10dB
    if m:
        s = m.group(1).replace("m", "-").replace("p", "+")
        try:
            return float(s)
        except:
            return None
    # alternativa: "sinr_10db" o con decimales
    m = re.match(r"^sinr_(-?\d+(?:\.\d+)?)dB$", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None

def collect_metrics_for_dataset_arch(root, dataset, arch):
    """
    Recorre todas las carpetas sinr_*dB del dataset y lee metrics_by_sinr.csv (una fila).
    Devuelve DataFrame con columnas: sinr, error_final_mean, nmse_db_mean, delta_sir_db_mean
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
            # si no existe para este arch/SINR, salta
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        # coge primera fila (para este dataset/SINR hay una sola línea)
        r = df.iloc[0].to_dict()
        rows.append({
            "sinr": float(sinr),
            "error_final_mean": float(r.get("error_final_mean", np.nan)),
            "nmse_db_mean": float(r.get("nmse_db_mean", np.nan)),
            "delta_sir_db_mean": float(r.get("delta_sir_db_mean", np.nan)),
            "count": int(r.get("count", 0))
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("sinr").reset_index(drop=True)
    return out

def plot_dataset_arch(df, dataset, arch, out_dir, dpi=160, save_pdf=False):
    if df.empty:
        print(f"[WARN] No hay datos para {dataset} / {arch}")
        return
    x = df["sinr"].values

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    # 1) ΔSIR
    axes[0].plot(x, df["delta_sir_db_mean"].values, marker="o", linewidth=1.8, label="ΔSIR (dB)")
    axes[0].set_ylabel("ΔSIR (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    # 2) error_final (lineal; recuerda: no comparable entre archs)
    axes[1].plot(x, df["error_final_mean"].values, marker="s", linewidth=1.8, label="error_final (mean)")
    axes[1].set_ylabel("Error final")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    # 3) NMSE (dB)
    axes[2].plot(x, df["nmse_db_mean"].values, marker="^", linewidth=1.8, label="NMSE (dB)")
    axes[2].set_xlabel("SINR (dB)")
    axes[2].set_ylabel("NMSE (dB)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle(f"{dataset} — {arch}", fontsize=12)
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{dataset}_{arch}_metrics_vs_sinr.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png, dpi=dpi)
    if save_pdf:
        pdf = os.path.join(out_dir, f"{dataset}_{arch}_metrics_vs_sinr.pdf")
        fig.savefig(pdf)
    plt.close(fig)
    print(f"[OK] {png}")

def main():
    args = parse_args()
    datasets = list_datasets(args.root, args.datasets)
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    if not datasets:
        print("[WARN] No se encontraron datasets. Revisa --root o --datasets.")
        return
    os.makedirs(args.out-dir if False else args.out_dir, exist_ok=True)  # asegura directorio raíz

    # Para cada dataset y arch, compila y grafica
    for ds in datasets:
        for arch in archs:
            df = collect_metrics_for_dataset_arch(args.root, ds, arch)
            plot_dataset_arch(df, ds, arch, args.out_dir, dpi=args.dpi, save_pdf=args.save_pdf)

if __name__ == "__main__":
    main()
