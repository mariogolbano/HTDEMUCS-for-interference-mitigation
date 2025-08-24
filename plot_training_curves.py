#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera curvas de entrenamiento concatenadas por fases para cada arquitectura.
Lee CSVs con patrón: loss_{fase}_{arch}.csv dentro de un directorio.

Fases (orden estricto):
  qpsk_le, qpsk_br, 64qam_le, 64qam_br, 1024qam_le, 1024qam_br

Para cada arquitectura (hybrid, time/temp, spec) crea una figura con:
  - Eje X: epoch global (concatenando fases en el orden anterior)
  - Eje Y izq.: train_loss y val_loss (totales)
  - Eje Y der.: val_nmse_db
  - Separadores/marcas entre fases y etiquetas con el nombre de fase

Uso:
  python plot_training_curves.py --logs-dir /ruta/a/csvs --out-dir /ruta/salida
Opcional:
  --archs hybrid,time,spec     (para filtrar arquitecturas a procesar)
  --dpi 160
  --no-phase-shading           (desactiva sombreado de fases)
  --save-pdf                   (además del PNG guarda PDF)

  
  EJEMPLO:

  python plot_training_curves.py \
  --logs-dir ./spec \
  --out-dir ./curves \
  --archs spec \
  --dpi 160 \
  --save-pdf
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

PHASES = ["qpsk_le", "qpsk_br", "64qam_le", "64qam_br", "1024qam_le", "1024qam_br"]
ARCHES_ALL = ["hybrid", "time", "spec"]

# Sinónimos aceptados en nombres de archivo
# p.ej. si tus CSV usan "temp" en vez de "time", o "spectral" en vez de "spec"
ARCH_SYNONYMS = {
    "hybrid":  ["hybrid"],
    "time":    ["time", "temp"],
    "spec":    ["spec", "spectral"],
}

def read_phase_csv(logs_dir: str, phase: str, arch: str) -> pd.DataFrame:
    """
    Intenta leer loss_{phase}_{arch}.csv admitiendo sinónimos del arch.
    Exige columnas: epoch, train_loss, val_loss, val_nmse_db
    """
    tried = []
    for a in ARCH_SYNONYMS.get(arch, [arch]):
        fname = f"loss_{phase}_{a}.csv"
        path = os.path.join(logs_dir, fname)
        tried.append(path)
        if os.path.exists(path):
            df = pd.read_csv(path)
            required = {"epoch","train_loss","val_loss","val_nmse_db"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Faltan columnas {missing} en {path}")
            df = df.copy()
            df["phase"] = phase
            df["arch"] = arch
            return df
    raise FileNotFoundError("No se encontró ninguno de: " + ", ".join(tried))

def concat_with_global_epoch(dfs):
    """Añade 'global_epoch' concatenando por fases en orden PHASES."""
    out = []
    offset = 0
    for phase in PHASES:
        dps = [d for d in dfs if d["phase"].iloc[0] == phase]
        if not dps:
            continue
        if len(dps) != 1:
            raise ValueError(f"Fase duplicada {phase}")
        d = dps[0].copy()
        d["epoch"] = d["epoch"].astype(int)
        d["global_epoch"] = d["epoch"] + offset
        out.append(d)
        last_ep = int(d["epoch"].iloc[-1])
        offset += last_ep
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=0, ignore_index=True)

def plot_one_arch(df: pd.DataFrame, arch: str, out_dir: str, dpi: int = 160, phase_shading: bool = True, save_pdf: bool = False):
    if df.empty:
        print(f"[WARN] No hay datos para arch={arch}.")
        return

    x = df["global_epoch"].values
    y_tr = df["train_loss"].values
    y_va = df["val_loss"].values
    y_nm = df["val_nmse_db"].values

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ln1, = ax.plot(x, y_tr, label="train_loss")
    ln2, = ax.plot(x, y_va, label="val_loss")
    ax.set_xlabel("Epoch (concatenated)")
    ax.set_yscale("log") 
    ax.set_ylim(0.001, 1)   # ← límites fijos para la loss
    ax.set_ylabel("Loss")

    ax2 = ax.twinx()
    ln3, = ax2.plot(x, y_nm, linestyle=":", label="val_nmse_db")   
    ax2.set_ylim(-25, 10)      # ← límites fijos para NMSE 
    ax2.set_ylabel("NMSE (dB)")
    # Calcular límites de cada fase en eje global
    bounds = []
    for phase in PHASES:
        d = df[df["phase"] == phase]
        if d.empty:
            continue
        start = int(d["global_epoch"].iloc[0])
        end = int(d["global_epoch"].iloc[-1])
        bounds.append((phase, start, end))

    # Sombreado por fase y líneas verticales entre fases
    if phase_shading:
        for i, (phase, start, end) in enumerate(bounds):
            ax.axvspan(start, end, alpha=0.08)
    for i in range(len(bounds)-1):
        _, _, end = bounds[i]
        ax.axvline(end, linestyle="--", alpha=0.5)

    # Etiquetas con nombre de fase encima
    try:
        ymin, ymax = ax.get_ylim()
        y_lbl = ymax - 0.06 * (ymax - ymin)   # ~6% por debajo del techo
        for phase, start, end in bounds:
            mid = (start + end) / 2.0
            ax.text(mid, y_lbl, phase, ha="center", va="top", fontsize=9)
    except Exception:
        pass

    # Leyenda combinada (dos ejes)
    lines = [ln1, ln2, ln3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best", ncol=3)

    ax.grid(True, alpha=0.3)
    ax.set_title(f"Training curves (arch={arch})")

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"training_curves_{arch}.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    if save_pdf:
        pdf_path = os.path.join(out_dir, f"training_curves_{arch}.pdf")
        fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Guardado {png_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", required=True, help="Carpeta con loss_{fase}_{arch}.csv")
    ap.add_argument("--out-dir", required=True, help="Carpeta de salida para las figuras")
    ap.add_argument("--archs", default=",".join(ARCHES_ALL),
                    help="Lista separada por comas de arquitecturas a procesar (default: hybrid,time,spec)")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--no-phase-shading", action="store_true")
    ap.add_argument("--save-pdf", action="store_true")
    args = ap.parse_args()

    archs_in = [a.strip() for a in args.archs.split(",") if a.strip()]
    # Normalizar sinónimos en la lista de arquitecturas pedidas
    normalized_archs = []
    for a in archs_in:
        if a in ARCH_SYNONYMS:
            normalized_archs.append(a)
        elif a == "temp":
            normalized_archs.append("time")
        elif a == "spectral":
            normalized_archs.append("spec")
        else:
            normalized_archs.append(a)

    for arch in normalized_archs:
        # leer los 6 CSV en orden estricto
        dfs = []
        for ph in PHASES:
            try:
                dfp = read_phase_csv(args.logs_dir, ph, arch)
                dfs.append(dfp)
            except FileNotFoundError as e:
                print(f"[WARN] {e}; se ignora esta fase para arch={arch}.")
            except Exception as e:
                print(f"[WARN] Problema con {ph}/{arch}: {e}; se ignora.")
        if not dfs:
            print(f"[WARN] No hay CSVs válidos para arch={arch}.")
            continue
        df_all = concat_with_global_epoch(dfs)
        plot_one_arch(df_all, arch, args.out_dir, dpi=args.dpi, phase_shading=(not args.no_phase_shading), save_pdf=args.save_pdf)

if __name__ == "__main__":
    main()
