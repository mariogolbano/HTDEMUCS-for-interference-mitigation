#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia single-target (HT-Demucs con stems=1) con evaluación y partición por SINR.

- Entrada obligatoria:
  * --in-h5       : HDF5 con 'dataset' (N, 2, T) [mixtura I/Q]
  * --ckpt        : checkpoint entrenado (puede o no contener cfg)
  * --out-dir     : carpeta de salida

- Entrada para evaluación (opcional pero recomendada):
  * --target-h5   : HDF5 con 'dataset' (N, 2, T) de la señal objetivo s (ground truth)
  * --interf-h5   : HDF5 con 'dataset' (N, 2, T) de la(s) interferencia(s) i (gt) para ΔSIR

- Cómo detecta SINR:
  * Si en in-h5 existe dataset 'sinr_db', usa ese valor por item (creado por create_rf_dataset.py).
  * Si no existe 'sinr_db' pero se pasan --target-h5 y --interf-h5, calcula 10*log10(||s||^2/||i||^2).

- Salidas en --out-dir:
  * <stem>_separated.h5            -> todas las estimaciones concatenadas (como antes).
  * metrics_by_sinr.csv            -> tabla agregada por SINR (error, NMSE, ΔSIR).
  * metrics_items.csv              -> métricas por item.
  * sinr_<v>_dB/                   -> subcarpetas por SINR con:
       mixture.h5, target_true.h5, interference_true.h5, separated.h5 y metrics.csv

- Métricas por item:
  * 'error_final':
      - arch==hybrid -> 0.5*L1 + 0.5*MR-STFT (como train)
      - arch==time   -> MSE
      - arch==spec   -> MR-STFT
  * 'nmse_db'  = 10*log10(||ŷ - s||^2 / ||s||^2)
  * 'delta_sir_db' = SIR_out - SIR_in, donde SIR_out computa la fuga de interferencia en la salida
                     proyectando el residuo r = ŷ - s sobre i (o el subespacio de {i_k}).

Nota: no modifica el formato del HDF5 de salida principal ni rompe comandos existentes.
"""
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import h5py
import numpy as np
from types import SimpleNamespace
import torch
from dataclasses import replace
from tqdm.auto import tqdm

from models.htdemucs import HTDemucs
from config_from_json import load_config_from_base
from presets import htdemucs_preset

EPS = 1e-8

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ------------------------- Utils HDF5 -------------------------

def read_h5_framesize(path: str):
    """Lee el atributo 'framesize' del H5 de entrada (raíz o dataset)."""
    try:
        with h5py.File(path, "r") as f:
            if "framesize" in f.attrs:
                try:
                    return int(f.attrs["framesize"])
                except Exception:
                    pass
            if "dataset" in f:
                ds = f["dataset"]
                if "framesize" in ds.attrs:
                    try:
                        return int(ds.attrs["framesize"])
                    except Exception:
                        pass
    except Exception:
        pass
    return None


def _sanitize(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)
    return safe or "target"


def read_optional(ds: h5py.File, key: str):
    if key in ds:
        return np.array(ds[key])
    return None


# ------------------------- Carga de modelo -------------------------

def _cfg_from_ckpt_or_base(ckpt: dict, base_dir: str | None, arch_cli: str | None):
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        cfgd = ckpt["cfg"].copy()
        # fuerza stems=1 por seguridad
        cfgd["stems"] = 1
        # si el ckpt no trae 'arch', usamos CLI; si trae, gana el ckpt
        if "arch" not in cfgd and arch_cli is not None:
            cfgd["arch"] = arch_cli
        return SimpleNamespace(**cfgd)
    if base_dir is None:
        raise ValueError("El checkpoint no contiene 'cfg'. Proporciona --base con mixture.json, o usa un ckpt reciente.")
    cfg_base = load_config_from_base(base_dir)
    cfg = htdemucs_preset(cfg_base)
    cfg = replace(cfg, stems=1)
    # adjunta arch desde CLI o 'hybrid' por defecto
    return SimpleNamespace(**{**cfg.__dict__, "arch": (arch_cli or "hybrid")})


def load_model_single(ckpt_path: str, device: torch.device, base_dir: str | None, arch_cli: str | None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = _cfg_from_ckpt_or_base(ckpt, base_dir, arch_cli)
    segment_len = int(cfg.segment_seconds * cfg.sample_rate)

    model = HTDemucs(
        in_channels=cfg.in_channels,
        stems=1,
        audio_channels=cfg.in_channels,
        n_fft=cfg.n_fft,
        depth=cfg.depth,
        base_channels=cfg.base_channels,
        transformer_dim=cfg.transformer_dim,
        transformer_heads=cfg.transformer_heads,
        transformer_layers=cfg.transformer_layers,
        segment_length=segment_len,
        arch=getattr(cfg, "arch", "hybrid"),
    ).to(device)

    state = ckpt["model"]
    # tolera pequeñas diferencias (por ejemplo si el ckpt es previo a este cambio)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, cfg, segment_len


# ------------------------- Pérdidas y métricas -------------------------

def _stft_mag_batch(x_1d: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    # x_1d: (B,T) en float32/bf16 on device
    win = torch.hann_window(n_fft, device=x_1d.device, dtype=torch.float32)
    X = torch.stft(x_1d.to(torch.float32), n_fft=n_fft, hop_length=hop,
                   window=win, center=True, normalized=True, return_complex=True)
    return torch.abs(X)

def mrstft_loss_like_train(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Replica la MR-STFT de train: magnitud, 3 FFTs {256,512,1024}, media de L1 espectral.
    y_*: (B,1,2,T)
    """
    B, S, C, T = y_hat.shape
    mono_hat  = y_hat.reshape(B * S, C, T).mean(1)
    mono_true = y_true.reshape(B * S, C, T).mean(1)
    loss = 0.
    for nfft in (256, 512, 1024):
        hop = nfft // 4
        loss = loss + torch.mean(torch.abs(
            _stft_mag_batch(mono_hat, nfft, hop) - _stft_mag_batch(mono_true, nfft, hop)))
    return loss / 3.0

def l1_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_hat - y_true))

def mse_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y_true) ** 2)

def nmse_db(y_hat_np: np.ndarray, y_true_np: np.ndarray) -> float:
    # y_*: (2,T) numpy
    num = float(np.sum((y_hat_np - y_true_np) ** 2) + 1e-12)
    den = float(np.sum(y_true_np ** 2) + 1e-12)
    return 10.0 * np.log10(num / den)

def project_leak_on_interf(residual: np.ndarray, interf: np.ndarray) -> np.ndarray:
    """
    Proyecta el residuo sobre la(s) interferencia(s) para medir fuga.
    residual: (2,T). interf: (2,T) o (K,2,T).
    Devuelve la fuga reconstruida con igual shape que residual.
    """
    r = residual.reshape(-1, 1).astype(np.float64, copy=False)  # (2T,1)
    if interf.ndim == 2:
        i = interf.reshape(-1, 1).astype(np.float64, copy=False)  # (2T,1)
        denom = float(np.sum(i * i) + 1e-12)
        alpha = float((i.T @ r) / denom)
        leak = (i * alpha).reshape(2, -1)
        return leak.astype(np.float32)
    else:
        K = interf.shape[0]
        I = interf.reshape(K, -1).transpose(1, 0).astype(np.float64, copy=False)  # (2T,K)
        alpha, *_ = np.linalg.lstsq(I, r, rcond=None)  # (K,1)
        leak = (I @ alpha).reshape(2, -1)
        return leak.astype(np.float32)

def sir_db(sig: np.ndarray, interf: np.ndarray) -> float:
    num = float(np.sum(sig * sig) + 1e-12)
    den = float(np.sum(interf * interf) + 1e-12)
    return 10.0 * np.log10(num / den)


# ------------------------- Overlap-add -------------------------

def build_xfade_window(L: int) -> torch.Tensor:
    """Ventana Hann [0..1..0] exacta de longitud L (incluye el 0 final)."""
    t = np.linspace(0.0, np.pi, L, endpoint=True, dtype=np.float32)
    return torch.from_numpy(0.5 * (1.0 - np.cos(t)))


@torch.no_grad()
def separate_one(example_mix: torch.Tensor,
                 model: torch.nn.Module,
                 device: torch.device,
                 segment_len: int,
                 overlap: float = 0.5,
                 use_bf16: bool = True) -> torch.Tensor:
    """
    example_mix: (2, T) tensor CPU/GPU
    Devuelve: (1, 2, T) en CPU float32 (un único stem).
    """
    C, T = example_mix.shape
    assert C == 2, "Se esperan 2 canales (I/Q)."

    orig_T = T  # guardamos longitud original para recortar luego

    # Si usamos rama espectral (arch != 'time'), aseguremos al menos 8 frames STFT
    need_spec = getattr(model, "arch", "hybrid") != "time"
    if need_spec and hasattr(model, "stft"):
        # hop y n_fft desde el STFT del modelo (fallback a n_fft//4)
        nfft = int(getattr(model.stft, "n_fft", 4096))
        hop  = int(getattr(model.stft, "hop", nfft // 4))
        min_frames = 8  # porque el primer conv2d usa kernel temporal de 8
        T_needed = min_frames * hop
        if T < T_needed:
            pad = T_needed - T
            # Pad al final (en tiempo) con ceros, sin afectar el RMS que calculamos más abajo
            example_mix = torch.cat([example_mix, example_mix.new_zeros(C, pad)], dim=1)
            T = example_mix.shape[-1]

    # Normalización RMS (como train normalize='rms')
    rms = torch.sqrt((example_mix**2).mean() + EPS).item()
    x = example_mix / (rms + EPS)

    L = int(segment_len)
    if L >= T:
        starts = [0]
        L_eff = T
    else:
        stride = max(1, int(L * (1.0 - overlap)))
        starts = list(range(0, max(1, T - L + 1), stride))
        if starts[-1] + L < T:
            starts.append(T - L)
        L_eff = L

    win = build_xfade_window(L_eff).to(device)

    x = x.to(device, non_blocking=True).contiguous()

    out_sum = torch.zeros((1, 2, orig_T), device=device, dtype=torch.float32)
    weight  = torch.zeros((orig_T,), device=device, dtype=torch.float32)

    autocast_ctx = torch.amp.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and use_bf16)
    )

    with autocast_ctx:
        for s0 in starts:
            s1 = s0 + L_eff
            seg = x[:, s0:s1].unsqueeze(0)        # (1,2,L)
            y_seg = model(seg)                    # (1,1,2,L)
            y_seg = y_seg.squeeze(0).to(torch.float32)  # (1,2,L)

            # recorte por si L_eff > orig_T (caso con padding)
            wlen = min(L_eff, orig_T - s0)
            if wlen > 0:
                out_sum[:, :, s0:s0+wlen] += y_seg[:, :, :wlen] * win[:wlen].view(1, 1, -1)
                weight[s0:s0+wlen] += win[:wlen]

    weight = torch.clamp(weight, min=1e-6)
    out = out_sum / weight.view(1, 1, -1)

    # Desnormaliza
    out = out * (rms + EPS)
    return out.cpu()  # (1,2,T)


# ------------------------- Lote H5 -> un H5 + evaluación -------------------------

def _prepare_group_outputs(out_root: Path, tag: str, C: int, T: int, M: int, framesize: Optional[int], save_interf: bool = True):
    """
    Crea carpeta sinr_<tag>/ con 4 H5: mixture, target_true, interference_true (si procede) y separated.
    Devuelve dict con file/dataset handles para escritura.
    """
    group_dir = out_root / f"sinr_{tag}"
    group_dir.mkdir(parents=True, exist_ok=True)
    def _mk(name):
        f = h5py.File(group_dir / f"{name}.h5", "w")
        ds = f.create_dataset("dataset", shape=(M, C, T), dtype="float32",
                              chunks=(1, C, min(T, 1 << 20)), compression="gzip", compression_opts=4)
        if framesize is not None:
            try:
                f.attrs["framesize"] = int(framesize); ds.attrs["framesize"] = int(framesize)
            except Exception:
                pass
        return f, ds
    f_mix, ds_mix = _mk("mixture")
    f_tar, ds_tar = _mk("target_true")
    f_sep, ds_sep = _mk("separated")
    if save_interf:
        f_int, ds_int = _mk("interference_true")
        return {"dir": group_dir, "mix": (f_mix, ds_mix), "tar": (f_tar, ds_tar), "sep": (f_sep, ds_sep), "int": (f_int, ds_int)}
    else:
        return {"dir": group_dir, "mix": (f_mix, ds_mix), "tar": (f_tar, ds_tar), "sep": (f_sep, ds_sep)}


def run_h5_single_to_file(input_h5: str,
                          output_h5: str,
                          ckpt_path: str,
                          base_dir: str | None,
                          device_str: str = "cuda",
                          overlap: float = 0.5,
                          batch_index_from: int | None = None,
                          batch_index_to: int | None = None,
                          use_bf16: bool = True,
                          stem_name: str = "target",
                          arch_cli: str | None = None,
                          target_h5: Optional[str] = None,
                          interf_h5: Optional[str] = None,
                          sinr_edges: Optional[List[float]] = None):
    """
    Lee 'dataset' de input_h5 (N,2,T) -> escribe UN archivo H5 de salida con 'dataset' -> (M,2,T) (legacy)
    Si se proporcionan --target-h5/--interf-h5, calcula métricas por item y agrega por SINR.
    También crea subcarpetas sinr_<v>_dB con H5s por grupo SINR.
    """
    out_path = Path(output_h5).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model, cfg, segment_len = load_model_single(ckpt_path, device, base_dir, arch_cli)
    arch = getattr(cfg, "arch", "hybrid")

    framesize_in = read_h5_framesize(input_h5)
    stem_name = _sanitize(stem_name)

    print(f"[INFO] ckpt={ckpt_path}")
    print(f"[INFO] arch={arch}")
    print(f"[INFO] segment_len={segment_len}  overlap={overlap}  device={device}  bf16={use_bf16}")
    print(f"[INFO] framesize(input)={framesize_in!r}")
    print(f"[INFO] output={out_path}  stem_name={stem_name}")

    # --------- abre H5 ---------
    fi = h5py.File(input_h5, "r")
    ds_in = fi["dataset"]  # (N,2,T)
    N, C, T = ds_in.shape
    assert C == 2, f"Se esperaban 2 canales (I/Q), recibido C={C}"
    i0 = batch_index_from or 0
    i1 = batch_index_to if batch_index_to is not None else N
    i0 = max(0, i0); i1 = min(N, i1)
    M_total = max(0, i1 - i0)

    # targets e interferencias (opcionales)
    ft = It = None
    ds_tar = ds_int = None
    if target_h5 is not None:
        ft = h5py.File(target_h5, "r"); ds_tar = ft["dataset"]
        assert ds_tar.shape == ds_in.shape, "target_h5 shape distinto al de in_h5"
    if interf_h5 is not None:
        It = h5py.File(interf_h5, "r"); ds_int = It["dataset"]
        assert ds_int.shape == ds_in.shape, "interf_h5 shape distinto al de in_h5"

    # sinr por item: desde in_h5 o estimado
    sinr_db_arr = read_optional(fi, "sinr_db")
    if sinr_db_arr is None and (ds_tar is not None and ds_int is not None):
        print("[INFO] 'sinr_db' no encontrado. Se estimará por item a partir de --target-h5/--interf-h5.")
        # compute for slice
        sinr_db_arr = np.empty((N,), dtype=np.float32)
        for k in range(N):
            s = ds_tar[k]; i = ds_int[k]
            sinr_db_arr[k] = sir_db(s, i)
    if sinr_db_arr is None:
        print("[WARN] No hay 'sinr_db' ni --target/--interf. No se podrá agregar por SINR ni calcular ΔSIR.")
    else:
        sinr_db_arr = np.array(sinr_db_arr).astype(np.float32)

    # --------- dataset de salida "legacy" (todo junto) ---------
    f_out = h5py.File(out_path, "w")
    ds_out = f_out.create_dataset(
        "dataset", shape=(M_total, C, T), dtype="float32",
        chunks=(1, C, min(T, 1 << 20)), compression="gzip", compression_opts=4,
    )
    ds_out.attrs["stem_name"] = stem_name.encode("utf-8")
    f_out.attrs["source_mixture_h5"] = str(Path(input_h5).resolve())
    f_out.attrs["checkpoint"] = str(Path(ckpt_path).resolve())
    f_out.attrs["segment_len"] = int(segment_len)
    f_out.attrs["overlap"] = float(overlap)
    if framesize_in is not None:
        f_out.attrs["framesize"] = int(framesize_in)
        ds_out.attrs["framesize"] = int(framesize_in)

    print(f"[INFO] input {input_h5} shape=(N={N}, C={C}, T={T}), running [{i0}:{i1}) -> {M_total} items")

    # --------- pre-binning por SINR ---------
    out_root = out_path.parent
    groups: Dict[str, Dict] = {}
    group_counts: Dict[str, int] = {}
    group_positions: Dict[str, int] = {}
    item_to_group: Dict[int, str] = {}

    def label_from_value(v: float) -> str:
        # etiqueta bonita tipo "-15dB" con 0 decimales
        return f"{int(round(v))}dB".replace("-", "m").replace("+", "p")

    if sinr_db_arr is not None:
        if sinr_edges is None:
            # usa valores discretos presentes (redondeados a 0.1 dB), luego etiqueta al entero más cercano
            vals = np.array(sorted(set([float(f"{x:.1f}") for x in sinr_db_arr.tolist()])), dtype=np.float32)
            edges = vals
        else:
            edges = np.array(sinr_edges, dtype=np.float32)
        # asigna grupo por valor más cercano en edges
        for idx in range(i0, i1):
            v = float(sinr_db_arr[idx])
            j = int(np.argmin(np.abs(edges - v)))
            lbl = label_from_value(edges[j])
            item_to_group[idx] = lbl
            group_counts[lbl] = group_counts.get(lbl, 0) + 1

        # crea H5 por grupo
        for lbl, cnt in group_counts.items():
            groups[lbl] = _prepare_group_outputs(out_root, lbl, C, T, cnt, framesize_in, save_interf=(ds_int is not None))
            group_positions[lbl] = 0

    # --------- buffers métricas por item ---------
    metrics = []  # dict por item
    t0 = time.time()

    # --------- loop ---------
    for j, idx in enumerate(tqdm(range(i0, i1), desc="Infer (items)", dynamic_ncols=True)):
        mix_np = ds_in[idx]  # (2,T) float
        x = torch.from_numpy(mix_np.astype(np.float32, copy=False))  # (2,T)
        y = separate_one(x, model, device, segment_len, overlap=overlap, use_bf16=use_bf16)  # (1,2,T)
        y_np = y[0].numpy()  # (2,T)

        # escribe en salida legacy
        ds_out[j, :, :] = y_np

        # métricas si hay gt
        item_metrics = {"index": int(idx)}
        if ds_tar is not None:
            s_np = np.asarray(ds_tar[idx]).astype(np.float32, copy=False)
            item_metrics["nmse_db"] = float(nmse_db(y_np, s_np))
            # error_final según arch
            if arch == "hybrid":
                with torch.no_grad():
                    y_t = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0)  # (1,1,2,T)
                    s_t = torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0)
                    ef = 0.5 * l1_loss(y_t, s_t) + 0.5 * mrstft_loss_like_train(y_t, s_t)
                    item_metrics["error_final"] = float(ef.cpu().item())
                    item_metrics["error_kind"] = "0.5*L1+0.5*MRSTFT"
            elif arch == "time":
                # MSE tal como has pedido para evaluación de temporal pura
                item_metrics["error_final"] = float(np.mean((y_np - s_np) ** 2))
                item_metrics["error_kind"] = "MSE"
            else:  # spec
                with torch.no_grad():
                    y_t = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0)
                    s_t = torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0)
                    ef = mrstft_loss_like_train(y_t, s_t)
                    item_metrics["error_final"] = float(ef.cpu().item())
                    item_metrics["error_kind"] = "MRSTFT"

            # ΔSIR si hay interferencia
            if ds_int is not None:
                i_np = np.asarray(ds_int[idx]).astype(np.float32, copy=False)
                sir_in = sir_db(s_np, i_np)
                leak = project_leak_on_interf(y_np - s_np, i_np)
                sir_out = sir_db(s_np, leak)
                item_metrics["sir_in_db"] = float(sir_in)
                item_metrics["sir_out_db"] = float(sir_out)
                item_metrics["delta_sir_db"] = float(sir_out - sir_in)

        # asigna a grupo SINR si procede (y guarda H5 por grupo)
        if sinr_db_arr is not None and (idx in item_to_group):
            lbl = item_to_group[idx]
            pos = group_positions[lbl]
            g = groups[lbl]
            # mixture / target / separated (+ interferencia si hay)
            g["mix"][1][pos, :, :] = mix_np.astype(np.float32, copy=False)
            if ds_tar is not None:
                g["tar"][1][pos, :, :] = s_np
            if ds_int is not None:
                g["int"][1][pos, :, :] = i_np
            g["sep"][1][pos, :, :] = y_np
            group_positions[lbl] += 1

        metrics.append(item_metrics)

    t1 = time.time()
    print(f"[OK] Wrote {out_path}  ({M_total} items). Elapsed={t1 - t0:.1f}s")

    # --------- cierra H5 de grupos y escribe métricas CSV ---------
    import csv
    # per-item CSV
    csv_items = out_path.parent / "metrics_items.csv"
    keys = sorted({k for m in metrics for k in m.keys()})
    with open(csv_items, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=keys)
        w.writeheader()
        for m in metrics:
            w.writerow(m)
    print(f"[OK] Guardado {csv_items}")

    # agrega por SINR
    if sinr_db_arr is not None:
        # cierra archivos por grupo
        for lbl, handles in groups.items():
            # metrics por grupo
            rows = [m for m in metrics if (m.get("index") in item_to_group and item_to_group[m["index"]] == lbl)]
            csv_grp = handles["dir"] / "metrics.csv"
            if rows:
                keys_g = sorted({k for m in rows for k in m.keys()})
                with open(csv_grp, "w", newline="") as fcsv:
                    w = csv.DictWriter(fcsv, fieldnames=keys_g); w.writeheader()
                    for m in rows: w.writerow(m)
            for k in ("mix","tar","sep","int"):
                if k in handles:
                    f, ds = handles[k]
                    try: ds.flush()
                    finally: f.close()

        # tabla agregada
        def _agg(vals: List[float]):
            a = np.array(vals, dtype=np.float64)
            return float(np.mean(a)), float(np.median(a)), float(np.std(a))

        csv_by = out_path.parent / "metrics_by_sinr.csv"
        with open(csv_by, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["sinr_label","count","error_final_mean","error_final_median","error_final_std",
                        "nmse_db_mean","nmse_db_median","nmse_db_std",
                        "delta_sir_db_mean","delta_sir_db_median","delta_sir_db_std"])
            labels_sorted = sorted(group_counts.keys(), key=lambda s: int(s.replace("m","-").replace("p","+").rstrip("dB")))
            for lbl in labels_sorted:
                rows = [m for m in metrics if (m.get("index") in item_to_group and item_to_group[m["index"]] == lbl)]
                ef_vals   = [m["error_final"] for m in rows if "error_final" in m]
                nmse_vals = [m["nmse_db"] for m in rows if "nmse_db" in m]
                dsir_vals = [m["delta_sir_db"] for m in rows if "delta_sir_db" in m]
                ef = _agg(ef_vals) if ef_vals else (np.nan, np.nan, np.nan)
                nm = _agg(nmse_vals) if nmse_vals else (np.nan, np.nan, np.nan)
                dsir = _agg(dsir_vals) if dsir_vals else (np.nan, np.nan, np.nan)
                w.writerow([lbl, len(rows), *ef, *nm, *dsir])
        print(f"[OK] Guardado {csv_by}")

    # --------- cierra archivos raíz ---------
    try: ds_out.flush()
    finally: f_out.close()
    if fi is not None: fi.close()
    if ft is not None: ft.close()
    if It is not None: It.close()


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="RF single-target inference -> one H5 + evaluación por SINR.")
    ap.add_argument("--ckpt", required=True, help="Ruta al checkpoint .pt")
    ap.add_argument("--in-h5", required=True, help="HDF5 con 'dataset' (N,2,T)")
    ap.add_argument("--out-dir", required=True, help="Directorio de salida")
    ap.add_argument("--base", default=None, help="Carpeta con mixture.json (si el ckpt no tiene 'cfg')")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--idx-from", type=int, default=None)
    ap.add_argument("--idx-to", type=int, default=None)
    ap.add_argument("--stem-name", default="target")
    ap.add_argument("--no-bf16", action="store_true")
    # NUEVO: arch opcional para forzar si el ckpt no trae cfg/arch
    ap.add_argument("--arch", choices=["hybrid","time","spec"], default=None,
                    help="Forzar arquitectura si el ckpt no trae 'cfg.arch'.")
    # evaluación
    ap.add_argument("--target-h5", default=None, help="H5 con ground truth del target (para métricas).")
    ap.add_argument("--interf-h5", default=None, help="H5 con interferencia(s) (para ΔSIR y guardado por SINR).")
    ap.add_argument("--sinr-edges", default=None,
                    help="Coma-separado con valores de SINR (dB) para agrupar (p.ej. '-30,-25,...,0'). "
                         "Si no se da, se usan los presentes en 'sinr_db' del H5 (redondeados).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = out_dir / f"{_sanitize(args.stem_name)}_separated.h5"

    edges = None
    if args.sinr_edges:
        edges = [float(x.strip()) for x in args.sinr_edges.split(",") if x.strip() != ""]

    run_h5_single_to_file(
        input_h5=args.in_h5,
        output_h5=str(out_h5),
        ckpt_path=args.ckpt,
        base_dir=args.base,
        device_str=args.device,
        overlap=args.overlap,
        batch_index_from=args.idx_from,
        batch_index_to=args.idx_to,
        use_bf16=(not args.no_bf16),
        stem_name=args.stem_name,
        arch_cli=args.arch,
        target_h5=args.target_h5,
        interf_h5=args.interf_h5,
        sinr_edges=edges,
    )


if __name__ == "__main__":
    main()
