#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create an RF dataset by mixing two generic HDF5 signals (I/Q) with metadata.

- Inputs:
  * --sig1 /path/to/signal1.h5  (expects dataset "dataset" with shape (N, 2, T), float or convertible)
  * --sig2 /path/to/signal2.h5
  * For each .h5 there must be a sibling JSON file with the same stem in the same directory:
      /path/to/signal1.json
      /path/to/signal2.json
    Each JSON must include at least: {"fs": <sampling frequency in Hz>}.

- Process:
  * Read fs for each signal from its JSON.
  * Select a common output sampling rate: fs_out = max(fs_sig1, fs_sig2)  (policy to preserve bandwidth).
  * Resample whichever signal has fs != fs_out (prints what was resampled to what).
  * Trim both signals to the common minimum length AFTER resampling.
  * Create mixture = sig1 + a * sig2, where a is --attenuation (default 1.0).
  * Copy HDF5 attribute 'framesize' into outputs (root and dataset attrs). If inputs differ, use the smallest.

- Outputs (written to --out-dir):
  * <sig1_name>.h5, <sig2_name>.h5, mixture.h5
    - Each contains dataset "dataset" with shape (N, 2, T_out), dtype=float32.
    - Each carries attribute 'framesize' (if present in inputs; smallest if they differ).
  * <sig1_name>.json, <sig2_name>.json
    - Copies of the original metadata updated with:
        - "fs" (updated to fs_out if resampled)
        - "n_samples" (integer T_out)
  * mixture.json
    - Dict with two entries named by the input stems:
        { "<sig1_name>": <sig1_meta_updated>, "<sig2_name>": <sig2_meta_updated> }

- Usage:
    python create_rf_dataset.py \
        --sig1 /path/to/sig1.h5 \
        --sig2 /path/to/sig2.h5 \
        --out-dir /path/to/outdir \
        --attenuation 0.8 \
        --batch 64
"""

import argparse
import os
import sys
import json
import h5py
import numpy as np
import math
from typing import Tuple, Optional
from fractions import Fraction
from scipy.signal import resample_poly

# --- NEW: helpers for SIR/SINR and augmentations ---

def _parse_range_or_list(text: str) -> Tuple[str, np.ndarray]:
    """
    Accepts formats like: "5:20" (uniform range inclusive) or "0,3,6" (choices).
    Returns ("range", np.array([lo, hi])) or ("list", np.array([...])).
    """
    t = text.strip()
    if ":" in t:
        a, b = t.split(":")
        lo = float(a.strip()); hi = float(b.strip())
        if hi < lo:
            lo, hi = hi, lo
        return "range", np.array([lo, hi], dtype=np.float32)
    vals = [float(x.strip()) for x in t.split(",") if x.strip() != ""]
    if len(vals) == 0:
        raise ValueError("Empty list for range/list parser.")
    return "list", np.array(vals, dtype=np.float32)


def _sample_values(spec: Tuple[str, np.ndarray], N: int, rng: np.random.Generator) -> np.ndarray:
    """Draw N values from ("range",[lo,hi]) or ("list",[...])."""
    kind, arr = spec
    if kind == "range":
        lo, hi = float(arr[0]), float(arr[1])
        return rng.uniform(lo, hi, size=N).astype(np.float32)
    else:
        idx = rng.integers(0, len(arr), size=N)
        return arr[idx].astype(np.float32)


def _rms_power(x: np.ndarray) -> float:
    """RMS (amplitude) over all channels/time. x shape (..., 2, T) or (2, T)."""
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _attenuation_for_target_sir_db(rms_sig: float, rms_int: float, sir_db: float) -> float:
    """
    Given desired SIR (Ps/Pi) in dB and current RMS of signal and interferer, return 'a' so that
    SIR = Ps / (a^2 * Pi) => a = sqrt(Ps / (Pi * 10^(SIR/10)))
    Using RMS^2 as power proxy.
    """
    Ps = rms_sig**2
    Pi = max(rms_int**2, 1e-20)
    a2 = Ps / (Pi * (10.0**(sir_db / 10.0)))
    return float(np.sqrt(max(a2, 0.0)))


def _noise_sigma_for_target_sinr_db(rms_sig: float, rms_int_scaled: float, sinr_db: float) -> float:
    """
    Given SINR = Ps / (Pi + Pn). With Ps (=rms_sig^2), Pi (=rms_int_scaled^2), solve for Pn.
    If required Pn <= 0 (already too much interference), return 0 (can't hit SINR exactly).
    Return noise_std (amplitude).
    """
    Ps = rms_sig**2
    Pi = (rms_int_scaled**2)
    denom = 10.0**(sinr_db / 10.0)
    Pn = Ps / denom - Pi
    if Pn <= 0.0:
        return 0.0
    return float(np.sqrt(Pn))


def _apply_phase_rotate(x: np.ndarray, deg: float) -> np.ndarray:
    """
    Global phase rotation of I/Q by 'deg' degrees. x: (..., 2, T)
    """
    if abs(deg) < 1e-9:
        return x
    rad = np.deg2rad(deg)
    c, s = np.cos(rad, dtype=np.float32), np.sin(rad, dtype=np.float32)
    I = x[..., 0, :]
    Q = x[..., 1, :]
    I2 = c * I - s * Q
    Q2 = s * I + c * Q
    out = np.stack([I2, Q2], axis=-2)  # keep (..., 2, T)
    return out.astype(np.float32, copy=False)


def _apply_cfo(x: np.ndarray, fs: float, f_hz: float) -> np.ndarray:
    """
    Small CFO: multiply by exp(j*2*pi*f*n/fs). x: (B, 2, T) or (2, T).
    """
    if abs(f_hz) < 1e-9:
        return x
    single = (x.ndim == 2)
    if single:
        x = x[np.newaxis, ...]  # (1, 2, T)
    B, C, T = x.shape
    n = np.arange(T, dtype=np.float32)
    phi = 2.0 * np.pi * f_hz * n / float(fs)
    c = np.cos(phi, dtype=np.float32)[None, None, :]
    s = np.sin(phi, dtype=np.float32)[None, None, :]
    I = x[:, 0, :]
    Q = x[:, 1, :]
    I2 = c * I - s * Q
    Q2 = s * I + c * Q
    y = np.stack([I2, Q2], axis=1)
    return y[0] if single else y


def _apply_time_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Circular time shift (samples). x: (..., 2, T)
    """
    if shift == 0:
        return x
    return np.roll(x, shift=shift, axis=-1)


# -------------------------
# I/O helpers
# -------------------------

def read_h5_shape(path: str) -> Tuple[int, int, int]:
    """Return (N, C, T) for the 'dataset' inside an HDF5 file and validate I/Q layout."""
    with h5py.File(path, "r") as f:
        if "dataset" not in f:
            raise KeyError(f"{path} does not contain a dataset named 'dataset'.")
        ds = f["dataset"]
        if ds.ndim != 3:
            raise ValueError(
                f"{path} -> 'dataset' must be 3D (N, 2, T). Found: {ds.shape}"
            )
        N, C, T = ds.shape
        if C != 2:
            raise ValueError(
                f"{path} -> expected 2 channels (I/Q) at dim=1. Found shape: {ds.shape}"
            )
        return N, C, T


def read_h5_framesize(path: str) -> Optional[int]:
    """
    Read 'framesize' attribute from an HDF5 file.
    Priority: file root attrs -> dataset 'dataset' attrs.
    Returns int if found and numeric; else None.
    """
    try:
        with h5py.File(path, "r") as f:
            # Try file root
            if "framesize" in f.attrs:
                val = f.attrs["framesize"]
                try:
                    return int(val)
                except Exception:
                    pass
            # Try dataset attrs
            if "dataset" in f:
                ds = f["dataset"]
                if "framesize" in ds.attrs:
                    val = ds.attrs["framesize"]
                    try:
                        return int(val)
                    except Exception:
                        pass
    except Exception:
        pass
    return None


def pick_chunks(shape: Tuple[int, int, int], bytes_target: int = 4 << 20,
                batch_hint: int = 64, dtype=np.float32):
    """Choose HDF5 chunk shape aiming for ~bytes_target per chunk and aligning with batch size."""
    N, C, T = shape
    item = np.dtype(dtype).itemsize
    chunk_N = max(1, min(batch_hint, N))
    max_T = max(1, bytes_target // (chunk_N * C * item))
    # Optionally round T to a power of 2 for efficiency, but do not exceed T
    chunk_T = 1 << int(np.floor(np.log2(max_T))) if max_T > 0 else 1
    chunk_T = int(max(1, min(T, chunk_T)))
    return (chunk_N, C, chunk_T)


def create_out_dataset(path: str, shape: Tuple[int, int, int], dtype="float32",
                       bytes_target: int = 4 << 20, batch_hint: int = 64,
                       compression: str = "gzip", compression_opts: int = 4,
                       shuffle: bool = True, fletcher32: bool = False):
    """Create an HDF5 file with a chunked/compressed dataset named 'dataset' and return (file, dataset)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    chunks = pick_chunks(shape, bytes_target=bytes_target, batch_hint=batch_hint, dtype=np.dtype(dtype))
    f = h5py.File(path, "w")
    ds = f.create_dataset(
        "dataset",
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=shuffle,
        fletcher32=fletcher32,
    )
    return f, ds


def json_path_for(h5_path: str) -> str:
    """Return the sibling JSON path for a given .h5 path (same directory, same stem)."""
    base, _ = os.path.splitext(h5_path)
    return base + ".json"


def load_json_with_fs(path: str) -> dict:
    """Load a JSON file and ensure it includes a numeric 'fs' field."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON metadata file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "fs" not in meta:
        raise KeyError(f"Metadata {path} must include field 'fs' (sampling frequency in Hz).")
    try:
        meta["fs"] = float(meta["fs"])
    except Exception as e:
        raise ValueError(f"Field 'fs' in {path} must be numeric. Got: {meta.get('fs')}") from e
    return meta


def save_json(path: str, data: dict):
    """Write a JSON file with UTF-8 encoding and pretty formatting."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -------------------------
# Resampling helpers
# -------------------------
def rational_up_down(fs_in: float, fs_out: float, max_den: int = 1000) -> tuple[int, int]:
    """
    Return (up, down) such that fs_out ≈ fs_in * up / down.
    Use Fraction.from_float because Fraction(x, y) requires Rationals.
    """
    num = Fraction.from_float(fs_out).limit_denominator(max_den)
    den = Fraction.from_float(fs_in).limit_denominator(max_den)
    frac = num / den
    return frac.numerator, frac.denominator


def resample_iq(arr: np.ndarray, fs_in: float, fs_out: float, target_len: int | None = None) -> np.ndarray:
    """
    Resample I/Q array (2,T) or batch (B,2,T) from fs_in to fs_out using resample_poly.
    If target_len is provided, only resample the minimum required input to produce at least target_len output
    (big speedup for large upsampling ratios).
    """
    if fs_in is None or fs_out is None:
        raise ValueError("resample_iq requires both fs_in and fs_out.")
    if abs(fs_in - fs_out) < 1e-12:
        # If we only need target_len, slice; otherwise return as is
        return arr if target_len is None else arr[..., :target_len]

    up, down = rational_up_down(fs_in, fs_out)

    # If we know how many output samples we want, figure out how many input samples are needed.
    if target_len is not None:
        # resample_poly uses a FIR with default numtaps ≈ 10*max(up,down)+1; keep a margin
        margin = 10 * max(up, down)
        in_needed = int(math.ceil(target_len * down / up) + margin)
        in_len = arr.shape[-1]
        use_len = min(in_len, in_needed)
        arr = arr[..., :use_len]

    out = resample_poly(arr, up, down, axis=-1, padtype="line")
    return out if target_len is None else out[..., :target_len]


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Build mixture.h5 = sig1 + a*sig2, resampling as needed based on JSON 'fs'. "
                                            "Now supports per-item SIR/SINR and basic RF augmentations.")
    p.add_argument("--sig1", required=True, help="Path to signal #1 HDF5 file (dataset 'dataset' with shape (N,2,T)).")
    p.add_argument("--sig2", required=True, help="Path to signal #2 HDF5 file (dataset 'dataset' with shape (N,2,T)).")
    p.add_argument("--out-dir", required=True, help="Output directory for <sig1>.h5, <sig2>.h5, mixture.h5 and JSON metadata.")
    # legacy attenuation (kept for backward compatibility)
    p.add_argument("-a", "--attenuation", type=float, default=1.0,
                   help="Legacy fixed attenuation applied to sig2 (ignored if --sir-db is set).")
    p.add_argument("--batch", type=int, default=64, help="Batch size (number of items) per I/O block.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for SIR/SINR and augmentations.")
    # NEW: per-item SIR/SINR specs
    p.add_argument("--sir-db", type=str, default=None,
                   help="Target SIR per item in dB. Use 'lo:hi' for uniform range or 'v1,v2,...' for discrete choices.")
    p.add_argument("--sinr-db", type=str, default=None,
                   help="Target SINR per item in dB (adds AWGN). Use same format as --sir-db.")
    p.add_argument("--awgn-ref", type=str, default="mixture", choices=["signal", "interf", "mixture"],
                   help="Reference when reporting/logging power for AWGN. (Noise sigma is solved from signal & interferer powers)")
    # NEW: lightweight RF augs on the interferer
    p.add_argument("--phase-deg", type=float, default=0.0,
                   help="Max absolute global phase rotation (deg) on the interferer (uniform in [-val, +val]).")
    p.add_argument("--cfo-hz", type=float, default=0.0,
                   help="Max absolute CFO in Hz on the interferer (uniform in [-val, +val]).")
    p.add_argument("--time-shift", type=int, default=0,
                   help="Max absolute circular time shift (samples) on the interferer (uniform in [-val, +val]).")
    args = p.parse_args()

    sig1_path = args.sig1
    sig2_path = args.sig2
    out_dir = args.out_dir
    att = float(args.attenuation)
    batch = int(args.batch)

    # Derive stems (used for printing and output JSON keys/filenames)
    sig1_name = os.path.splitext(os.path.basename(sig1_path))[0]
    sig2_name = os.path.splitext(os.path.basename(sig2_path))[0]

    # 1) Read input shapes and validate
    N1, C1, T1 = read_h5_shape(sig1_path)
    N2, C2, T2 = read_h5_shape(sig2_path)
    N = min(N1, N2)
    if N1 != N2:
        print(f"[WARN] N_{sig1_name}={N1} and N_{sig2_name}={N2} differ. Will use N={N} (minimum).")

    # 1.b) Read framesize attributes (optional)
    fsz1 = read_h5_framesize(sig1_path)
    fsz2 = read_h5_framesize(sig2_path)
    framesize_out = None
    if fsz1 is None and fsz2 is None:
        framesize_out = None
    elif fsz1 is None:
        framesize_out = int(fsz2)
    elif fsz2 is None:
        framesize_out = int(fsz1)
    else:
        framesize_out = int(min(fsz1, fsz2))  # choose the smallest if they differ

    print("=== INPUT SHAPES ===")
    print(f"{sig1_name:>12s} : {sig1_path}")
    print(f"  - shape : (N={N1}, C={C1}, T={T1})  [expected C=2 (I/Q)]")
    print(f"{sig2_name:>12s} : {sig2_path}")
    print(f"  - shape : (N={N2}, C={C2}, T={T2})  [expected C=2 (I/Q)]")

    print("\n=== HDF5 'framesize' attribute (if present) ===")
    print(f"{sig1_name:>12s} : framesize = {fsz1!r}")
    print(f"{sig2_name:>12s} : framesize = {fsz2!r}")
    if framesize_out is not None:
        pick_info = "min of both" if (fsz1 is not None and fsz2 is not None and fsz1 != fsz2) else "propagated"
        print(f"Chosen framesize_out = {framesize_out} ({pick_info})")
    else:
        print("No framesize attribute found in inputs; outputs will omit it.")

    # 2) Load metadata JSONs and decide fs_out
    sig1_json_in = json_path_for(sig1_path)
    sig2_json_in = json_path_for(sig2_path)
    sig1_meta = load_json_with_fs(sig1_json_in)
    sig2_meta = load_json_with_fs(sig2_json_in)

    fs1 = sig1_meta["fs"]
    fs2 = sig2_meta["fs"]
    fs_out = max(fs1, fs2)  # policy: preserve bandwidth by choosing the higher fs

    print("\n=== SAMPLING RATES (from JSON) ===")
    print(f"fs_{sig1_name} = {fs1} Hz")
    print(f"fs_{sig2_name} = {fs2} Hz")
    print(f"Chosen fs_out = {fs_out} Hz (policy: max of both)")

    need_resample_1 = abs(fs1 - fs_out) > 1e-12
    need_resample_2 = abs(fs2 - fs_out) > 1e-12

    if need_resample_1:
        print(f"[INFO] Will resample {sig1_name} from {fs1} Hz to {fs_out} Hz.")
    if need_resample_2:
        print(f"[INFO] Will resample {sig2_name} from {fs2} Hz to {fs_out} Hz.")
    if not (need_resample_1 or need_resample_2):
        print("[INFO] No resampling needed (both already at fs_out).")

    # 3) Determine T_out after potential resampling (by probing first item)
    with h5py.File(sig1_path, "r") as f1, h5py.File(sig2_path, "r") as f2:
        ds1 = f1["dataset"]
        ds2 = f2["dataset"]

        probe1 = ds1[0, :, :]  # (2, T1)
        probe2 = ds2[0, :, :]  # (2, T2)

        s1_rs = resample_iq(probe1.astype(np.float32, copy=False), fs1, fs_out) if need_resample_1 else probe1.astype(np.float32, copy=False)
        s2_rs = resample_iq(probe2.astype(np.float32, copy=False), fs2, fs_out) if need_resample_2 else probe2.astype(np.float32, copy=False)

        T1r = s1_rs.shape[-1]
        T2r = s2_rs.shape[-1]
        T_out = min(T1r, T2r)

    print("\n=== LENGTHS AFTER RESAMPLING ===")
    if need_resample_1 or need_resample_2:
        print(f"Estimated lengths (index 0): {sig1_name} -> {T1r} samples, {sig2_name} -> {T2r} samples")
    else:
        print(f"No resampling: {sig1_name} -> {T1r} samples, {sig2_name} -> {T2r} samples")
    print(f"T_out = min(len1, len2) = {T_out}")
    print(f"All three outputs will be uniform: shape (N={N}, C=2, T={T_out})")

    # 4) Prepare output HDF5 files
    out_sig1 = os.path.join(out_dir, f"{sig1_name}.h5")
    out_sig2 = os.path.join(out_dir, f"{sig2_name}.h5")
    out_mix  = os.path.join(out_dir, "mixture.h5")

    f1_out, ds1_out = create_out_dataset(out_sig1, (N, 2, T_out))
    f2_out, ds2_out = create_out_dataset(out_sig2, (N, 2, T_out))
    fm_out, dsm_out = create_out_dataset(out_mix,  (N, 2, T_out))

    # 4.b) Write framesize attribute to outputs (file root and dataset), if available
    if framesize_out is not None:
        for f, ds, tag in ((f1_out, ds1_out, sig1_name), (f2_out, ds2_out, sig2_name), (fm_out, dsm_out, "mixture")):
            try:
                f.attrs["framesize"] = int(framesize_out)
                ds.attrs["framesize"] = int(framesize_out)
            except Exception as e:
                print(f"[WARN] Could not write 'framesize' to {tag}: {e}")

    # 5) Process in batches: optional resampling -> trim -> write  (UPDATED)
    print("\n=== PROCESSING ===")
    print(f"Attenuation base a (applied to {sig2_name}) = {att:.6g}  [ignored if --sir-db is set]")
    if args.sir_db is not None:
        print(f"Per-item target SIR[dB] from: {args.sir_db}")
    if args.sinr_db is not None:
        print(f"Per-item target SINR[dB] from: {args.sinr_db} (adds AWGN)")
        print(f"AWGN reference: {args.awgn_ref}  (noise power computed vs this reference)")

    print("Processing in batches...")

    rng = np.random.default_rng(args.seed)
    sir_spec = _parse_range_or_list(args.sir_db) if args.sir_db is not None else None
    sinr_spec = _parse_range_or_list(args.sinr_db) if args.sinr_db is not None else None

    # Allocate per-item logs (saved later)
    att_per_item   = np.zeros((N,), dtype=np.float32)
    sir_db_items   = np.full((N,), np.nan, dtype=np.float32)
    sinr_db_items  = np.full((N,), np.nan, dtype=np.float32)
    noise_std_item = np.zeros((N,), dtype=np.float32)
    phase_deg_item = np.zeros((N,), dtype=np.float32)
    cfo_hz_item    = np.zeros((N,), dtype=np.float32)
    tshift_item    = np.zeros((N,), dtype=np.int32)

    try:
        with h5py.File(sig1_path, "r") as f1, h5py.File(sig2_path, "r") as f2:
            ds1 = f1["dataset"]
            ds2 = f2["dataset"]

            # Pre-sample SIR/SINR targets for all items if requested
            if sir_spec is not None:
                sir_targets = _sample_values(sir_spec, N, rng)
            if sinr_spec is not None:
                sinr_targets = _sample_values(sinr_spec, N, rng)

            for start in range(0, N, batch):
                end = min(start + batch, N)
                idxs = np.arange(start, end)

                blk1 = ds1[start:end, :, :]   # (B, 2, T1)
                blk2 = ds2[start:end, :, :]   # (B, 2, T2)
                blk1 = blk1.astype(np.float32, copy=False)
                blk2 = blk2.astype(np.float32, copy=False)

                # Resample as needed (operates along last axis)
                s1_rs = resample_iq(blk1, fs1, fs_out) if need_resample_1 else blk1
                s2_rs = resample_iq(blk2, fs2, fs_out) if need_resample_2 else blk2

                # Trim to common T_out
                s1_rs = s1_rs[..., :T_out]
                s2_rs = s2_rs[..., :T_out]

                # --- Optional RF augmentations on interferer BEFORE mixing ---
                if args.phase_deg is not None and abs(args.phase_deg) > 0:
                    # random uniform in [-phase_deg, +phase_deg]
                    ph = rng.uniform(-abs(args.phase_deg), abs(args.phase_deg), size=(end-start,)).astype(np.float32)
                else:
                    ph = np.zeros((end-start,), dtype=np.float32)
                if args.cfo_hz is not None and abs(args.cfo_hz) > 0:
                    # random uniform in [-cfo, +cfo]
                    cfo = rng.uniform(-abs(args.cfo_hz), abs(args.cfo_hz), size=(end-start,)).astype(np.float32)
                else:
                    cfo = np.zeros((end-start,), dtype=np.float32)
                if args.time_shift is not None and args.time_shift != 0:
                    # random integer shift in [-K, K]
                    tsh = rng.integers(-abs(args.time_shift), abs(args.time_shift)+1, size=(end-start,), dtype=np.int32)
                else:
                    tsh = np.zeros((end-start,), dtype=np.int32)

                # Apply per-item augments to s2 (interferer)
                for j in range(end-start):
                    s2_rs[j] = _apply_phase_rotate(s2_rs[j], float(ph[j]))
                    s2_rs[j] = _apply_cfo(s2_rs[j], fs_out, float(cfo[j]))
                    s2_rs[j] = _apply_time_shift(s2_rs[j], int(tsh[j]))

                # Compute per-item attenuation/noise
                mix_blk = np.empty_like(s1_rs)
                for k, i in enumerate(idxs):
                    # Per-item RMS
                    rms1 = _rms_power(s1_rs[k])
                    rms2 = _rms_power(s2_rs[k])

                    if sir_spec is not None:
                        target_sir = float(sir_targets[i])
                        a_i = _attenuation_for_target_sir_db(rms1, rms2, target_sir)
                        sir_db_items[i] = target_sir
                    else:
                        a_i = att  # fixed attenuation as before

                    # Build mix without noise first
                    mix_i = s1_rs[k] + a_i * s2_rs[k]

                    # If SINR target specified, add AWGN to reach it
                    if sinr_spec is not None:
                        target_sinr = float(sinr_targets[i])
                        if args.awgn_ref == "signal":
                            ref_rms = rms1
                        elif args.awgn_ref == "interf":
                            ref_rms = abs(a_i) * rms2
                        else:  # mixture
                            ref_rms = _rms_power(mix_i)
                        # noise std solving Ps/(Pi+Pn)=SINR, using Ps from chosen reference
                        # For stability we compute sigma from the real signal power:
                        sigma = _noise_sigma_for_target_sinr_db(rms_sig=rms1, rms_int_scaled=abs(a_i)*rms2, sinr_db=target_sinr)
                        if sigma > 0.0:
                            noise = rng.normal(0.0, sigma, size=mix_i.shape).astype(np.float32)
                            mix_i = mix_i + noise
                        sinr_db_items[i] = target_sinr
                        noise_std_item[i] = sigma

                    att_per_item[i] = a_i
                    phase_deg_item[i] = ph[k]
                    cfo_hz_item[i] = cfo[k]
                    tshift_item[i] = tsh[k]
                    mix_blk[k] = mix_i

                # Write individual signals
                ds1_out[start:end, :, :] = s1_rs
                ds2_out[start:end, :, :] = s2_rs
                dsm_out[start:end, :, :] = mix_blk

                print(f"  - Batch [{start:6d}:{end:6d}] -> "
                      f"{sig1_name} {s1_rs.shape}, {sig2_name} {s2_rs.shape}, MIX {mix_blk.shape}")
    finally:
        # Always close output files
        f1_out.close()
        f2_out.close()
        fm_out.close()

    # 6) Quick verification on a few random indices (robust to per-item a_i and optional noise)
    print("\n=== QUICK VERIFICATION ===")
    rng = np.random.default_rng(0)
    check_idxs = rng.choice(N, size=min(5, N), replace=False) if N > 0 else []
    with h5py.File(out_sig1, "r") as f1, \
        h5py.File(out_sig2, "r") as f2, \
        h5py.File(out_mix,  "r") as fm:
        D1 = f1["dataset"]; D2 = f2["dataset"]; DM = fm["dataset"]

        # detect if we actually used per-item fields
        used_sir  = (args.sir_db  is not None)
        used_sinr = (args.sinr_db is not None)

        for i in check_idxs:
            x1 = D1[i]; x2 = D2[i]; xm = DM[i]
            if used_sir:
                # si hubo SIR objetivo, usa la atenuación por ítem
                a_i = float(att_per_item[i])
            else:
                a_i = att

            recon = x1 + a_i * x2

            if used_sinr:
                # si además hubo SINR objetivo, había ruido añadido -> solo comprobamos orden de magnitud
                err_rms = np.sqrt(np.mean((xm - recon) ** 2))
                print(f"  idx={int(i):6d} | rms(mix - (x1 + a_i*x2)) = {err_rms:.3e}  [noise present]")
            else:
                # igualdad exacta salvo redondeos
                err = np.max(np.abs(xm - recon))
                rms_m = np.sqrt(np.mean(xm**2))
                print(f"  idx={int(i):6d} | max|mix - (x1 + a_i*x2)| = {err:.3e} | rms(mix)={rms_m:.3e}")

    # 7) Write updated/copy JSONs to output directory
    os.makedirs(out_dir, exist_ok=True)
    sig1_json_out = os.path.join(out_dir, f"{sig1_name}.json")
    sig2_json_out = os.path.join(out_dir, f"{sig2_name}.json")
    mix_json_out  = os.path.join(out_dir, "mixture.json")

    sig1_meta_out = dict(sig1_meta)  # copy all fields
    sig2_meta_out = dict(sig2_meta)

    # Update fs if resampled; add n_samples (time length after trimming)
    sig1_meta_out["fs"] = fs_out if need_resample_1 else fs1
    sig2_meta_out["fs"] = fs_out if need_resample_2 else fs2
    sig1_meta_out["n_samples"] = int(T_out)
    sig2_meta_out["n_samples"] = int(T_out)

    save_json(sig1_json_out, sig1_meta_out)
    save_json(sig2_json_out, sig2_meta_out)

    # Build mixture.json with two dicts (copying all fields from the updated JSONs)
    mixture_meta = {
        sig1_name: sig1_meta_out,
        sig2_name: sig2_meta_out
    }
        # Add optional summary of generation policy (non-breaking)
    mixture_meta["_generation"] = {
        "sir_db": args.sir_db,
        "sinr_db": args.sinr_db,
        "awgn_ref": args.awgn_ref,
        "phase_deg_max": args.phase_deg,
        "cfo_hz_max": args.cfo_hz,
        "time_shift_max": args.time_shift,
        "seed": args.seed
    }
    save_json(mix_json_out, mixture_meta)

    # 6.bis) Save per-item generation arrays only if used (avoid useless datasets on disk)
    with h5py.File(out_mix, "a") as fm:
        if args.sir_db is not None:
            fm.create_dataset("sir_db", data=sir_db_items, compression="gzip", compression_opts=4)
            fm.create_dataset("attenuation", data=att_per_item, compression="gzip", compression_opts=4)
        if args.sinr_db is not None:
            fm.create_dataset("sinr_db", data=sinr_db_items, compression="gzip", compression_opts=4)
            fm.create_dataset("noise_std", data=noise_std_item, compression="gzip", compression_opts=4)
        # augs solo si se pidieron rangos != 0
        if (args.phase_deg or 0.0) != 0.0:
            fm.create_dataset("phase_deg", data=phase_deg_item, compression="gzip", compression_opts=4)
        if (args.cfo_hz or 0.0) != 0.0:
            fm.create_dataset("cfo_hz", data=cfo_hz_item, compression="gzip", compression_opts=4)
        if (args.time_shift or 0) != 0:
            fm.create_dataset("time_shift", data=tshift_item, compression="gzip", compression_opts=4)

    print("\n=== SUMMARY ===")
    print(f"Written to {out_dir}:")
    print(f"  - {out_sig1}")
    print(f"  - {out_sig2}")
    print(f"  - {out_mix}")
    print(f"  - {sig1_json_out}")
    print(f"  - {sig2_json_out}")
    print(f"  - {mix_json_out}")
    print(f"Final shape: (N={N}, 2, T_out={T_out})")
    if framesize_out is not None:
        print(f"framesize written to outputs: {framesize_out}")
    else:
        print("framesize not written (not found in inputs).")
    if need_resample_1 or need_resample_2:
        print("Resampling performed:")
        if need_resample_1:
            print(f"  {sig1_name}: {fs1} Hz -> {fs_out} Hz")
        if need_resample_2:
            print(f"  {sig2_name}: {fs2} Hz -> {fs_out} Hz")
    else:
        print("No resampling performed (both inputs already at the selected fs_out).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
