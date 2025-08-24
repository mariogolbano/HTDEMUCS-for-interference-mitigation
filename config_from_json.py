# config_from_json.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Dict, Any, List, Optional, Tuple

@dataclass
class Config:
    # Data (all paths resolved relative to base_dir)
    mix_h5: str
    stem_h5_list: tuple
    segment_seconds: float
    sample_rate: float

    # Model
    stems: int
    in_channels: int = 2
    n_fft: int = 1024
    depth: int = 4
    base_channels: int = 64
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 5

    # Train
    batch_size: int = 8
    epochs: int = 100
    epoch_k: int = 8 
    lr: float = 1e-4
    device: str = "cuda"

def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _next_pow2_at_least(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

def _derive_stft_from_ofdm(meta: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Try deriving (n_fft, hop) from OFDM fields.
    Priority:
      1) FFTLength (L_u) and cyclicPrefixLength (L_cp) in samples.
      2) subcarrierSpacing (Δf) -> L_u ≈ fs/Δf, L_cp if provided.
    Returns (n_fft, hop) or None if not enough info.
    """
    try:
        L_u = meta.get("FFTLength", None)
        L_cp = meta.get("cyclicPrefixLength", 0)
        fs = float(meta["fs"])
        if L_u is None:
            # Fall back to subcarrierSpacing
            df = meta.get("subcarrierSpacing", None)
            if df is None:
                return None
            L_u = int(round(fs / float(df)))
        else:
            L_u = int(L_u)
        L_cp = int(L_cp)
        hop = max(1, L_u + L_cp)
        n_fft = _next_pow2_at_least(4 * hop)
        # Ensure n_fft is a multiple of 4 so hop = n_fft//4 stays integer
        while n_fft % 4 != 0:
            n_fft *= 2
        hop = n_fft // 4
        return n_fft, hop
    except Exception:
        return None

def _choose_segment_seconds(meta_any: Dict[str, Any], fs: float) -> float:
    """
    Choose segment length in seconds.
    Policy:
      - If n_samples exists: segment = min(8 ms, n_samples/fs).
      - Else default to 8 ms.
    """
    n_samples = meta_any.get("n_samples", None)
    if n_samples is not None:
        try:
            t_avail = float(n_samples) / fs
            return max(1e-3, min(8e-3, t_avail))  # clamp to [1 ms, 8 ms]
        except Exception:
            pass
    return 8e-3

def _find_ofdm_key_ordered(d: Dict[str, Any]) -> Optional[str]:
    """Return the first key that looks like OFDM in insertion order; else None."""
    for k in d.keys():
        if "ofdm" in k.lower():
            return k
    return None

def load_config_from_base(base_dir: str) -> Config:
    """
    Build Config by reading mixture.json located at <base_dir>/mixture.json.
    All .h5 paths are built from <base_dir>.
    """
    BASE = Path(base_dir).resolve()
    mix_json = BASE / "mixture.json"
    if not mix_json.exists():
        raise FileNotFoundError(f"mixture.json not found at: {mix_json}")

    meta_all = _load_json(mix_json)
    if not isinstance(meta_all, dict) or len(meta_all) == 0:
        raise ValueError("mixture.json must be a non-empty object mapping stem_name -> metadata dict.")

    # Preserve JSON insertion order for stems (Python 3.7+ dict preserves order)
    stem_names = [k for k in meta_all.keys() if not k.startswith("_")]
    stems = len(stem_names)

    # fs: assert all equal (they should after your preprocessing)
    fs_vals = [float(meta_all[name]["fs"]) for name in stem_names]
    fs = fs_vals[0]
    for v in fs_vals[1:]:
        if abs(v - fs) > 1e-6:
            print(f"[WARN] fs differs across stems ({fs} vs {v}). Using the first.")

    # Try deriving STFT from any OFDM-like stem; else fallback
    n_fft = 1024  # fallback default
    hop = n_fft // 4
    ofdm_key = _find_ofdm_key_ordered(meta_all)
    if ofdm_key is not None:
        derived = _derive_stft_from_ofdm(meta_all[ofdm_key])
        if derived is not None:
            n_fft, hop = derived
    else:
        # If *any* stem has enough fields, try it anyway
        for name in stem_names:
            derived = _derive_stft_from_ofdm(meta_all[name])
            if derived is not None:
                n_fft, hop = derived
                break

    # Segment length (use the first stem that has n_samples; else default)
    seg_sec = _choose_segment_seconds(meta_all[stem_names[0]], fs)

    # Build paths relative to base
    mix_h5 = str((BASE / "mixture.h5").resolve())
    stem_h5_list = tuple(str((BASE / f"{name}.h5").resolve()) for name in stem_names)

    # Assemble Config
    cfg = Config(
        mix_h5=mix_h5,
        stem_h5_list=stem_h5_list,
        segment_seconds=seg_sec,
        sample_rate=fs,
        stems=stems,
        n_fft=n_fft,
        # the rest stay at their defaults but you can tweak here if you want
    )

    # Print a short summary
    print("=== Derived Config from mixture.json ===")
    print(f"base_dir         : {BASE}")
    print(f"mix_h5           : {cfg.mix_h5}")
    print(f"stem_h5_list     : {cfg.stem_h5_list}")
    print(f"stems            : {cfg.stems}")
    print(f"fs               : {cfg.sample_rate} Hz")
    print(f"n_fft / hop      : {cfg.n_fft} / {cfg.n_fft // 4}")
    print(f"segment_seconds  : {cfg.segment_seconds:.6f} s (~{int(cfg.segment_seconds*cfg.sample_rate)} samples)")
    return cfg

# Optional CLI
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build Config from <base_dir>/mixture.json")
    ap.add_argument("--base", required=True, help="Base directory containing mixture.json, mixture.h5, and <stem>.h5 files.")
    ap.add_argument("--write", default=None, help="Optional path to write a Python file with a Config instance (config_auto.py).")
    args = ap.parse_args()

    cfg = load_config_from_base(args.base)

    if args.write:
        out = Path(args.write).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("# Auto-generated from mixture.json\n")
            f.write("from dataclasses import dataclass\n\n")
            f.write("@dataclass\nclass Config:\n")
            for k, v in cfg.__dict__.items():
                if isinstance(v, tuple):
                    f.write(f"    {k}: tuple = {v!r}\n")
                elif isinstance(v, str):
                    f.write(f"    {k}: str = r\"\"\"{v}\"\"\"\n")
                else:
                    f.write(f"    {k}: {type(v).__name__} = {v!r}\n")
        print(f"[OK] Wrote {out}")
