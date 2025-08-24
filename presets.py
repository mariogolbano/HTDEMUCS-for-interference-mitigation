# presets.py
from dataclasses import replace
from config_from_json import Config

def htdemucs_preset(cfg: Config) -> Config:
    """
    Make the model hyperparams match the canonical HT‑Demucs recipe.
    Keeps data-dependent fields (sample_rate, n_fft, segment_seconds, stems, in_channels).
    """
    return replace(
        cfg,
        # MODEL (as in HT‑Demucs v4)
        depth=4,                 # 4 outer + 2 inner (Transformer)  ← paper/repos
        base_channels=128,        # width per stage (commonly 64 in Demucs family)
        transformer_dim=1024,     # token dim at the cross-domain transformer
        transformer_heads=16,     # multi-head attention
        transformer_layers=12,    # number of transformer blocks
        # TRAIN defaults (you ajust after we know your GPU)
        batch_size=8,
        epochs=200,
        lr=1e-4,
    )
