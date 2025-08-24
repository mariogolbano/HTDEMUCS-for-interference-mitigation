# train_single_target.py
"""
Entrenamiento focalizado en un único stem (interferencia / señal deseada).
Admite:
  --name  : nombre base de los checkpoints (ej. qpsk_le)
  --base  : carpeta con mixture.h5 + stems (train)
  --valid : carpeta con mixture.h5 + stems (val)
  --init  : checkpoint para inicializar (opcional)

Early Stopping:
  Si tras reducir LR (ReduceLROnPlateau) no mejora la val_loss en
  --es-patience-after-reduce épocas, se detiene el entrenamiento.
"""

from __future__ import annotations
import math, csv, argparse, os
from datetime import datetime
from pathlib import Path
from dataclasses import replace, asdict 
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from torch.utils.data import DataLoader
from models.htdemucs import HTDemucs
from data.dataset_h5 import H5SeparationDataset
from data.epoch_wrap import EpochMultiply
from config_from_json import load_config_from_base
from presets import htdemucs_preset

# ---------- CONFIGURACIÓN ----------
TARGET_STEM_IDX  = 0  # <–– cámbialo si tu stem objetivo no es el primero

# ---------- MR-STFT helpers ----------
_HANN_WINDOWS = {}
def _get_hann(n_fft: int, device: torch.device):
    key = (n_fft, device)
    win = _HANN_WINDOWS.get(key)
    if win is None or win.device != device:
        win = torch.hann_window(n_fft, device=device, dtype=torch.float32)
        _HANN_WINDOWS[key] = win
    return win

def _stft_mag_batch(x_1d: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = _get_hann(n_fft, x_1d.device)
    X = torch.stft(
        x_1d, n_fft=n_fft, hop_length=hop,
        window=win, center=True, normalized=True, return_complex=True
    )
    return torch.abs(X)

def mrstft_loss(y_hat: torch.Tensor, y_true: torch.Tensor):
    B, S, C, T = y_hat.shape
    mono_hat  = y_hat.reshape(B * S, C, T).mean(1)
    mono_true = y_true.reshape(B * S, C, T).mean(1)
    loss = 0.
    for nfft in (256, 512, 1024):
        hop = nfft // 4
        loss += torch.mean(torch.abs(
            _stft_mag_batch(mono_hat, nfft, hop) -
            _stft_mag_batch(mono_true, nfft, hop)))
    return loss / 3.

def _combine_losses(l_t: torch.Tensor, l_s: torch.Tensor, arch: str):
    """
    Devuelve (loss_total, l_t_eff, l_s_eff) según la arquitectura:
      - time : loss = l_t, l_s_eff = 0
      - spec : loss = l_s, l_t_eff = 0
      - hybrid: loss = 0.5*l_t + 0.5*l_s
    """
    if arch == "time":
        return l_t, l_t, torch.zeros_like(l_s)
    elif arch == "spec":
        return l_s, torch.zeros_like(l_t), l_s
    else:  # hybrid
        return 0.5 * (l_t + l_s), l_t, l_s

# ---------- VALIDACIÓN ----------
@torch.no_grad()
def valid_epoch(loader, model, device, arch: str = "hybrid"):
    model.eval()
    tot, tot_t, tot_s = 0., 0., 0.
    num_sum = den_sum = 0.

    for x_mix, y_true_all in loader:
        x_mix  = x_mix.to(device, non_blocking=True)
        y_true = y_true_all[:, TARGET_STEM_IDX:TARGET_STEM_IDX+1].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            y_hat  = model(x_mix)
            l_t    = torch.mean(torch.abs(y_hat - y_true))
            l_s    = mrstft_loss(y_hat, y_true)

            loss, l_t_eff, l_s_eff = _combine_losses(l_t, l_s, arch)

        bs      = x_mix.size(0)
        tot    += loss.item()   * bs
        tot_t  += l_t_eff.item()* bs
        tot_s  += l_s_eff.item()* bs
        num_sum += torch.sum((y_hat - y_true) ** 2).item()
        den_sum += torch.sum(y_true ** 2).item()

    n = len(loader.dataset)
    nmse_db = 10. * math.log10(num_sum / (den_sum + 1e-12) + 1e-12)
    return tot / n, nmse_db, tot_t / n, tot_s / n

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",  type=str, required=True, help="Nombre base para los checkpoints (ej. qpsk_le)")
    parser.add_argument("--base",  type=str, required=True, help="Carpeta con mixture.h5 y stems (train)")
    parser.add_argument("--valid", type=str, required=True, help="Carpeta con mixture.h5 y stems (val)")
    parser.add_argument("--init",  type=str, default=None,       help="Checkpoint para inicializar (opcional)")
    # NUEVO: selector de arquitectura
    parser.add_argument("--arch", choices=["hybrid","time","spec"], default="hybrid",
                        help="Arquitectura del modelo: hybrid (por defecto), time (sólo temporal), spec (sólo espectral).")
    # Early stopping
    parser.add_argument("--es-patience-after-reduce", type=int, default=5,
                        help="Épocas sin mejora tras reducir LR antes de parar.")
    parser.add_argument("--min-epochs", type=int, default=10,
                        help="Mínimo de épocas antes de permitir early stopping.")
    args = parser.parse_args()

    # Configuración base
    cfg_base = load_config_from_base(args.base)
    cfg      = htdemucs_preset(cfg_base)
    cfg      = replace(cfg, stems=1)
    Tseg     = int(cfg.segment_seconds * cfg.sample_rate)

    # Dataset train (con 'epoch multiply' si está disponible/querido)
    ds_tr = H5SeparationDataset(cfg.mix_h5, list(cfg_base.stem_h5_list),
                                segment=Tseg, normalize='rms')
    ds_tr = EpochMultiply(ds_tr, k=cfg.epoch_k)

    # Dataset val
    cfg_val_base = load_config_from_base(args.valid)
    ds_val = H5SeparationDataset(cfg_val_base.mix_h5, list(cfg_val_base.stem_h5_list),
                                 segment=Tseg, normalize='rms')

    # DataLoaders
    dl_tr = DataLoader(ds_tr,  batch_size=cfg.batch_size, shuffle=True,
                       num_workers=6, pin_memory=True, prefetch_factor=2)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, prefetch_factor=2)

    # Modelo (pasa arch=args.arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HTDemucs(in_channels=cfg.in_channels, stems=1,
                      audio_channels=cfg.in_channels, n_fft=cfg.n_fft,
                      depth=cfg.depth, base_channels=cfg.base_channels,
                      transformer_dim=cfg.transformer_dim,
                      transformer_heads=cfg.transformer_heads,
                      transformer_layers=cfg.transformer_layers,
                      segment_length=Tseg,
                      arch=args.arch).to(device)

    if args.init is not None and os.path.isfile(args.init):
        ckpt = torch.load(args.init, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"[INIT] Pesos desde: {args.init}  (missing={len(missing)}, unexpected={len(unexpected)})")

    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Logs
    logs = Path("logs"); logs.mkdir(exist_ok=True)
    csvf = (logs / f"loss_{args.name}.csv").open("a", newline="")
    writer = csv.writer(csvf)
    if csvf.tell() == 0:
        writer.writerow(["epoch", "train_loss", "train_t", "train_s",
                         "val_loss", "val_t", "val_s", "val_nmse_db",
                         "lr", "time_iso"]); csvf.flush()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, threshold=1e-4,
        cooldown=2, min_lr=1e-6)

    # Early stopping trackers
    best_val = float("inf")
    after_reduce_counter = None  # None = no estamos en ventana post-reduce
    eps_improve = 1e-6

    def run(loader, train=True):
        model.train(train)
        arch = getattr(model, "arch", None) or getattr(args, "arch", "hybrid")
        tot = tot_t = tot_s = 0.
        with torch.set_grad_enabled(train):
            for x_mix, y_true_all in loader:
                x_mix  = x_mix.to(device, non_blocking=True)
                y_true = y_true_all[:, TARGET_STEM_IDX:TARGET_STEM_IDX+1].to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=(device.type == "cuda")):
                    y_hat = model(x_mix)
                    l_t   = torch.mean(torch.abs(y_hat - y_true))
                    l_s   = mrstft_loss(y_hat, y_true)
                    loss, l_t_eff, l_s_eff = _combine_losses(l_t, l_s, arch)
                if train:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                    scaler.step(opt); scaler.update()
                bs     = x_mix.size(0)
                tot   += loss.item()    * bs
                tot_t += l_t_eff.item() * bs
                tot_s += l_s_eff.item() * bs

        n = len(loader.dataset)
        return tot / n, tot_t / n, tot_s / n

    ckpt_dir = Path("ckpts"); ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        tr, tr_t, tr_s = run(dl_tr, True)
        va, nmse, va_t, va_s = valid_epoch(dl_val, model, device, arch=args.arch)

        # Detectar si hay reducción de LR este paso
        prev_lr = opt.param_groups[0]["lr"]
        scheduler.step(va)
        new_lr = opt.param_groups[0]["lr"]
        reduced = (new_lr < prev_lr - 1e-12)
        if reduced:
            after_reduce_counter = 0  # empezamos ventana post-reduce

        lr = new_lr
        print(f"[{epoch:03d}] train {tr:.6f} (t {tr_t:.6f}|s {tr_s:.6f})  "
              f"val {va:.6f} (t {va_t:.6f}|s {va_s:.6f})  "
              f"NMSE {nmse:.2f} dB  lr={lr:.2e}")

        writer.writerow([epoch, f"{tr:.6f}", f"{tr_t:.6f}", f"{tr_s:.6f}",
                         f"{va:.6f}", f"{va_t:.6f}", f"{va_s:.6f}",
                         f"{nmse:.2f}", f"{lr:.6g}",
                         datetime.now().isoformat(timespec="seconds")])
        csvf.flush()

        # Checkpointing + mejora
        if va < best_val - eps_improve:
            best_val = va
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": va,
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if device.type=="cuda" else None,
                "cfg": {**asdict(cfg), "arch": args.arch}  # <= NUEVO: cfg embebido con 'arch'
            }, ckpt_dir / f"{args.name}_best.pt")
            print("   ↳ Best checkpoint actualizado.")
            after_reduce_counter = None
        else:
            if after_reduce_counter is not None:
                after_reduce_counter += 1
                if epoch >= args.min_epochs and after_reduce_counter >= args.es_patience_after_reduce:
                    print(f"   ↳ Early stopping (sin mejora {after_reduce_counter} épocas tras reducir LR).")
                    break

    # Guardar último (también con cfg)
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if device.type=="cuda" else None,
        "cfg": {**asdict(cfg), "arch": args.arch}
    }, ckpt_dir / f"{args.name}_last.pt")
    csvf.close()

if __name__ == "__main__":
    main()
