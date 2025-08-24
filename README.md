# HTDemucs for Interference Mitigation

This repository adapts the [HTDemucs](https://github.com/facebookresearch/demucs) architecture for interference mitigation in wireless communication signals.  
Instead of separating musical sources, this version separates a **signal of interest (SOI)** from **interferers** in complex I/Q radio mixtures.  

It builds upon:
- [TFM-foundation-model-wireless-signals](https://github.com/mariogolbano/TFM-foundation-model-wireless-signals): framework for generating synthetic RF datasets in HDF5+JSON format.  
- [Demucs v4](https://github.com/facebookresearch/demucs): original dual-domain audio separation architecture.  

---

## Repository Structure

### Top-level training and inference scripts
- **`train_single_target.py`** – train HTDemucs on a single target stem (desired signal or interferer). Supports checkpoints, logging, and early stopping:contentReference[oaicite:0]{index=0}.  
- **`infer_single_target.py`** – run inference with a trained checkpoint, producing separated HDF5 outputs:contentReference[oaicite:1]{index=1}.  


### Dataset creation and configuration
- **`create_rf_dataset.py`** – build a mixture dataset from two HDF5 I/Q signals with JSON metadata. Supports attenuation, SIR/SINR targets, and RF augmentations (phase rotation, CFO, time shift):contentReference[oaicite:3]{index=3}.  
- **`config_from_json.py`** – generates model/training configuration by parsing `mixture.json` (segment length, FFT size, sample rate, etc.):contentReference[oaicite:4]{index=4}.  
- **`presets.py`** – overrides hyperparameters to match canonical HTDemucs recipe (depth, channels, transformer layers, etc.):contentReference[oaicite:5]{index=5}.  

### Data loaders
- **`dataset_h5.py`** – PyTorch `Dataset` for mixtures + stems stored in HDF5. Handles cropping, padding, normalization, and SNR augmentation:contentReference[oaicite:6]{index=6}.  
- **`epoch_wrap.py`** – wrapper to artificially extend epochs by factor *k* without duplicating memory:contentReference[oaicite:7]{index=7}.  

### Model components
- **`htdemucs.py`** – main HTDemucs model: time branch, spectral branch, and cross-domain transformer:contentReference[oaicite:8]{index=8}.  
- **`blocks.py`** – 1D/2D encoder/decoder building blocks with gated linear units (GLUs) and dilated conv residuals:contentReference[oaicite:9]{index=9}.  
- **`transformer.py`** – cross-domain transformer (time ↔ spectral attention):contentReference[oaicite:10]{index=10}.  
- **`stft.py`** – STFT/ISTFT wrappers for complex I/Q channels:contentReference[oaicite:11]{index=11}.  

### Documentation and references
- **`2409.08839v2.pdf`** – RF Challenge paper (MIT/UC3M/Bar-Ilan/MIT-LL collaboration):contentReference[oaicite:12]{index=12}.  

---

## Pipeline Overview

1. **Dataset preparation**  
   - Start from raw RF signals generated with [TFM-foundation-model-wireless-signals](https://github.com/mariogolbano/TFM-foundation-model-wireless-signals).  
   - Use `create_rf_dataset.py` to mix two signals:  
     ```bash
     python create_rf_dataset.py \
       --sig1 path/to/signal1.h5 \
       --sig2 path/to/signal2.h5 \
       --out-dir DATASET_OUT/qpsk_wifi \
       --sir-db -10:20 \
       --sinr-db 0:30
     ```
   - Output: `mixture.h5`, `<sig1>.h5`, `<sig2>.h5` + JSON metadata.

2. **Training**  
   - Train on mixtures with target stem (e.g., SOI).  
   - Example single training:  
     ```bash
     python train_single_target.py \
       --name qpsk_le \
       --base DATASET_OUT/train/qpsk_le \
       --valid DATASET_OUT/val/qpsk_le
     ```
   - Logs: stored under `logs/loss_<name>.csv`  
   - Checkpoints: stored under `ckpts/<name>_best.pt` and `ckpts/<name>_last.pt`

   - **Options** (`train_single_target.py`):  
     - `--name` (str, required): experiment name (used for logs + ckpts).  
     - `--base` (dir, required): training dataset folder (`mixture.h5`, `<stem>.h5`, `mixture.json`).  
     - `--valid` (dir, required): validation dataset folder.  
     - `--init` (path, optional): checkpoint to initialize weights.  
     - `--es-patience-after-reduce` (int, default=5): early stopping after LR reduction.  
     - `--min-epochs` (int, default=10): minimum epochs before early stop.  

2. **Inference**  
   - Run separation on unseen mixtures:  
     ```bash
     python infer_single_target.py \
       --ckpt ckpts/qpsk_le_best.pt \
       --in-h5 DATASET_OUT/test/qpsk_wifi/mixture.h5 \
       --out-dir RESULTS/qpsk_le \
       --stem-name qpsk_clean
     ```
   - Output: `RESULTS/qpsk_le/qpsk_clean_separated.h5`  
   - Options:  
     - `--ckpt`: checkpoint path.  
     - `--in-h5`: input mixture file.  
     - `--out-dir`: output folder.  
     - `--base`: provide base dataset folder if checkpoint lacks config.  
     - `--device`: `cuda` or `cpu`.  
     - `--overlap`: overlap-add ratio (default 0.5).  
     - `--idx-from` / `--idx-to`: process a subset of items.  
     - `--stem-name`: custom stem output name.  
     - `--no-bf16`: disable BF16 autocast.  

---

## Outputs

- **Logs**: CSV with per-epoch metrics (L1, multi-resolution STFT loss, NMSE, LR).  
- **Checkpoints**: `ckpts/<name>_best.pt` and `_last.pt`.  
- **Inference results**: `<stem>_separated.h5` with `(N,2,T)` I/Q waveforms.  

---

## References

- Dataset generation framework:  
  [TFM-foundation-model-wireless-signals](https://github.com/mariogolbano/TFM-foundation-model-wireless-signals)  

- Original Demucs repo:  
  [facebookresearch/demucs](https://github.com/facebookresearch/demucs)  

- RF Challenge paper (Lancho et al., ICASSP 2024 / arXiv:2409.08839):  
  [RF Challenge PDF](./2409.08839v2.pdf)  

---

## Acknowledgements

- Adapted from **HTDemucs** (Défossez et al., Facebook AI Research).  
- RF domain adaptation and dataset framework developed as part of the UC3M Master’s Thesis on data-driven interference rejection.  
- Inspired by and connected to the RF Challenge community (MIT, UC3M, Bar-Ilan, MIT-LL).  
