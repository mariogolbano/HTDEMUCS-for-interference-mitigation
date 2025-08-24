# data/dataset_h5.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class H5SeparationDataset(Dataset):
    """
    Dataset para separación de señales desde ficheros HDF5.
    
    Archivos esperados:
      - mixture.h5  -> dataset "dataset" con forma (N, 2, T)
      - stems.h5    -> uno o más, cada uno con "dataset" (N, 2, T)

    Devuelve:
      x_mix: tensor (2, Tseg)
      y    : tensor (S, 2, Tseg)
    """

    def __init__(self, mix_h5, stem_h5_list, segment=None, normalize='rms', snr_range=None):
        self.mix_h5 = mix_h5
        self.stem_h5_list = stem_h5_list
        self.segment = segment
        self.normalize = normalize
        self.snr_range = snr_range

        # handles perezosos
        self.hm = None
        self.hs = []

        # nº de muestras
        with h5py.File(self.mix_h5, "r") as f:
            self.N = len(f["dataset"])

    def _ensure_open(self):
        if self.hm is None:
            self.hm = h5py.File(self.mix_h5, "r")
            self.hs = [h5py.File(p, "r") for p in self.stem_h5_list]

    def __len__(self):
        return self.N

    def _as_tensor(self, x: np.ndarray):
        return torch.from_numpy(x.astype(np.float32, copy=False))

    def _compute_rms(self, x: np.ndarray):
        """Devuelve RMS global o por canal según self.normalize"""
        if self.normalize == 'rms':
            return np.sqrt(np.mean(x**2))
        elif self.normalize == 'per_channel_rms':
            return np.sqrt(np.mean(x**2, axis=1, keepdims=True))  # (2,1)
        return None

    def _apply_normalize(self, x, mix_rms):
        if self.normalize == 'none' or mix_rms is None:
            return x
        return x / np.maximum(mix_rms, 1e-6)

    def __getitem__(self, idx):
        self._ensure_open()

        # carga mezcla y stems
        xm = self.hm["dataset"][idx]                  # (2, T)
        stems = [h["dataset"][idx] for h in self.hs]  # lista de (2, T)

        # recorta todos al mínimo T
        T = min([xm.shape[-1]] + [s.shape[-1] for s in stems])
        xm = xm[:, :T]
        stems = [s[:, :T] for s in stems]

        # crop o pad
        if self.segment is not None:
            if T > self.segment:
                start = random.randint(0, T - self.segment)
                end = start + self.segment
                xm = xm[:, start:end]
                stems = [s[:, start:end] for s in stems]
            elif T < self.segment:
                pad = self.segment - T
                xm = np.pad(xm, ((0, 0), (0, pad)))
                stems = [np.pad(s, ((0, 0), (0, pad))) for s in stems]

        # normalización
        mix_rms = self._compute_rms(xm)
        xm = self._apply_normalize(xm, mix_rms)
        stems = [self._apply_normalize(s, mix_rms) for s in stems]

        # añadir ruido si procede
        if self.snr_range is not None:
            snr_db = random.uniform(*self.snr_range)
            pwr = np.mean(xm**2)
            noise_pwr = pwr / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_pwr), xm.shape)
            xm = xm + noise

        # salida
        y = np.stack(stems, axis=0)  # (S, 2, Tseg)
        return self._as_tensor(xm), self._as_tensor(y)
