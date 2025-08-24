# models/stft.py
import torch
import torch.nn as nn
import math

class STFT(nn.Module):
    """
    STFT wrapper for multi-channel I/Q signals.
    - Input:  (B, C, T) real-valued (I/Q channels).
    - Output: complex as channels (Re, Im) => (B, C*2, F, S) where F = n_fft//2 + 1, S = #frames.
    Uses Hann window, hop = n_fft//4, center=True, normalized=True.
    """
    def __init__(self, n_fft=4096, hop=None, win=None, device=None, dtype=torch.float32):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop) if hop is not None else self.n_fft // 4
        self.win = win
        if self.win is None:
            window = torch.hann_window(self.n_fft, periodic=True, dtype=dtype)
        else:
            window = self.win
        self.register_buffer("window", window)
        self.device_ = device
        self.dtype_ = dtype

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        Z = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
            normalized=True,
            pad_mode="reflect",
        )  # (B*C, F, S) complex
        Zr = Z.real
        Zi = Z.imag
        Zc = torch.stack([Zr, Zi], dim=1)  # (B*C, 2, F, S)
        Zc = Zc.reshape(B, C * 2, Zc.shape[-2], Zc.shape[-1])  # (B, C*2, F, S)
        return Zc

class ISTFT(nn.Module):
    """
    Inverse STFT counterpart.
    - Input: complex as channels (B, C*2, F, S).
    - Output: (B, C, T)
    """
    def __init__(self, n_fft=4096, hop=None, win=None, device=None, dtype=torch.float32, length=None):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop) if hop is not None else self.n_fft // 4
        self.win = win
        if self.win is None:
            window = torch.hann_window(self.n_fft, periodic=True, dtype=dtype)
        else:
            window = self.win
        self.register_buffer("window", window)
        self.length = length

    def forward(self, Zc, length=None):
        """
        Zc: (B, C*2, F, S) en formato CAC -> reconstruye complejo e iSTFT por canal.
        Retorna: (B, C, T)
        """
        B, CC2, Fbins, Sframes = Zc.shape
        assert CC2 % 2 == 0, "C*2 channels expected for CAC input"
        C = CC2 // 2

        # Asegura float32 antes de construir complejos (evita problemas con bf16)
        Zc_f32 = Zc.to(torch.float32)

        Zc_f32 = Zc_f32.reshape(B * C, 2, Fbins, Sframes)
        Z_complex = torch.complex(Zc_f32[:, 0], Zc_f32[:, 1])  # (B*C, F, S)

        # Desactiva autocast durante la iSTFT para estabilidad numérica
        target_len = length if length is not None else self.length
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = torch.istft(
                Z_complex,
                n_fft=self.n_fft,
                hop_length=self.hop,
                window=self.window,
                center=True,
                normalized=True,
                length=target_len,
            )  # (B*C, T)

        x = x.reshape(B, C, x.shape[-1])  # (B, C, T)
        return x