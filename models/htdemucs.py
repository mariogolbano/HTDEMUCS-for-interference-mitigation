# models/htdemucs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .stft import STFT, ISTFT
from .blocks import Enc1d, Dec1d, Enc2dFull, Dec2dFull
from .transformer import CrossDomainTransformer

class HTDemucs(nn.Module):
    """
    HT-Demucs con dos ramas (tiempo y espectro) y selector de arquitectura:
      - arch='hybrid' : usa ambas ramas y suma en el dominio temporal.
      - arch='time'   : sólo rama temporal.
      - arch='spec'   : sólo rama espectral (iSTFT para volver a tiempo).
    La salida SIEMPRE es (B, stems, C, T) en dominio temporal (I/Q), para que
    la función de pérdida, logs, y scripts no cambien.
    """
    def __init__(self, in_channels=2, stems=1, audio_channels=2,
                 n_fft=4096, depth=4, base_channels=64,
                 transformer_dim=512, transformer_heads=8, transformer_layers=5,
                 segment_length=None, arch: str = "hybrid"):
        super().__init__()
        assert arch in ("hybrid", "time", "spec")
        self.arch = arch
        self.stems = stems
        self.audio_channels = audio_channels
        self.segment_length = segment_length

        # STFT/iSTFT (siempre disponibles; iSTFT sólo se usa en 'spec' o 'hybrid')
        self.stft = STFT(n_fft=n_fft)
        self.istft = ISTFT(n_fft=n_fft)

        # ===== Rama TIEMPO =====
        ch = base_channels
        enc_t = []
        in_ch = in_channels
        for _ in range(depth):
            enc_t.append(Enc1d(in_ch, ch, k=8, s=4))
            in_ch = ch
            ch *= 2
        self.enc_time = nn.ModuleList(enc_t)
        self.bottleneck_dim_t = in_ch

        self.proj_t   = nn.Conv1d(self.bottleneck_dim_t, transformer_dim, kernel_size=1)
        self.unproj_t = nn.Conv1d(transformer_dim, self.bottleneck_dim_t, kernel_size=1)

        dec_t = []
        ch = self.bottleneck_dim_t
        for i in range(depth-1, -1, -1):
            out_ch = (self.audio_channels if i == 0 else ch // 2)
            dec_t.append(Dec1d(ch, out_ch, k=8, s=4))
            ch = out_ch
        self.dec_time = nn.ModuleList(dec_t)
        # Head temporal produce stems*C en tiempo
        self.time_head = nn.Conv1d(self.audio_channels, self.audio_channels * self.stems, kernel_size=1)

        # ===== Rama ESPECTRO =====
        ch = base_channels
        enc_s = []
        in_ch = in_channels * 2  # CAC: (C*2, F, S) -> Re/Im por canal
        for _ in range(depth):
            enc_s.append(Enc2dFull(in_ch, ch, k=(4,8), s=(2,4)))
            in_ch = ch
            ch *= 2
        self.enc_spec = nn.ModuleList(enc_s)
        self.bottleneck_dim_s = in_ch

        self.proj_s   = nn.Conv1d(self.bottleneck_dim_s, transformer_dim, kernel_size=1)
        self.unproj_s = nn.Conv1d(transformer_dim, self.bottleneck_dim_s, kernel_size=1)

        dec_s = []
        ch = self.bottleneck_dim_s
        for i in range(depth-1, -1, -1):
            out_ch = (self.audio_channels * 2 if i == 0 else ch // 2)
            dec_s.append(Dec2dFull(ch, out_ch, k=(4,8), s=(2,4)))
            ch = out_ch
        self.dec_spec = nn.ModuleList(dec_s)
        # Head espectral mapea C*2 -> stems*(C*2) en CAC
        self.spec_head = nn.Conv2d(self.audio_channels * 2, self.audio_channels * 2 * self.stems, kernel_size=1)

        # ===== Transformer cruzado tiempo <-> espectro =====
        self.xformer = CrossDomainTransformer(
            d_model=transformer_dim, nhead=transformer_heads,
            num_layers=transformer_layers, dim_feedforward=transformer_dim*4, dropout=0.0
        )

    # ---------- helpers ----------
    def _down_time(self, x):
        skips = []
        for enc in self.enc_time:
            skips.append(x)
            x = enc(x)
        return x, skips

    def _up_time(self, x, skips):
        for dec in self.dec_time:
            skip = skips.pop()
            x = dec(x, skip)
        return x  # (B, C=audio_channels, T)

    def _down_spec(self, Zc):
        skips = []
        x = Zc
        for enc in self.enc_spec:
            skips.append(x)
            x = enc(x)
        return x, skips

    def _up_spec(self, x2d, skips):
        for dec in self.dec_spec:
            skip = skips.pop()
            x2d = dec(x2d, skip)
        return x2d  # (B, C*2, F, S) en CAC

    def _xformer_tokens(self, x_t_1d, x_s_2d):
        """
        x_t_1d: (B, Ct, S_t)
        x_s_2d: (B, Cs, Fk, S_k)
        Proyecta a D, corre transformer (secuencia a lo largo de S) y devuelve
        actualizaciones inyectadas de vuelta a canales originales.
        """
        # tokens espectrales por promediado sobre F
        s_tokens = x_s_2d.mean(dim=2)  # (B, Cs, S)
        # proyecta a D y corre transformer
        t = self.proj_t(x_t_1d).permute(0, 2, 1)  # (B, S, D)
        s = self.proj_s(s_tokens).permute(0, 2, 1)
        t, s = self.xformer(t, s)
        # des-proyecta
        t = self.unproj_t(t.permute(0, 2, 1))  # (B, Ct, S)
        s = self.unproj_s(s.permute(0, 2, 1))  # (B, Cs, S)
        # inyección de vuelta al mapa 2D por broadcasting sobre F
        s = s.unsqueeze(2).expand(-1, -1, x_s_2d.shape[2], -1)  # (B, Cs, F, S)
        x_s_2d = x_s_2d + s
        x_t_1d = x_t_1d + t
        return x_t_1d, x_s_2d

    # ---------- forward ----------
    def forward(self, x):
        """
        x: (B, C, T)  mezcla I/Q
        return: (B, stems, C, T) en tiempo
        """
        B, C, T = x.shape

        # Precalcula STFT sólo si hace falta
        need_spec = (self.arch != "time")
        if need_spec:
            Zc = self.stft(x)  # (B, C*2, F, S)
            Fbins, Sframes = Zc.shape[-2], Zc.shape[-1]

        # Down-paths según arquitectura
        if self.arch in ("hybrid", "time"):
            xt, skips_t = self._down_time(x)   # (B, Ct, S_t)
        if self.arch in ("hybrid", "spec"):
            xs2d, skips_s = self._down_spec(Zc)  # (B, Cs, Fk, S_k)

        # Transformer cruzado sólo si 'hybrid'; si no, evitamos coste extra
        if self.arch == "hybrid":
            xt, xs2d = self._xformer_tokens(xt, xs2d)

        # Up-paths y cabezas
        y_time = None
        if self.arch in ("hybrid", "time"):
            y_time_feat = self._up_time(xt, skips_t)           # (B, C, T)
            y_time = self.time_head(y_time_feat)               # (B, stems*C, T)

        y_spec_time = None
        if self.arch in ("hybrid", "spec"):
            y_spec_feat2d = self._up_spec(xs2d, skips_s)       # (B, C*2, F', S')
            # Alinea a (Fbins, Sframes) si hiciera falta
            Fp, Sp = y_spec_feat2d.shape[-2], y_spec_feat2d.shape[-1]
            if (Fp != Fbins) or (Sp != Sframes):
                pad_F = max(0, Fbins - Fp)
                pad_S = max(0, Sframes - Sp)
                y_spec_feat2d = F.pad(y_spec_feat2d, (0, pad_S, 0, pad_F))
                y_spec_feat2d = y_spec_feat2d[..., :Fbins, :Sframes]
            # Mapea a stems*(C*2) y aplica iSTFT por stem -> (B, stems, C, T)
            y_spec_cac = self.spec_head(y_spec_feat2d)  # (B, stems*(C*2), F, S)
            y_spec_cac = y_spec_cac.view(B, self.stems, self.audio_channels * 2, Fbins, Sframes)
            # iSTFT en fp32 y sin autocast para estabilidad
            outs = []
            with torch.amp.autocast(device_type="cuda", enabled=False):
                for s in range(self.stems):
                    cac = y_spec_cac[:, s, :, :, :]                 # (B, C*2, F, S)
                    x_time = self.istft(cac, length=T)              # (B, C, T)
                    outs.append(x_time)
            y_spec_time = torch.stack(outs, dim=1)  # (B, stems, C, T)

        # Unifica salidas al dominio temporal
        if self.arch == "time":
            y = y_time.view(B, self.stems, self.audio_channels, T)
        elif self.arch == "spec":
            y = y_spec_time
        else:  # hybrid
            # y_time: (B, stems*C, T) -> (B, stems, C, T)
            y_time = y_time.view(B, self.stems, self.audio_channels, T)
            # Suma simple (sin parámetros) para no romper checkpoints antiguos
            y = y_time + y_spec_time

        return y
