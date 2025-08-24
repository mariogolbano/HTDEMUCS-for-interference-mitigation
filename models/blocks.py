# models/blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def glu(x):
    # Gated Linear Unit: split channels in half and gate
    a, b = x.chunk(2, dim=1)
    return a * torch.sigmoid(b)

class DConv1d(nn.Module):
    """
    Simple dilated conv residual block: two conv1d with increasing dilation.
    """
    def __init__(self, channels, kernel=3, dilations=(1, 2), groups=1):
        super().__init__()
        pads = [ (kernel-1)//2 * d for d in dilations ]
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel, padding=pads[0], dilation=dilations[0], groups=groups),
            nn.Conv1d(channels, channels, kernel, padding=pads[1], dilation=dilations[1], groups=groups),
        ])
        self.norms = nn.ModuleList([nn.GroupNorm(1, channels), nn.GroupNorm(1, channels)])

    def forward(self, x):
        res = x
        for conv, gn in zip(self.convs, self.norms):
            x = conv(x)
            x = F.gelu(gn(x))
        return x + res

class Enc1d(nn.Module):
    """
    Encoder stage for time branch: Conv1d (k=8,s=4) -> GN -> 1x1 Conv -> GLU -> DConv residual
    """
    def __init__(self, in_ch, out_ch, k=8, s=4, groups=1):
        super().__init__()
        pad = (k - s) // 2  # keep divisibility
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, groups=groups)
        self.norm = nn.GroupNorm(1, out_ch)
        self.rewrite = nn.Conv1d(out_ch, out_ch * 2, kernel_size=1)
        self.dconv = DConv1d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = glu(self.rewrite(x))
        x = self.dconv(x)
        return x

class Dec1d(nn.Module):
    """
    Decoder stage for time branch: ConvTranspose1d (k=8,s=4) + skip -> GN -> 1x1 Conv -> GLU -> DConv residual
    """
    def __init__(self, in_ch, out_ch, k=8, s=4, groups=1):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=s, padding=(k - s)//2, output_padding=0, groups=groups)
        self.norm = nn.GroupNorm(1, out_ch)
        self.rewrite = nn.Conv1d(out_ch, out_ch * 2, kernel_size=1)
        self.dconv = DConv1d(out_ch)

    def forward(self, x, skip):
        x = self.deconv(x)
        # Align time axis if off-by-one
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, max(0, diff)))
            if diff < 0:
                x = x[..., :skip.shape[-1]]
        x = x + skip
        x = self.norm(x)
        x = glu(self.rewrite(x))
        x = self.dconv(x)
        return x

# Enc2d: no hagas mean(F) dentro; devuelve 2D y, si quieres, añade una rama 1D para tokens
class Enc2dFull(nn.Module):
    def __init__(self, in_ch, out_ch, k=(4,8), s=(2,4), groups=1):
        super().__init__()
        pad = ((k[0]-s[0])//2, (k[1]-s[1])//2)
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, groups=groups)
        self.norm2d = nn.GroupNorm(1, out_ch)
        self.rewrite = nn.Conv2d(out_ch, out_ch*2, kernel_size=1)
        self.post_dconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=1),
        )
    def forward(self, x):              # x: (B, C, F, S)
        x = self.conv2d(x)
        x = self.norm2d(x)
        x = glu(self.rewrite(x))
        x = x + self.post_dconv(x)     # ligera ref.
        return x                       # (B, C, F', S')

class Dec2dFull(nn.Module):
    def __init__(self, in_ch, out_ch, k=(4,8), s=(2,4), groups=1):
        super().__init__()
        pad = ((k[0]-s[0])//2, (k[1]-s[1])//2)
        self.deconv2d = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, output_padding=(0,0), groups=groups)
        self.norm2d = nn.GroupNorm(1, out_ch)
        self.rewrite = nn.Conv2d(out_ch, out_ch*2, kernel_size=1)
        self.post_dconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
    def forward(self, x, skip):        # x, skip: (B, C, F, S)
        x = self.deconv2d(x)
        # alinear F,S si hay off-by-one
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, max(0, diff), 0, 0))
            if diff < 0: x = x[..., :skip.shape[-1]]
        if x.shape[-2] != skip.shape[-2]:
            diffF = skip.shape[-2] - x.shape[-2]
            x = F.pad(x, (0,0, 0, max(0, diffF)))
            if diffF < 0: x = x[:, :, :skip.shape[-2], :]
        x = x + skip
        x = self.norm2d(x)
        x = glu(self.rewrite(x))
        x = x + self.post_dconv(x)
        return x
