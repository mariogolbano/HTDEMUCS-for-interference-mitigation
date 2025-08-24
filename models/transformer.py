# models/transformer.py
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """Standard 1D sinusoidal positional encodings."""
    def __init__(self, d_model, max_len=100000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)

class CrossDomainBlock(nn.Module):
    """
    One layer comprising:
    - Self-attn on time tokens
    - Self-attn on spec tokens
    - Cross-attn time<-spec and spec<-time (bidirectional)
    - FFNs with residual + pre-norm
    """
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_s1 = nn.LayerNorm(d_model)
        self.self_attn_t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_s = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_t2 = nn.LayerNorm(d_model)
        self.norm_s2 = nn.LayerNorm(d_model)
        self.cross_t_from_s = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_s_from_t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_t3 = nn.LayerNorm(d_model)
        self.norm_s3 = nn.LayerNorm(d_model)
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_s = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, t_tokens, s_tokens, attn_mask_t=None, attn_mask_s=None):
        # Self-attention in each domain
        tt = self.norm_t1(t_tokens)
        ts, _ = self.self_attn_t(tt, tt, tt, attn_mask=attn_mask_t, need_weights=False)
        t_tokens = t_tokens + ts

        ss = self.norm_s1(s_tokens)
        ss2, _ = self.self_attn_s(ss, ss, ss, attn_mask=attn_mask_s, need_weights=False)
        s_tokens = s_tokens + ss2

        # Cross-attention (bidirectional)
        tt2 = self.norm_t2(t_tokens)
        ss2 = self.norm_s2(s_tokens)
        t_from_s, _ = self.cross_t_from_s(tt2, ss2, ss2, need_weights=False)  # Q=t, K/V=s
        s_from_t, _ = self.cross_s_from_t(ss2, tt2, tt2, need_weights=False)  # Q=s, K/V=t
        t_tokens = t_tokens + t_from_s
        s_tokens = s_tokens + s_from_t

        # FFNs
        tff = self.ffn_t(self.norm_t3(t_tokens))
        sff = self.ffn_s(self.norm_s3(s_tokens))
        t_tokens = t_tokens + tff
        s_tokens = s_tokens + sff

        return t_tokens, s_tokens

class CrossDomainTransformer(nn.Module):
    """
    Stack of CrossDomainBlocks with sinusoidal positional encodings.
    Inputs are sequences of shape (B, T_steps, D) for both domains (time/spec).
    """
    def __init__(self, d_model=512, nhead=8, num_layers=5, dim_feedforward=2048, dropout=0.0, max_len=100000):
        super().__init__()
        self.pe_t = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.pe_s = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            CrossDomainBlock(d_model, nhead, dim_feedforward, dropout)
        for _ in range(num_layers)])

    def forward(self, t_tokens, s_tokens):
        t_tokens = self.pe_t(t_tokens)
        s_tokens = self.pe_s(s_tokens)
        for layer in self.layers:
            t_tokens, s_tokens = layer(t_tokens, s_tokens)
        return t_tokens, s_tokens
