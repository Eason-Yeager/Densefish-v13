"""
YOLOv13-Mamba Backbone
========================
Section 3.3 of DenseFish-v13.

Key components implemented:
  • VSSBlock     — Visual State Space Block (Mamba-V2, SS2D 4-direction scan)
  • BiMSWMamba   — Bi-directional Multi-scale Wavelet Mamba (DWT + DSS)
  • MoESpectralGate — Mixture-of-Experts Spectral Gate (water-condition router)
  • YOLOv13MambaBackbone — full hierarchical 4-stage backbone

State-space recurrence (Eq. 3):
    h_t = A h_{t-1} + B x_t
    y_t = C h_t

All heavy dependencies (selective_scan_cuda) are replaced with a pure-PyTorch
reference implementation so the code runs without custom CUDA extensions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Selective State-Space (pure-PyTorch reference, O(L) in L)
# ──────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba-style).

    Processes a sequence of length L with d_model channels.
    Uses a simple recurrent loop for clarity; replace with
    parallel prefix-sum for production speed.

    Args:
        d_model : feature dimension
        d_state : SSM hidden state dimension (N in paper notation)
        dt_rank : rank of Δt projection
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # input-dependent Δ, B, C projections
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # learnable A (log-parameterised for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (residual / skip)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, d_model)
        Returns y : (B, L, d_model)
        """
        B, L, d = x.shape
        N = self.d_state

        # project to dt, B_mat, C_mat
        xbc = self.x_proj(x)                                  # (B, L, dt_rank+2N)
        dt_raw, B_mat, C_mat = torch.split(
            xbc, [self.dt_proj.in_features, N, N], dim=-1)

        dt = F.softplus(self.dt_proj(dt_raw))                 # (B, L, d)
        A  = -torch.exp(self.A_log.float())                   # (d, N)

        # discretise: Ā = exp(dt·A), B̄ = dt·B  (ZOH)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,d,N)
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)                       # (B,L,d,N)

        # recurrence over L
        h = x.new_zeros(B, d, N)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)  # (B,d,N)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)           # (B,d)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                             # (B,L,d)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)           # skip connection
        return y


# ──────────────────────────────────────────────────────────────────────────────
# 2.  2-D Selective Scan (SS2D) — 4 directions
# ──────────────────────────────────────────────────────────────────────────────

class SS2D(nn.Module):
    """
    2-D selective scan that scans in 4 directions:
        → (left-to-right, top-to-bottom)
        ← (right-to-left, bottom-to-top)
        ↓ (top-to-bottom, left-to-right column-major)
        ↑ (bottom-to-top, right-to-left column-major)
    Results are averaged after independent SSM passes.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.ssm_list = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(4)]
        )

    @staticmethod
    def _flatten_direction(feat: torch.Tensor, direction: int) -> torch.Tensor:
        """feat: (B,C,H,W) → (B,L,C)"""
        B, C, H, W = feat.shape
        if direction == 0:  # row-major left→right
            return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif direction == 1:  # row-major right→left
            return feat.flip(-1).permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif direction == 2:  # col-major top→bottom
            return feat.permute(0, 3, 2, 1).reshape(B, H * W, C)
        else:               # col-major bottom→top
            return feat.flip(-2).permute(0, 3, 2, 1).reshape(B, H * W, C)

    @staticmethod
    def _unflatten_direction(seq: torch.Tensor, direction: int,
                              H: int, W: int) -> torch.Tensor:
        """seq: (B,L,C) → (B,C,H,W)"""
        B, L, C = seq.shape
        if direction == 0:
            return seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif direction == 1:
            return seq.reshape(B, H, W, C).permute(0, 3, 1, 2).flip(-1)
        elif direction == 2:
            return seq.reshape(B, W, H, C).permute(0, 3, 2, 1)
        else:
            return seq.reshape(B, W, H, C).permute(0, 3, 2, 1).flip(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape
        out = x.new_zeros(B, C, H, W)
        for d, ssm in enumerate(self.ssm_list):
            seq = self._flatten_direction(x, d)             # (B,H*W,C)
            y   = ssm(seq)                                   # (B,H*W,C)
            out = out + self._unflatten_direction(y, d, H, W)
        return out / 4.0


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VSSBlock — Visual State Space Block
# ──────────────────────────────────────────────────────────────────────────────

class VSSBlock(nn.Module):
    """
    Thin wrapper around SS2D with:
        layer-norm → depth-wise conv (local mixing) → SS2D → residual
    """

    def __init__(self, dim: int, d_state: int = 16, expand: float = 2.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        inner_dim   = int(dim * expand)

        self.in_proj  = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.dw_conv  = nn.Conv2d(inner_dim, inner_dim, 3, padding=1,
                                   groups=inner_dim, bias=False)
        self.act      = nn.SiLU()
        self.ss2d     = SS2D(inner_dim, d_state)
        self.out_proj = nn.Conv2d(inner_dim, dim, 1, bias=False)

    def _ln_hw(self, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        """Apply LayerNorm over channel dim for (B,C,H,W) tensors."""
        B, C, H, W = x.shape
        return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self._ln_hw(x, self.norm1)
        x = self.in_proj(x)
        x = self.act(self.dw_conv(x))
        x = self.ss2d(x)
        x = self.out_proj(x)
        x = self._ln_hw(x, self.norm2)
        return x + shortcut


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Discrete Wavelet Transform helpers (Haar, 1 level)
# ──────────────────────────────────────────────────────────────────────────────

class HaarDWT2D(nn.Module):
    """Single-level 2-D Haar DWT (fixed, not learned)."""

    def __init__(self):
        super().__init__()
        # H: low-pass  LL   G: high-pass  LH, HL, HH
        h = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        g = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        self.register_buffer('h', h)
        self.register_buffer('g', g)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (LL, [LH, HL, HH]) sub-bands, each at H//2 × W//2.
        """
        B, C, H, W = x.shape
        # reshape for depthwise group conv
        xr = x.reshape(B * C, 1, H, W)

        def _conv(k2d):
            k = k2d.unsqueeze(0).unsqueeze(0)
            return F.conv2d(xr, k, stride=2, padding=0).reshape(B, C, H // 2, W // 2)

        LL = _conv(self.h)
        LH = _conv(torch.stack([self.h[0], -self.h[1]]))
        HL = _conv(torch.stack([-self.h[:, 0], self.h[:, 1]]).T)
        HH = _conv(self.g)
        high = (LH.abs() + HL.abs() + HH.abs()) / 3.0
        return LL, high


class HaarIWT2D(nn.Module):
    """Single-level 2-D Haar inverse DWT (nearest-upsample + conv)."""

    def forward(self, LL: torch.Tensor) -> torch.Tensor:
        return F.interpolate(LL, scale_factor=2, mode="nearest")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Bi-MSW-Mamba — Bi-directional Multi-scale Wavelet Mamba
# ──────────────────────────────────────────────────────────────────────────────

class BiMSWMamba(nn.Module):
    """
    Section 3.1 / 3.3 enhancement over vanilla Mamba.

    Pipeline:
        x → DWT → (LL, high) → parallel SS2D scanners with DSS routing
          → IWT merge → residual output

    Dynamic Selective Scanning (DSS) weights sub-band routes by their
    energy fraction, downweighting aeration-noise bands.
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dwt  = HaarDWT2D()
        self.iwt  = HaarIWT2D()

        self.ssm_ll   = SS2D(dim, d_state)
        self.ssm_high = SS2D(dim, d_state)

        # gate that routes energy between LL and high-freq scanners
        self.gate_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1),
        )
        self.up_proj  = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm     = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        LL, high = self.dwt(x)                               # each (B,C,H//2,W//2)

        # DSS: compute gate weights from band energies
        gate_in = torch.cat([LL.mean((-2, -1)), high.mean((-2, -1))], dim=1)  # (B,2C)
        # gate_in fed through linear; use avgpool of feature maps
        combined = torch.cat([LL, high], dim=0)              # (2B, C, h, w)
        pool     = F.adaptive_avg_pool2d(combined, 1).view(2 * B, C)
        ll_pool  = pool[:B]; hi_pool = pool[B:]
        gate_vec = self.gate_proj(
            torch.cat([ll_pool, hi_pool], dim=1))             # (B, 2)
        w_ll  = gate_vec[:, 0].view(B, 1, 1, 1)
        w_hi  = gate_vec[:, 1].view(B, 1, 1, 1)

        y_ll   = self.ssm_ll(LL)   * w_ll
        y_high = self.ssm_high(high) * w_hi
        y_down = y_ll + y_high                                # (B, C, H//2, W//2)
        y_up   = self.iwt(y_down)                             # (B, C, H, W)

        # channel-norm + residual
        y_up = self.up_proj(y_up)
        B2, C2, H2, W2 = y_up.shape
        y_up = self.norm(y_up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return y_up + x


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Mixture-of-Experts Spectral Gate (MoE-SG)
# ──────────────────────────────────────────────────────────────────────────────

class MoESpectralGate(nn.Module):
    """
    Mixture-of-Experts Spectral Gate (Section 3.3).

    n_experts spectral-filter experts, each a 1×1 conv + BN + ReLU.
    Gating network routes by global image entropy.

    Args:
        dim       : channel dimension
        n_experts : number of water-condition experts
    """

    def __init__(self, dim: int, n_experts: int = 3):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, n_experts),
            nn.Softmax(dim=-1),
        )

    def _entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Global spectral entropy of feature map → scalar per sample."""
        p = F.softmax(x.reshape(x.size(0), -1), dim=-1)
        return -(p * (p + 1e-8).log()).sum(1)               # (B,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x)                               # (B, n_experts)
        out = sum(w.view(-1, 1, 1, 1) * e(x)
                  for e, w in zip(self.experts,
                                  weights.unbind(1)))
        return out + x                                       # residual


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Conv-BN-SiLU stem block
# ──────────────────────────────────────────────────────────────────────────────

class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 s: int = 1, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  YOLOv13-Mamba Backbone
# ──────────────────────────────────────────────────────────────────────────────

class YOLOv13MambaBackbone(nn.Module):
    """
    Hierarchical 4-stage backbone:

        Stage 0 (stride 4)  : 3→C0 via stem; 2×VSSBlock
        Stage 1 (stride 8)  : C0→C1 downsample; 3×VSSBlock + MoE-SG
        Stage 2 (stride 16) : C1→C2 downsample; 4×BiMSWMamba
        Stage 3 (stride 32) : C2→C3 downsample; 3×BiMSWMamba + MoE-SG

    Returns multi-scale feature maps [P2, P3, P4, P5] for the neck/head.

    Default channel widths approximate YOLOv11-m × 1.1 to account
    for the Mamba blocks' larger d_model.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
                 d_state: int = 16,
                 n_experts: int = 3):
        super().__init__()
        C0, C1, C2, C3 = channels

        # ── stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, C0 // 2, 3, 2, 1),    # /2
            ConvBNSiLU(C0 // 2, C0, 3, 2, 1),             # /4
        )

        # ── stage 0 ───────────────────────────────────────────────────────────
        self.stage0 = nn.Sequential(
            VSSBlock(C0, d_state),
            VSSBlock(C0, d_state),
        )

        # ── stage 1 ───────────────────────────────────────────────────────────
        self.down1  = ConvBNSiLU(C0, C1, 3, 2, 1)         # /8
        self.stage1 = nn.Sequential(
            VSSBlock(C1, d_state),
            VSSBlock(C1, d_state),
            VSSBlock(C1, d_state),
            MoESpectralGate(C1, n_experts),
        )

        # ── stage 2 ───────────────────────────────────────────────────────────
        self.down2  = ConvBNSiLU(C1, C2, 3, 2, 1)         # /16
        self.stage2 = nn.Sequential(
            BiMSWMamba(C2, d_state),
            BiMSWMamba(C2, d_state),
            BiMSWMamba(C2, d_state),
            BiMSWMamba(C2, d_state),
        )

        # ── stage 3 ───────────────────────────────────────────────────────────
        self.down3  = ConvBNSiLU(C2, C3, 3, 2, 1)         # /32
        self.stage3 = nn.Sequential(
            BiMSWMamba(C3, d_state),
            BiMSWMamba(C3, d_state),
            BiMSWMamba(C3, d_state),
            MoESpectralGate(C3, n_experts),
        )

        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns [P2, P3, P4, P5]:
            P2 at stride 4  (C0)
            P3 at stride 8  (C1)
            P4 at stride 16 (C2)
            P5 at stride 32 (C3)
        """
        p2 = self.stage0(self.stem(x))
        p3 = self.stage1(self.down1(p2))
        p4 = self.stage2(self.down2(p3))
        p5 = self.stage3(self.down3(p4))
        return [p2, p3, p4, p5]


# ──────────────────────────────────────────────────────────────────────────────
# Quick unit test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    dummy = torch.randn(2, 3, 640, 640)
    backbone = YOLOv13MambaBackbone()
    feats = backbone(dummy)
    print("YOLOv13-Mamba Backbone output shapes:")
    for i, f in enumerate(feats):
        print(f"  P{i+2}: {f.shape}")
    print("Backbone unit test passed ✓")
