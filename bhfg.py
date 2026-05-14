"""
Bio-Harmonic Frequency Gate (B-HFG)
=====================================
Section 3.2 of DenseFish-v13.

Operates in the 2-D FFT frequency domain to:
  • preserve quasi-periodic biological textures (fish scales, contours)
  • suppress broadband aeration-bubble noise

Pipeline
--------
  Input X  ──►  FFT  ──►  magnitude ⊙ σ(LearnableAttention)
                          phase unchanged
               IFFT  ──►  X_clean

The learnable Harmonic Attention Map ℳ ∈ ℝ^{H×W×C} is implemented as
  1×1 Conv → BN → Sigmoid
applied to the log-magnitude spectrum.

Additionally, the Hydrodynamic-Informed Constraint (HIC) Loss is defined
here as a callable that penalises physically-impossible centroid jumps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HarmonicAttention(nn.Module):
    """
    Lightweight 1×1-conv attention over the log-magnitude spectrum.
    Input/output: (B, C, H, W)  — treated as spatial feature maps.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, log_mag: torch.Tensor) -> torch.Tensor:
        return self.net(log_mag)           # (B, C, H, W) ∈ [0,1]


class BioHarmonicFrequencyGate(nn.Module):
    """
    B-HFG module (Eq. 1–2 in the paper).

    Args:
        channels : number of feature channels C
        reduction: channel compression ratio for the attention branch
        eps      : small constant for numerical stability in log

    Forward input  : X  ∈ ℝ^{B×C×H×W}
    Forward output : X_clean ∈ ℝ^{B×C×H×W}   (real-valued)
    """

    def __init__(self, channels: int, reduction: int = 4, eps: float = 1e-8):
        super().__init__()
        self.channels  = channels
        self.eps       = eps
        self.attention = HarmonicAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Step 1: 2-D FFT  (Eq. 1) ──────────────────────────────────────
        # torch.fft.rfft2 returns complex spectrum of shape (B, C, H, W//2+1)
        F_x = torch.fft.rfft2(x, norm="ortho")          # complex64

        mag   = F_x.abs()                                # |F(X)|
        phase = torch.angle(F_x)                         # ∠F(X)

        # ── Step 2: learnable harmonic attention ────────────────────────────
        # We feed the log-magnitude into a spatial attention module.
        # To keep H×W consistent we use rfft2 output size (H, W//2+1)
        # and let the conv operate on that grid.
        log_mag = torch.log(mag + self.eps)              # (B, C, H, W//2+1)
        gate    = self.attention(log_mag)                # (B, C, H, W//2+1)

        # ── Step 3: modulate magnitude, preserve phase  (Eq. 2) ────────────
        mag_filtered = mag * gate                        # element-wise
        F_clean = torch.polar(mag_filtered, phase)       # complex from r & θ

        # ── Step 4: inverse FFT → spatial domain ────────────────────────────
        x_clean = torch.fft.irfft2(F_clean, s=(H, W), norm="ortho")

        return x_clean.contiguous()

    # ── Visualisation helper ─────────────────────────────────────────────────

    @torch.no_grad()
    def spectral_heatmap(self, x: torch.Tensor) -> dict:
        """
        Return dict with 'before' and 'after' log-magnitude averaged over
        channels – useful for Figure 9 in the paper.
        """
        F_x     = torch.fft.rfft2(x, norm="ortho")
        mag_in  = F_x.abs().mean(1)                      # (B, H, W//2+1)
        x_c     = self(x)
        F_c     = torch.fft.rfft2(x_c, norm="ortho")
        mag_out = F_c.abs().mean(1)
        return {"before": torch.log(mag_in + 1e-8),
                "after":  torch.log(mag_out + 1e-8)}


# ──────────────────────────────────────────────────────────────────────────────
# Hydrodynamic-Informed Constraint (HIC) Loss
# ──────────────────────────────────────────────────────────────────────────────

class HICLoss(nn.Module):
    """
    Hydrodynamic-Informed Constraint Loss (Section 3.2).

    Penalises physically-implausible centroid acceleration between
    consecutive frame predictions.

    Given predicted centroids across T frames for M tracks:
        centroids : (B, M, T, 2)   where dim-2 = (cx, cy)

    The loss is the L2 norm of the second-order finite-difference
    (i.e. centroid acceleration), clamped by a physical upper bound
    derived from the Navier-Stokes kinematic constraint.

        a_t = p_{t+1} - 2·p_t + p_{t-1}
        L_HIC = mean( relu(|a_t| - a_max) )

    Args:
        a_max    : maximum physically plausible acceleration (pixels/frame²)
        weight   : loss weighting coefficient
    """

    def __init__(self, a_max: float = 15.0, weight: float = 0.05):
        super().__init__()
        self.a_max  = a_max
        self.weight = weight

    def forward(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        centroids : (B, M, T, 2)
        Returns scalar loss.
        """
        if centroids.shape[2] < 3:
            return centroids.new_zeros(1).squeeze()

        # finite-difference acceleration
        acc = (centroids[:, :, 2:, :]
               - 2 * centroids[:, :, 1:-1, :]
               + centroids[:, :, :-2, :])                # (B, M, T-2, 2)

        acc_norm = acc.norm(dim=-1)                       # (B, M, T-2)
        loss = F.relu(acc_norm - self.a_max).mean()
        return self.weight * loss


# ──────────────────────────────────────────────────────────────────────────────
# Quick unit test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 64, 80, 80)
    bhfg = BioHarmonicFrequencyGate(channels=64)
    out  = bhfg(x)
    print(f"B-HFG  in: {x.shape}  → out: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    # spectral visualisation
    hm = bhfg.spectral_heatmap(x)
    print(f"  Heatmap 'before': {hm['before'].shape}, "
          f"'after': {hm['after'].shape}")

    # HIC loss
    hic  = HICLoss(a_max=15.0)
    traj = torch.randn(2, 10, 5, 2)          # B=2, M=10 fish, T=5 frames
    lv   = hic(traj)
    print(f"HIC loss value: {lv.item():.4f}")
    print("B-HFG + HIC unit test passed ✓")
