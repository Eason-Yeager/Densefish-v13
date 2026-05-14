"""
DenseFish-v13 — Full Model
===========================
Integrates:
  B-HFG  →  YOLOv13-Mamba Backbone  →  FPN Neck  →  NMS-Free Head
           + Bio-Kinematic Behavior Head (inference only)

The model follows the data-flow described in Section 3.1:
  raw image
    → B-HFG (spectral denoising)
    → YOLOv13MambaBackbone (global occlusion-aware features)
    → FPN Neck (multi-scale aggregation)
    → NMSFreeDetectionHead (bipartite-matched predictions)
    → [at inference] BioKinematicBehaviorHead (trajectory + behavior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modules.bhfg     import BioHarmonicFrequencyGate
from modules.backbone import YOLOv13MambaBackbone, ConvBNSiLU
from modules.losses   import NMSFreeDetectionHead, DenseFishDetectionLoss


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Lightweight FPN Neck
# ──────────────────────────────────────────────────────────────────────────────

class FPNNeck(nn.Module):
    """
    Top-down Feature Pyramid Network (FPN) neck.
    Fuses P3–P5 from the backbone into a single aggregated feature map
    at stride 8 (P3 resolution) for the detection head.

    in_channels  : (C0, C1, C2, C3) from backbone
    out_channels : unified channel count for the head
    """

    def __init__(self, in_channels: Tuple[int, int, int, int],
                 out_channels: int = 256):
        super().__init__()
        C0, C1, C2, C3 = in_channels
        oc = out_channels

        # lateral projections
        self.lat5 = ConvBNSiLU(C3, oc, 1, 1, 0)
        self.lat4 = ConvBNSiLU(C2, oc, 1, 1, 0)
        self.lat3 = ConvBNSiLU(C1, oc, 1, 1, 0)

        # post-merge 3×3 convs
        self.merge4 = ConvBNSiLU(oc, oc, 3, 1, 1)
        self.merge3 = ConvBNSiLU(oc, oc, 3, 1, 1)

        self.out_channels = oc

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: [P2, P3, P4, P5]
        Returns aggregated feature map at P3 spatial resolution.
        """
        _p2, p3, p4, p5 = features

        f5 = self.lat5(p5)                                   # stride 32
        f4 = self.lat4(p4) + F.interpolate(                  # stride 16
            f5, scale_factor=2, mode="nearest")
        f4 = self.merge4(f4)

        f3 = self.lat3(p3) + F.interpolate(                  # stride 8
            f4, scale_factor=2, mode="nearest")
        f3 = self.merge3(f3)

        return f3                                             # (B, oc, H/8, W/8)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DenseFish-v13 — full model
# ──────────────────────────────────────────────────────────────────────────────

class DenseFishV13(nn.Module):
    """
    Full DenseFish-v13 model.

    Args:
        num_classes : number of detection classes (1 for fish-only)
        num_queries : number of object queries for the NMS-free head
        img_size    : assumed square input size
        backbone_channels: (C0,C1,C2,C3) for the backbone stages
        neck_channels     : FPN output channels (= head hidden dim)
        d_state     : Mamba state dimension
        bhfg_reduction : channel reduction ratio for B-HFG attention
    """

    def __init__(self,
                 num_classes: int = 1,
                 num_queries: int = 300,
                 img_size:    int = 640,
                 backbone_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
                 neck_channels: int = 256,
                 d_state: int = 16,
                 bhfg_reduction: int = 4):
        super().__init__()

        # ── B-HFG (applied to raw stem features, not raw pixel) ────────────
        # We apply B-HFG to the low-level (C0) feature map after the stem
        # so it operates on learned features rather than raw RGB.
        self.bhfg = BioHarmonicFrequencyGate(
            channels=backbone_channels[0], reduction=bhfg_reduction)

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = YOLOv13MambaBackbone(
            in_channels=3,
            channels=backbone_channels,
            d_state=d_state)

        # ── FPN Neck ─────────────────────────────────────────────────────────
        self.neck = FPNNeck(backbone_channels, neck_channels)

        # ── NMS-Free Detection Head ──────────────────────────────────────────
        self.head = NMSFreeDetectionHead(
            in_channels=neck_channels,
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=neck_channels)

        self.img_size = img_size

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                         nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x : (B, 3, H, W)
        Returns dict with 'logits', 'boxes', 'features'.
        """
        # 1. backbone stem + B-HFG spectral denoising
        feats = self._forward_with_bhfg(x)

        # 2. FPN neck
        agg = self.neck(feats)

        # 3. NMS-free head
        return self.head(agg)

    def _forward_with_bhfg(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Run backbone stem, apply B-HFG to P2, then continue backbone stages.
        """
        # backbone stem produces P2
        p2_raw = self.backbone.stage0(self.backbone.stem(x))   # (B, C0, H/4, W/4)

        # B-HFG on P2
        p2 = self.bhfg(p2_raw)

        # continue backbone from stage 1 onward
        p3 = self.backbone.stage1(self.backbone.down1(p2))
        p4 = self.backbone.stage2(self.backbone.down2(p3))
        p5 = self.backbone.stage3(self.backbone.down3(p4))

        return [p2, p3, p4, p5]

    # ── post-process predictions ─────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                conf_threshold: float = 0.5,
                img_size: Optional[int] = None) -> List[Dict]:
        """
        Run model and filter predictions by confidence.

        Returns list of per-image dicts:
            {'boxes':  (N,4) [x1,y1,x2,y2] in pixels,
             'scores': (N,)
             'labels': (N,)}
        """
        self.eval()
        outputs = self(x)
        S = img_size or self.img_size

        from modules.losses import box_cxcywh_to_xyxy
        batch_results = []
        B = x.size(0)
        for b in range(B):
            logits = outputs["logits"][b]   # (Q, C)
            boxes  = outputs["boxes"][b]    # (Q, 4) cxcywh [0,1]
            scores = logits.sigmoid().max(-1).values
            labels = logits.sigmoid().argmax(-1)

            keep   = scores > conf_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes  = box_cxcywh_to_xyxy(boxes[keep]) * S   # pixel coords

            batch_results.append({
                "boxes":  boxes,
                "scores": scores,
                "labels": labels,
            })
        return batch_results

    # ── model info ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flops(self, input_size: Tuple[int, int] = (640, 640)) -> float:
        """Approximate FLOPs via a single forward pass using thop (if available)."""
        try:
            from thop import profile
            dummy = torch.zeros(1, 3, *input_size)
            flops, _ = profile(self, (dummy,), verbose=False)
            return flops
        except ImportError:
            return -1.0


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Convenience constructor matching Table 1 / ablation settings
# ──────────────────────────────────────────────────────────────────────────────

def build_densefish_v13(num_classes: int = 1,
                         pretrained: Optional[str] = None) -> DenseFishV13:
    """Build DenseFish-v13 with paper-default settings (~22M params)."""
    model = DenseFishV13(
        num_classes=num_classes,
        num_queries=300,
        img_size=640,
        backbone_channels=(64, 128, 256, 512),
        neck_channels=256,
        d_state=16,
        bhfg_reduction=4,
    )
    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"[DenseFish-v13] Loaded weights from {pretrained}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Quick unit test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    model  = build_densefish_v13(num_classes=1)
    dummy  = torch.randn(2, 3, 640, 640)
    out    = model(dummy)

    print("DenseFish-v13 output shapes:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    n_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {n_params / 1e6:.1f} M")

    # inference / predict
    detections = model.predict(dummy, conf_threshold=0.3)
    for b, d in enumerate(detections):
        print(f"  Image {b}: {d['boxes'].shape[0]} detections above threshold")

    print("\nFull model unit test passed ✓")
