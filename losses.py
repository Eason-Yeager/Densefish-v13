"""
Asymmetry-Aware Density Repulsion Loss & NMS-Free Detection Head
=================================================================
Section 3.4 of DenseFish-v13.

Key components:
  • HungarianMatcher     — bipartite matching via scipy linear_sum_assignment
  • LatentRepulsionLoss  — Eq. 5–6: IoU-weighted cosine penalty
  • DensityAwareRepulsionLoss — full Eq. 8 repulsion term with CPO prototype bank
  • NMSFreeDetectionHead — regression + classification head feeding the losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Box utilities
# ──────────────────────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2,
                         cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2,
                         x2 - x1, y2 - y1], dim=-1)


def box_iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two sets of boxes.
    boxes_a: (M, 4)  boxes_b: (N, 4)  both in [x1,y1,x2,y2]
    Returns (M, N) IoU matrix.
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CIoU loss for matched box pairs.
    pred, target : (N, 4) in [x1,y1,x2,y2]
    Returns (N,) loss vector.
    """
    pw = pred[:, 2] - pred[:, 0];  ph = pred[:, 3] - pred[:, 1]
    tw = target[:, 2] - target[:, 0]; th = target[:, 3] - target[:, 1]
    pcx = (pred[:, 0] + pred[:, 2]) / 2
    pcy = (pred[:, 1] + pred[:, 3]) / 2
    tcx = (target[:, 0] + target[:, 2]) / 2
    tcy = (target[:, 1] + target[:, 3]) / 2

    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = pw * ph + tw * th - inter + 1e-6
    iou   = inter / union

    # enclosing box diagonal²
    enc_x1 = torch.min(pred[:, 0], target[:, 0])
    enc_y1 = torch.min(pred[:, 1], target[:, 1])
    enc_x2 = torch.max(pred[:, 2], target[:, 2])
    enc_y2 = torch.max(pred[:, 3], target[:, 3])
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-6

    # centre distance²
    d2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # aspect-ratio penalty
    v = (4 / (math.pi ** 2)) * (torch.atan(tw / (th + 1e-6)) -
                                  torch.atan(pw / (ph + 1e-6))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)

    return 1 - iou + d2 / c2 + alpha * v


import math


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Hungarian Matcher  (Eq. 4)
# ──────────────────────────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """
    Bipartite matching between predictions and ground-truth targets.
    Cost = λ_cls * focal_cls_cost + λ_box * l1_cost + λ_iou * giou_cost
    """

    def __init__(self, cost_class: float = 1.0,
                 cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits: torch.Tensor,
                pred_boxes: torch.Tensor,
                gt_labels: List[torch.Tensor],
                gt_boxes:  List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        pred_logits : (B, num_queries, num_classes)
        pred_boxes  : (B, num_queries, 4)  cx cy w h  [normalised 0–1]
        gt_labels   : list of (Ni,) per image
        gt_boxes    : list of (Ni, 4)  [x1,y1,x2,y2 in pixel, needs normalisation]
        Returns: list of (pred_idx, gt_idx) per image.
        """
        B, Q, C = pred_logits.shape
        results = []

        pred_prob = pred_logits.sigmoid()                    # focal-style

        for b in range(B):
            if len(gt_labels[b]) == 0:
                results.append((torch.zeros(0, dtype=torch.long),
                                 torch.zeros(0, dtype=torch.long)))
                continue

            pb = pred_boxes[b]                               # (Q, 4) cxcywh
            pp = pred_prob[b]                                # (Q, C)
            gl = gt_labels[b]                                # (N,)
            gb = gt_boxes[b]                                 # (N, 4)
            N  = len(gl)

            # classification cost  (negative of prob for matching target class)
            cost_cls = -pp[:, gl]                            # (Q, N)

            # L1 cost in cxcywh
            pb_rep = pb.unsqueeze(1).expand(-1, N, -1)       # (Q, N, 4)
            gb_cx  = box_xyxy_to_cxcywh(gb)                  # (N, 4)
            cost_l1 = torch.cdist(pb.float(),
                                   gb_cx.float(), p=1)       # (Q, N)

            # GIoU cost
            pb_xyxy = box_cxcywh_to_xyxy(pb)
            iou_mat = box_iou_matrix(pb_xyxy, gb)            # (Q, N)
            cost_iou = -iou_mat

            C_mat = (self.cost_class * cost_cls +
                     self.cost_bbox  * cost_l1 +
                     self.cost_giou  * cost_iou).cpu().numpy()

            row_idx, col_idx = linear_sum_assignment(C_mat)
            results.append((torch.as_tensor(row_idx, dtype=torch.long),
                             torch.as_tensor(col_idx, dtype=torch.long)))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Latent Repulsion Loss  (Eq. 5–6)
# ──────────────────────────────────────────────────────────────────────────────

class LatentRepulsionLoss(nn.Module):
    """
    Penalises cosine-similar latent features for spatially overlapping boxes.

    Eq. 5:  w_ij = 1{IoU(Bi,Bj) > τ} · IoU(Bi,Bj)
    Eq. 6:  L_rep = (1/N_pair) Σ_{i≠j} w_ij · (fi · fj) / (|fi||fj|)

    Args:
        tau   : IoU threshold τ
        weight: overall loss scale λ_rep
    """

    def __init__(self, tau: float = 0.5, weight: float = 0.2):
        super().__init__()
        self.tau    = tau
        self.weight = weight

    def forward(self, features: torch.Tensor,
                boxes: torch.Tensor) -> torch.Tensor:
        """
        features : (N, D)  latent feature vectors for N predicted instances
        boxes    : (N, 4)  [x1,y1,x2,y2]
        Returns scalar repulsion loss.
        """
        N = features.size(0)
        if N < 2:
            return features.new_zeros(1).squeeze()

        # pairwise IoU
        iou = box_iou_matrix(boxes, boxes)                   # (N, N)
        mask = (iou > self.tau).float()
        torch.diagonal(mask).fill_(0)                        # exclude self

        n_pairs = mask.sum().clamp(min=1)

        # cosine similarity matrix
        f_norm = F.normalize(features, dim=-1)               # (N, D)
        cos_sim = f_norm @ f_norm.T                          # (N, N)

        loss = (mask * iou * cos_sim).sum() / n_pairs
        return self.weight * loss.clamp(min=0)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Contrastive Prototype Orthogonalisation (CPO)
# ──────────────────────────────────────────────────────────────────────────────

class CPOPrototypeBank(nn.Module):
    """
    Contrastive Prototype Orthogonalisation (Section 3.4).

    Maintains a dynamic bank of K biological prototypes.
    During training, enforces:
      (a) maximise MI with assigned prototype,
      (b) Gram–Schmidt orthogonality against overlapping neighbours.

    Args:
        dim         : feature dimension D
        n_prototypes: K – number of orientation/scale prototypes
        momentum    : EMA update rate for prototype update
    """

    def __init__(self, dim: int, n_prototypes: int = 16, momentum: float = 0.99):
        super().__init__()
        self.dim         = dim
        self.n_prototypes = n_prototypes
        self.momentum    = momentum
        # prototype bank (not a parameter — updated by EMA)
        self.register_buffer('prototypes',
                             F.normalize(torch.randn(n_prototypes, dim), dim=-1))

    @torch.no_grad()
    def _assign(self, features: torch.Tensor) -> torch.Tensor:
        """Assign each feature to its nearest prototype. Returns (N,) indices."""
        sims = features @ self.prototypes.T                  # (N, K)
        return sims.argmax(dim=-1)

    @torch.no_grad()
    def _update(self, features: torch.Tensor, assignments: torch.Tensor):
        """EMA update of prototypes with assigned features."""
        for k in range(self.n_prototypes):
            mask = (assignments == k)
            if mask.sum() == 0:
                continue
            mean_feat = F.normalize(features[mask].mean(0), dim=-1)
            self.prototypes[k] = (self.momentum * self.prototypes[k] +
                                  (1 - self.momentum) * mean_feat)
        self.prototypes = F.normalize(self.prototypes, dim=-1)

    def forward(self, features: torch.Tensor,
                overlap_mask: torch.Tensor) -> torch.Tensor:
        """
        features     : (N, D)
        overlap_mask : (N, N) bool – which pairs overlap
        Returns scalar CPO loss.
        """
        N, D = features.shape
        if N == 0:
            return features.new_zeros(1).squeeze()

        feat_norm = F.normalize(features.detach(), dim=-1)
        assign    = self._assign(feat_norm)
        if self.training:
            self._update(feat_norm, assign)

        # (a) Mutual-information proxy: maximise cos-sim to assigned prototype
        proto_assign = self.prototypes[assign]               # (N, D)
        mi_loss = 1 - (F.normalize(features, dim=-1) * proto_assign).sum(-1).mean()

        # (b) Gram–Schmidt orthogonality among overlapping pairs
        orth_loss = features.new_zeros(1).squeeze()
        pairs_found = 0
        for i in range(N):
            neighbours = overlap_mask[i].nonzero(as_tuple=False).squeeze(-1)
            for j in neighbours:
                if j <= i:
                    continue
                fi = F.normalize(features[i], dim=-1)
                fj = F.normalize(features[j], dim=-1)
                orth_loss = orth_loss + fi.dot(fj).abs()
                pairs_found += 1
        if pairs_found:
            orth_loss = orth_loss / pairs_found

        return mi_loss + orth_loss


# ──────────────────────────────────────────────────────────────────────────────
# 5.  NMS-Free Detection Head
# ──────────────────────────────────────────────────────────────────────────────

class NMSFreeDetectionHead(nn.Module):
    """
    Prediction head that outputs (logits, boxes, latent_features) for
    each of `num_queries` learnable object queries.

    The head is intentionally lightweight — feature pyramid aggregation
    (FPN/PAN neck) is assumed to be handled externally.

    Args:
        in_channels  : channel dimension of the aggregated feature map
        num_classes  : number of object classes (1 for fish-only)
        num_queries  : number of learnable detection queries
        hidden_dim   : internal projection dimension
    """

    def __init__(self, in_channels: int, num_classes: int = 1,
                 num_queries: int = 300, hidden_dim: int = 256):
        super().__init__()
        self.num_queries = num_queries

        # learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # feature projection
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)

        # cross-attention between queries and features
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8,
                                                 dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # output heads
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),                # predict normalised cxcywh ∈ [0,1]
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_embed.weight, std=0.01)
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.constant_(self.cls_head.bias, 0)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        feat : (B, C, H, W)  — aggregated FPN feature at the smallest stride
        Returns dict:
            'logits'   (B, Q, num_classes)
            'boxes'    (B, Q, 4)  cxcywh normalised
            'features' (B, Q, hidden_dim)  latent representations
        """
        B, C, H, W = feat.shape
        feat_proj = self.input_proj(feat)                    # (B, hid, H, W)
        feat_flat = feat_proj.flatten(2).permute(0, 2, 1)   # (B, H*W, hid)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, hid)

        # cross-attention: queries attend to spatial features
        q_out, _ = self.cross_attn(queries, feat_flat, feat_flat)
        q_out = self.norm1(queries + q_out)
        q_out = self.norm2(q_out + self.ffn(q_out))         # (B, Q, hid)

        logits = self.cls_head(q_out)                        # (B, Q, num_classes)
        boxes  = self.box_head(q_out)                        # (B, Q, 4)

        return {"logits": logits, "boxes": boxes, "features": q_out}


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Composite Detection Loss  (Eq. 8)
# ──────────────────────────────────────────────────────────────────────────────

class DenseFishDetectionLoss(nn.Module):
    """
    ℒ_total = λ_box·ℒ_CIoU + λ_cls·ℒ_BCE + λ_dfl·ℒ_DFL + λ_rep·ℒ_rep

    Args (loss weights):
        lam_box, lam_cls, lam_dfl, lam_rep
    """

    def __init__(self,
                 lam_box: float = 5.0,
                 lam_cls: float = 1.0,
                 lam_dfl: float = 1.5,
                 lam_rep: float = 0.2,
                 tau: float = 0.5,
                 num_classes: int = 1,
                 img_size: int = 640,
                 repulsion_start_epoch: int = 50):
        super().__init__()
        self.lam_box = lam_box
        self.lam_cls = lam_cls
        self.lam_dfl = lam_dfl
        self.lam_rep = lam_rep
        self.img_size = img_size
        self.rep_start = repulsion_start_epoch
        self._current_epoch = 0

        self.matcher  = HungarianMatcher()
        self.rep_loss = LatentRepulsionLoss(tau=tau, weight=lam_rep)
        self.cpo      = CPOPrototypeBank(dim=256, n_prototypes=16)

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: List[dict]) -> Dict[str, torch.Tensor]:
        """
        outputs : dict from NMSFreeDetectionHead
        targets : list of dicts with 'boxes' (N,4) xyxy pixel, 'labels' (N,)
        """
        pred_logits = outputs["logits"]   # (B, Q, C)
        pred_boxes  = outputs["boxes"]    # (B, Q, 4) cxcywh [0,1]
        pred_feats  = outputs["features"] # (B, Q, D)

        B = pred_logits.size(0)
        S = self.img_size

        gt_labels = [t["labels"] for t in targets]
        gt_boxes  = [t["boxes"]  for t in targets]   # pixel xyxy

        # normalise gt boxes to [0,1]
        gt_boxes_norm = [b / S for b in gt_boxes]

        indices = self.matcher(pred_logits, pred_boxes, gt_labels, gt_boxes_norm)

        loss_cls  = pred_logits.new_zeros(1).squeeze()
        loss_box  = pred_logits.new_zeros(1).squeeze()
        loss_rep  = pred_logits.new_zeros(1).squeeze()
        n_matched = 0

        for b in range(B):
            pred_idx, gt_idx = indices[b]
            if len(pred_idx) == 0:
                continue
            n_matched += len(pred_idx)

            # matched predictions & targets
            matched_logits = pred_logits[b][pred_idx]        # (M, C)
            matched_boxes  = pred_boxes[b][pred_idx]         # (M, 4) cxcywh [0,1]
            matched_feats  = pred_feats[b][pred_idx]         # (M, D)
            tgt_labels     = gt_labels[b][gt_idx]            # (M,)
            tgt_boxes_norm = gt_boxes_norm[b][gt_idx]        # (M, 4) xyxy [0,1]
            tgt_boxes_cx   = box_xyxy_to_cxcywh(tgt_boxes_norm)  # (M,4) cxcywh

            # ── classification (BCE) ────────────────────────────────────────
            tgt_onehot = F.one_hot(tgt_labels,
                                    num_classes=matched_logits.size(-1)).float()
            loss_cls = loss_cls + F.binary_cross_entropy_with_logits(
                matched_logits, tgt_onehot, reduction="sum")

            # ── box regression (CIoU) ────────────────────────────────────────
            p_xyxy = box_cxcywh_to_xyxy(matched_boxes)
            t_xyxy = box_cxcywh_to_xyxy(tgt_boxes_cx)
            loss_box = loss_box + ciou_loss(p_xyxy, t_xyxy).sum()

            # ── repulsion (only after warm-up) ───────────────────────────────
            if self._current_epoch >= self.rep_start:
                p_boxes_px = box_cxcywh_to_xyxy(matched_boxes) * S
                loss_rep = loss_rep + self.rep_loss(matched_feats, p_boxes_px)

        norm = max(n_matched, 1)
        loss_cls  = loss_cls  / norm
        loss_box  = loss_box  / norm
        loss_dfl  = pred_logits.new_zeros(1).squeeze()      # DFL placeholder

        total = (self.lam_cls * loss_cls +
                 self.lam_box * loss_box +
                 self.lam_dfl * loss_dfl +
                 loss_rep)                                   # already weighted

        return {
            "loss_total": total,
            "loss_cls":   loss_cls.detach(),
            "loss_box":   loss_box.detach(),
            "loss_dfl":   loss_dfl.detach(),
            "loss_rep":   loss_rep.detach(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Quick unit test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    # Synthetic inputs
    B, Q, D, C = 2, 300, 256, 1
    pred = {
        "logits":   torch.randn(B, Q, C),
        "boxes":    torch.sigmoid(torch.randn(B, Q, 4)),
        "features": torch.randn(B, Q, D),
    }
    targets = [
        {"labels": torch.zeros(8, dtype=torch.long),
         "boxes":  torch.rand(8, 4).clamp(0, 640) * 640},
        {"labels": torch.zeros(12, dtype=torch.long),
         "boxes":  torch.rand(12, 4).clamp(0, 640) * 640},
    ]
    # fix boxes to be proper x1y1x2y2
    for t in targets:
        b = t["boxes"]
        b[:, 2:] = b[:, :2] + b[:, 2:].abs() + 10

    criterion = DenseFishDetectionLoss()
    criterion.set_epoch(55)                                  # after repulsion start
    losses = criterion(pred, targets)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    print("Detection loss unit test passed ✓")
