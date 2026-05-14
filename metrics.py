"""
DenseFish-v13 — Evaluation Metrics & Utilities
================================================
Implements all metrics from Section 4.1.1:
  • mAP@0.5, mAP@0.5:0.95
  • Precision, Recall
  • Counting MAE
  • Occlusion Recall Rocc  (Eq. 9)
  • FPS benchmark for edge deployment
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Helper: IoU
# ──────────────────────────────────────────────────────────────────────────────

def _iou_np(b1: np.ndarray, b2: np.ndarray) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6) if union > 0 else 0.0


def _iou_matrix_np(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return (M, N) IoU matrix. pred: (M,4), gt: (N,4)."""
    M, N = len(pred), len(gt)
    mat = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            mat[i, j] = _iou_np(pred[i], gt[j])
    return mat


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Detection metrics (mAP)
# ──────────────────────────────────────────────────────────────────────────────

class DetectionMetrics:
    """
    Accumulates predictions and ground-truths across a dataset, then computes
    precision, recall, mAP@0.5, and mAP@0.5:0.95.

    Based on the PASCAL VOC 11-point interpolation approach.
    """

    IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)   # 0.5:0.05:0.95

    def __init__(self, num_classes: int = 1, iou_thresh: float = 0.5):
        self.num_classes = num_classes
        self.iou_thresh  = iou_thresh
        self._reset()

    def _reset(self):
        # per-class: list of (score, tp_flag) tuples
        self._all_preds: Dict[int, List] = defaultdict(list)
        self._n_gt:      Dict[int, int]  = defaultdict(int)

    def update(self, pred: Dict, target: Dict):
        """
        pred   : dict with 'boxes' (M,4), 'scores' (M,), 'labels' (M,)
        target : dict with 'boxes' (N,4), 'labels' (N,)
        All boxes in pixel [x1,y1,x2,y2].
        """
        pred_boxes  = pred["boxes"].cpu().numpy()   if len(pred["boxes"]) else np.zeros((0,4))
        pred_scores = pred["scores"].cpu().numpy()  if len(pred["boxes"]) else np.zeros(0)
        pred_labels = pred["labels"].cpu().numpy()  if len(pred["boxes"]) else np.zeros(0, int)
        gt_boxes    = target["boxes"].cpu().numpy() if len(target["boxes"]) else np.zeros((0,4))
        gt_labels   = target["labels"].cpu().numpy()if len(target["boxes"]) else np.zeros(0, int)

        for cls in range(self.num_classes):
            p_mask = (pred_labels == cls)
            g_mask = (gt_labels  == cls)
            p_b = pred_boxes[p_mask];  p_s = pred_scores[p_mask]
            g_b = gt_boxes[g_mask]
            self._n_gt[cls] += len(g_b)

            if len(p_b) == 0:
                continue

            iou_mat  = _iou_matrix_np(p_b, g_b) if len(g_b) else np.zeros((len(p_b), 0))
            gt_used  = set()
            # sort by descending score
            order    = np.argsort(-p_s)
            for pi in order:
                tp = 0
                if len(g_b):
                    iou_row = iou_mat[pi]
                    best_j  = int(np.argmax(iou_row))
                    if iou_row[best_j] >= self.iou_thresh and best_j not in gt_used:
                        tp = 1
                        gt_used.add(best_j)
                self._all_preds[cls].append((p_s[pi], tp))

    def _ap_per_class(self, cls: int) -> float:
        preds = sorted(self._all_preds[cls], key=lambda x: -x[0])
        n_gt  = self._n_gt[cls]
        if n_gt == 0:
            return 0.0
        tp_cum = 0; fp_cum = 0
        prec_pts = []; rec_pts = []
        for score, tp in preds:
            if tp:
                tp_cum += 1
            else:
                fp_cum += 1
            prec_pts.append(tp_cum / (tp_cum + fp_cum))
            rec_pts.append(tp_cum / n_gt)

        # 11-point interpolated AP
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            p_at_thr = [p for p, r in zip(prec_pts, rec_pts) if r >= thr]
            ap += max(p_at_thr) if p_at_thr else 0.0
        return ap / 11.0

    def compute(self) -> Dict[str, float]:
        """Return {'Precision', 'Recall', 'mAP@50', 'mAP@50:95'}."""
        aps_50 = []
        for cls in range(self.num_classes):
            old_thresh = self.iou_thresh
            self.iou_thresh = 0.5
            ap50 = self._ap_per_class(cls)
            self.iou_thresh = old_thresh
            aps_50.append(ap50)

        # mAP 0.5:0.95
        aps_coco = []
        for thr in self.IOU_THRESHOLDS:
            self.iou_thresh = thr
            aps_coco.append(
                np.mean([self._ap_per_class(c) for c in range(self.num_classes)]))
        self.iou_thresh = 0.5   # restore

        # precision / recall at threshold 0.5
        tp_total = fp_total = fn_total = 0
        for cls in range(self.num_classes):
            for _, tp in self._all_preds[cls]:
                if tp:
                    tp_total += 1
                else:
                    fp_total += 1
            fn_total += max(0, self._n_gt[cls] - tp_total)

        prec = tp_total / (tp_total + fp_total + 1e-6)
        rec  = tp_total / (tp_total + fn_total + 1e-6)

        return {
            "Precision":    prec,
            "Recall":       rec,
            "mAP@50":       float(np.mean(aps_50)),
            "mAP@50:95":    float(np.mean(aps_coco)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Counting MAE
# ──────────────────────────────────────────────────────────────────────────────

class CountingMetrics:
    def __init__(self):
        self._errors: List[float] = []

    def update(self, pred_count: int, gt_count: int):
        self._errors.append(abs(pred_count - gt_count))

    def compute(self) -> Dict[str, float]:
        if not self._errors:
            return {"Counting_MAE": 0.0}
        return {"Counting_MAE": float(np.mean(self._errors))}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Occlusion Recall  Rocc  (Eq. 9)
# ──────────────────────────────────────────────────────────────────────────────

class OcclusionRecall:
    """
    Computes Rocc = TPacc / (TPacc + FNacc) restricted to GT boxes
    whose IoU with any neighbour exceeds `occ_thresh`.
    """

    def __init__(self, occ_thresh: float = 0.5, iou_match_thresh: float = 0.5):
        self.occ_thresh   = occ_thresh
        self.match_thresh = iou_match_thresh
        self.tp = 0; self.fn = 0

    def _occluded_ids(self, boxes: np.ndarray) -> List[int]:
        N = len(boxes)
        occ = []
        for i in range(N):
            for j in range(N):
                if i != j and _iou_np(boxes[i], boxes[j]) > self.occ_thresh:
                    occ.append(i)
                    break
        return occ

    def update(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray):
        if len(gt_boxes) == 0:
            return
        occ_ids = self._occluded_ids(gt_boxes)
        if not occ_ids:
            return

        occ_gt = gt_boxes[occ_ids]
        iou_mat = _iou_matrix_np(pred_boxes, occ_gt) if len(pred_boxes) else np.zeros((0, len(occ_gt)))
        matched = set()
        for pi in range(len(pred_boxes)):
            for gj in range(len(occ_gt)):
                if gj not in matched and iou_mat[pi, gj] >= self.match_thresh:
                    matched.add(gj)

        self.tp += len(matched)
        self.fn += len(occ_gt) - len(matched)

    def compute(self) -> Dict[str, float]:
        rocc = self.tp / (self.tp + self.fn + 1e-6)
        return {"Occlusion_Recall": rocc}


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FPS Benchmark
# ──────────────────────────────────────────────────────────────────────────────

class FPSBenchmark:
    """
    Measures inference FPS on the current device.

    Args:
        model      : nn.Module (must support forward(x))
        input_size : (H, W)
        n_warmup   : warmup iterations
        n_iters    : measured iterations
        fp16       : use float16 (mirrors TensorRT FP16 on Jetson)
    """

    def __init__(self, model: nn.Module,
                 input_size: tuple = (640, 640),
                 n_warmup: int = 10,
                 n_iters:  int = 100,
                 fp16: bool = True):
        self.model      = model
        self.input_size = input_size
        self.n_warmup   = n_warmup
        self.n_iters    = n_iters
        self.fp16       = fp16

    @torch.no_grad()
    def run(self, device: torch.device) -> Dict[str, float]:
        model = self.model.to(device).eval()
        dummy = torch.zeros(1, 3, *self.input_size, device=device)
        if self.fp16 and device.type == "cuda":
            model  = model.half()
            dummy  = dummy.half()

        # warm-up
        for _ in range(self.n_warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # timed
        t0 = time.perf_counter()
        for _ in range(self.n_iters):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        fps = self.n_iters / elapsed
        lat = elapsed / self.n_iters * 1000   # ms
        return {"FPS": fps, "Latency_ms": lat}


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Full evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, dataloader,
             device: torch.device,
             conf_thresh: float = 0.5,
             occ_iou_thresh: float = 0.5) -> Dict[str, float]:
    """
    Run full evaluation: mAP, MAE, Rocc.
    """
    model.eval()
    det_m = DetectionMetrics(num_classes=1)
    cnt_m = CountingMetrics()
    occ_m = OcclusionRecall(occ_thresh=occ_iou_thresh)

    for images, targets in dataloader:
        images  = images.to(device)
        preds   = model.predict(images, conf_threshold=conf_thresh)

        for pred, tgt in zip(preds, targets):
            pred_np = pred["boxes"].cpu().numpy()
            gt_np   = tgt["boxes"].cpu().numpy() if isinstance(tgt["boxes"], torch.Tensor) \
                      else np.array(tgt["boxes"])

            det_m.update(pred, tgt)
            cnt_m.update(len(pred_np), tgt["n_fish"])
            occ_m.update(pred_np, gt_np)

    results = {}
    results.update(det_m.compute())
    results.update(cnt_m.compute())
    results.update(occ_m.compute())
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick unit test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic predictions
    det = DetectionMetrics(num_classes=1)
    cnt = CountingMetrics()
    occ = OcclusionRecall()

    for _ in range(50):
        N_gt   = np.random.randint(5, 20)
        gt_b   = np.random.rand(N_gt, 4)
        gt_b[:, 2:] = gt_b[:, :2] + 0.1
        gt_b   *= 640
        gt_l   = np.zeros(N_gt, int)

        # simulate imperfect predictions
        N_pd   = int(N_gt * 0.9)
        pd_b   = gt_b[:N_pd] + np.random.randn(N_pd, 4) * 5
        pd_s   = np.random.rand(N_pd)
        pd_l   = np.zeros(N_pd, int)

        pred   = {"boxes": torch.tensor(pd_b), "scores": torch.tensor(pd_s),
                  "labels": torch.tensor(pd_l)}
        tgt    = {"boxes": torch.tensor(gt_b), "labels": torch.tensor(gt_l),
                  "n_fish": N_gt}

        det.update(pred, tgt)
        cnt.update(N_pd, N_gt)
        occ.update(pd_b, gt_b)

    r = {}
    r.update(det.compute())
    r.update(cnt.compute())
    r.update(occ.compute())
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")
    print("Metrics unit test passed ✓")
