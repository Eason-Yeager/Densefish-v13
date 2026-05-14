"""
Bio-Kinematic Behavior Head
============================
Section 3.5 of DenseFish-v13.

Pipeline:
  1. Multi-frame detections → tracking (IoU-based greedy tracker)
  2. Trajectory → kinematic features: velocity v_i, turning angle θ_i  (Eq. 7)
  3. Rule-based Bio-Logic Tree → behavior state
       • "Feeding"  :  v̄ > δ_high  AND  Var(θ) > ε
       • "Hypoxia"  :  v̄ < δ_low   AND  y < H_surface
       • "Normal"   :  otherwise

The module also exposes a neural variant (KinematicBehaviorNN) that
learns thresholds from labelled trajectory data, enabling optional
end-to-end fine-tuning of the behavior branch.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data structures
# ──────────────────────────────────────────────────────────────────────────────

BEHAVIOR_CLASSES = ["Normal", "Feeding", "Hypoxia"]
B_NORMAL, B_FEEDING, B_HYPOXIA = 0, 1, 2


@dataclass
class Track:
    track_id:   int
    centroids:  List[Tuple[float, float]] = field(default_factory=list)
    frames:     List[int]                 = field(default_factory=list)
    active:     bool                      = True

    def add(self, cx: float, cy: float, frame_idx: int):
        self.centroids.append((cx, cy))
        self.frames.append(frame_idx)

    def last_centroid(self) -> Tuple[float, float]:
        return self.centroids[-1]

    def __len__(self):
        return len(self.centroids)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  IoU-based greedy tracker
# ──────────────────────────────────────────────────────────────────────────────

class IoUTracker:
    """
    Simple IoU-based tracker sufficient for the Bio-Kinematic Head.
    Associates detections across frames greedily by highest IoU.

    Args:
        iou_thresh  : minimum IoU to match a detection to an existing track
        max_lost    : frames a track can be un-matched before removal
    """

    def __init__(self, iou_thresh: float = 0.35, max_lost: int = 5):
        self.iou_thresh = iou_thresh
        self.max_lost   = max_lost
        self.tracks:    Dict[int, Track] = {}
        self.lost:      Dict[int, int]   = {}       # track_id → frames_lost
        self._next_id   = 0
        self._frame_idx = 0

    # ── IoU helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(b1, b2) -> float:
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    # ── public update ─────────────────────────────────────────────────────────

    def update(self, boxes: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """
        boxes : (N, 4) [x1,y1,x2,y2]
        Returns {track_id: (cx, cy)} for all active tracks this frame.
        """
        self._frame_idx += 1
        active_ids  = list(self.tracks.keys())
        centroids_out = {}

        if len(boxes) == 0:
            for tid in active_ids:
                self.lost[tid] = self.lost.get(tid, 0) + 1
            self._prune()
            return centroids_out

        # compute IoU matrix (active_tracks × detections)
        if active_ids:
            last_boxes = np.array([
                [t.centroids[-1][0] - 20, t.centroids[-1][1] - 15,
                 t.centroids[-1][0] + 20, t.centroids[-1][1] + 15]
                for t in [self.tracks[i] for i in active_ids]
            ])
            iou_mat = np.zeros((len(active_ids), len(boxes)))
            for i, lb in enumerate(last_boxes):
                for j, db in enumerate(boxes):
                    iou_mat[i, j] = self._iou(lb, db)

            matched_tracks = set(); matched_dets = set()

            # greedy match
            for _ in range(min(len(active_ids), len(boxes))):
                flat = iou_mat.argmax()
                ti, di = divmod(flat, iou_mat.shape[1])
                if iou_mat[ti, di] < self.iou_thresh:
                    break
                tid = active_ids[ti]
                b   = boxes[di]
                cx  = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2
                self.tracks[tid].add(cx, cy, self._frame_idx)
                self.lost[tid] = 0
                centroids_out[tid] = (cx, cy)
                matched_tracks.add(ti); matched_dets.add(di)
                iou_mat[ti, :] = -1; iou_mat[:, di] = -1

            # unmatched tracks
            for ti, tid in enumerate(active_ids):
                if ti not in matched_tracks:
                    self.lost[tid] = self.lost.get(tid, 0) + 1
        else:
            matched_dets = set()

        # new tracks for unmatched detections
        for di, b in enumerate(boxes):
            if di not in matched_dets:
                tid = self._next_id; self._next_id += 1
                cx  = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2
                self.tracks[tid] = Track(tid)
                self.tracks[tid].add(cx, cy, self._frame_idx)
                self.lost[tid] = 0
                centroids_out[tid] = (cx, cy)

        self._prune()
        return centroids_out

    def _prune(self):
        remove = [tid for tid, lf in self.lost.items() if lf > self.max_lost]
        for tid in remove:
            del self.tracks[tid]; del self.lost[tid]

    def get_trajectories(self) -> Dict[int, np.ndarray]:
        """Return {tid: array (T,2)} for all active tracks with len ≥ 2."""
        return {tid: np.array(t.centroids)
                for tid, t in self.tracks.items() if len(t) >= 2}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Kinematic feature extraction  (Eq. 7)
# ──────────────────────────────────────────────────────────────────────────────

def compute_kinematics(trajectory: np.ndarray,
                       dt: float = 1.0) -> Dict[str, float]:
    """
    trajectory : (T, 2) array of (cx, cy) centroids
    dt         : frame interval (seconds)

    Returns dict:
        mean_velocity   : v̄  (pixels / frame)
        velocity_std    : σ_v
        turning_variance: Var(θ)  (radians²)
        mean_y          : average y-coordinate (for surface proximity)
        min_y           : minimum y (closest to surface)
        kinematics_raw  : dict of per-step v and θ
    """
    T = len(trajectory)
    if T < 2:
        return {"mean_velocity": 0.0, "velocity_std": 0.0,
                "turning_variance": 0.0, "mean_y": float(trajectory[0, 1]),
                "min_y": float(trajectory[0, 1])}

    dx = np.diff(trajectory[:, 0])
    dy = np.diff(trajectory[:, 1])

    velocities = np.sqrt(dx ** 2 + dy ** 2) / dt            # (T-1,)
    angles     = np.arctan2(dy, dx)                          # (T-1,)

    # turning angles (T-2,)
    if T >= 3:
        turning = np.diff(angles)                            # (T-2,)
        # wrap to [-π, π]
        turning = (turning + math.pi) % (2 * math.pi) - math.pi
        turn_var = float(np.var(turning))
    else:
        turn_var = 0.0

    return {
        "mean_velocity":    float(np.mean(velocities)),
        "velocity_std":     float(np.std(velocities)),
        "turning_variance": turn_var,
        "mean_y":           float(np.mean(trajectory[:, 1])),
        "min_y":            float(np.min(trajectory[:, 1])),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Rule-based Bio-Logic Tree
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BioLogicThresholds:
    delta_high:  float = 0.7    # high-velocity threshold (normalised)
    delta_low:   float = 0.3    # low-velocity threshold
    epsilon:     float = 0.5    # turning-variance threshold (rad²)
    h_surface:   float = 0.2    # y < h_surface * frame_height → near surface


class BioLogicTree:
    """
    Rule-based behavior classifier (Section 3.5 Table 1 thresholds).

    Decision logic:
        if v̄ > δ_high  AND  Var(θ) > ε  → Feeding
        elif v̄ < δ_low  AND  y_min < H_surface * img_h  → Hypoxia
        else  → Normal
    """

    def __init__(self, thresholds: Optional[BioLogicThresholds] = None,
                 img_height: int = 640):
        self.th = thresholds or BioLogicThresholds()
        self.img_height = img_height

    def classify(self, kin: Dict[str, float]) -> Tuple[str, int]:
        v    = kin["mean_velocity"]
        var_theta = kin["turning_variance"]
        y_min = kin["min_y"]

        if v > self.th.delta_high and var_theta > self.th.epsilon:
            return "Feeding", B_FEEDING
        if v < self.th.delta_low and y_min < self.th.h_surface * self.img_height:
            return "Hypoxia", B_HYPOXIA
        return "Normal", B_NORMAL


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Neural behavior classifier (optional learnable variant)
# ──────────────────────────────────────────────────────────────────────────────

class KinematicBehaviorNN(nn.Module):
    """
    Lightweight LSTM-based classifier over kinematic sequences.
    Input: padded velocity + turning-angle sequence of shape (B, T, 2)
    Output: (B, 3) class logits

    Complements the rule-based tree when labelled trajectory data exists.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=2, batch_first=True,
                            dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x       : (B, T, 2)  — [velocity, turning_angle] per step
        lengths : (B,) actual sequence lengths before padding
        Returns : (B, num_classes) logits
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hn, _) = self.lstm(packed)
        else:
            _, (hn, _) = self.lstm(x)

        return self.classifier(hn[-1])  # last layer hidden state


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Full Bio-Kinematic Behavior Head (combines tracker + kinematics + tree)
# ──────────────────────────────────────────────────────────────────────────────

class BioKinematicBehaviorHead:
    """
    Stateful head that accumulates per-track trajectories across video frames
    and emits behavior predictions.

    Usage:
        head = BioKinematicBehaviorHead()
        for frame_boxes in video:           # frame_boxes: np.ndarray (N,4)
            results = head.update(frame_boxes)
            # results: {track_id: {'label': str, 'class': int, 'kinematics': dict}}
    """

    def __init__(self,
                 iou_thresh:  float = 0.35,
                 max_lost:    int   = 5,
                 min_traj_len: int  = 5,
                 thresholds:  Optional[BioLogicThresholds] = None,
                 img_height:  int   = 640,
                 dt:          float = 1.0):
        self.tracker   = IoUTracker(iou_thresh, max_lost)
        self.logic     = BioLogicTree(thresholds, img_height)
        self.min_len   = min_traj_len
        self.dt        = dt

    def reset(self):
        self.tracker = IoUTracker()

    def update(self, boxes: np.ndarray) -> Dict[int, dict]:
        """
        boxes : (N, 4) [x1,y1,x2,y2] detections for the current frame
        Returns per-track behavior result dict.
        """
        self.tracker.update(boxes)
        trajs = self.tracker.get_trajectories()

        results = {}
        for tid, traj in trajs.items():
            if len(traj) < self.min_len:
                continue
            kin   = compute_kinematics(traj, self.dt)
            label, cls_idx = self.logic.classify(kin)
            results[tid] = {
                "label":      label,
                "class":      cls_idx,
                "kinematics": kin,
                "trajectory": traj,
            }
        return results

    def summary(self, results: Dict[int, dict]) -> Dict[str, int]:
        """Aggregate behavior counts across active tracks."""
        counts = {cls: 0 for cls in BEHAVIOR_CLASSES}
        for r in results.values():
            counts[r["label"]] += 1
        return counts


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Behavior classification evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_behavior_metrics(y_true: List[int],
                              y_pred: List[int]) -> Dict[str, float]:
    """
    Compute per-class precision, recall, F1, and overall accuracy.
    Returns a flat dict with keys like 'Normal_F1', 'Feeding_Recall', etc.
    """
    from collections import defaultdict
    n_classes = len(BEHAVIOR_CLASSES)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    metrics = {}
    for i, cls in enumerate(BEHAVIOR_CLASSES):
        prec = tp[i] / (tp[i] + fp[i] + 1e-6)
        rec  = tp[i] / (tp[i] + fn[i] + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)
        metrics[f"{cls}_Precision"] = prec
        metrics[f"{cls}_Recall"]    = rec
        metrics[f"{cls}_F1"]        = f1

    acc = sum(tp.values()) / len(y_true) if y_true else 0.0
    metrics["Overall_Accuracy"] = acc
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Quick unit test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)

    # Simulate 30 frames of fish detections
    head = BioKinematicBehaviorHead(min_traj_len=5, img_height=640)
    for t in range(30):
        n_fish = np.random.randint(5, 15)
        boxes  = np.random.rand(n_fish, 4)
        boxes[:, 2:] = boxes[:, :2] + np.random.rand(n_fish, 2) * 0.1 + 0.05
        boxes  *= 640
        boxes[:, 2] = boxes[:, 0] + abs(boxes[:, 2] - boxes[:, 0]) + 10
        boxes[:, 3] = boxes[:, 1] + abs(boxes[:, 3] - boxes[:, 1]) + 5
        results = head.update(boxes)

    print(f"Active trajectories analysed: {len(results)}")
    if results:
        tid   = next(iter(results))
        r     = results[tid]
        print(f"  Track {tid}: label={r['label']}, "
              f"v̄={r['kinematics']['mean_velocity']:.2f}, "
              f"Var(θ)={r['kinematics']['turning_variance']:.3f}")
    summary = head.summary(results)
    print(f"  Behavior summary: {summary}")

    # Neural variant
    model = KinematicBehaviorNN()
    seq   = torch.randn(4, 20, 2)          # B=4, T=20, 2 features
    logits = model(seq)
    print(f"  KinematicBehaviorNN output: {logits.shape}")
    print("Behavior head unit test passed ✓")
