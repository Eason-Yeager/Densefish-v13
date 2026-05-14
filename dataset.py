"""
DenseFish-v13 — Dataset Construction
=====================================
Handles three datasets described in the paper:
  1. Dense-Aqua         (proprietary, simulated here via data generator)
  2. Pond Fish Detection Dataset   [ref 35]
  3. Healthy and Loser Salmon Dataset  [ref 36]

Each dataset is wrapped in a unified FishDataset class that returns
  (image, boxes, labels, density_level, occlusion_mask)
so that all training, ablation, and evaluation pipelines share one interface.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Density-level annotation helper
# ──────────────────────────────────────────────────────────────────────────────

DENSITY_THRESHOLDS = {
    "low":     (0,  15),
    "medium":  (15, 30),
    "high":    (30, 50),
    "extreme": (50, 9999),
}


def assign_density_level(n_fish: int) -> str:
    for level, (lo, hi) in DENSITY_THRESHOLDS.items():
        if lo <= n_fish < hi:
            return level
    return "extreme"


def compute_occlusion_mask(boxes: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """
    Return a boolean mask of length N indicating which GT boxes are
    'occluded' (IoU with any neighbour > iou_thresh).
    boxes: (N, 4) in [x1, y1, x2, y2] format.
    """
    N = len(boxes)
    mask = np.zeros(N, dtype=bool)
    if N < 2:
        return mask
    for i in range(N):
        for j in range(i + 1, N):
            iou = _box_iou(boxes[i], boxes[j])
            if iou > iou_thresh:
                mask[i] = True
                mask[j] = True
    return mask


def _box_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Dense-Aqua synthetic generator
#     (used when no real Dense-Aqua images are available)
# ──────────────────────────────────────────────────────────────────────────────

class DenseAquaGenerator:
    """
    Synthesises 640×640 underwater-like images with procedural fish sprites,
    bubble noise, turbidity, and motion blur to mimic the Dense-Aqua benchmark.

    Usage:
        gen = DenseAquaGenerator(n_samples=500, seed=42)
        gen.build(output_dir='data/dense_aqua')
    """

    IMG_SIZE = 640
    FISH_COLORS = [(180, 120, 60), (100, 160, 200), (220, 180, 80)]

    def __init__(self, n_samples: int = 500,
                 density_split: Dict[str, float] = None,
                 seed: int = 0):
        self.n_samples = n_samples
        self.density_split = density_split or {
            "low": 0.25, "medium": 0.25, "high": 0.25, "extreme": 0.25
        }
        np.random.seed(seed)
        random.seed(seed)

    # ── fish sprite ──────────────────────────────────────────────────────────

    def _draw_fish(self, canvas: np.ndarray, cx: int, cy: int,
                   w: int, h: int, angle: float, color: Tuple) -> Tuple[int, int, int, int]:
        """Draw an ellipse + triangle fin and return (x1,y1,x2,y2)."""
        axes = (w // 2, h // 2)
        cv2.ellipse(canvas, (cx, cy), axes, angle, 0, 360, color, -1)
        # tail
        tail_pts = np.array([
            [cx + w // 2, cy],
            [cx + w // 2 + h // 2, cy - h // 3],
            [cx + w // 2 + h // 2, cy + h // 3],
        ], dtype=np.int32)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.transform(tail_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        cv2.fillPoly(canvas, [rotated], color)
        x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
        x2, y2 = min(self.IMG_SIZE - 1, cx + w // 2 + h // 2), min(self.IMG_SIZE - 1, cy + h // 2)
        return x1, y1, x2, y2

    def _add_bubbles(self, canvas: np.ndarray, intensity: float = 0.3):
        n_bubbles = int(intensity * 400)
        for _ in range(n_bubbles):
            r = random.randint(2, 8)
            cx = random.randint(0, self.IMG_SIZE - 1)
            cy = random.randint(0, self.IMG_SIZE - 1)
            cv2.circle(canvas, (cx, cy), r, (200, 200, 220), 1)
            cv2.circle(canvas, (cx - r // 3, cy - r // 3), max(1, r // 4),
                       (240, 240, 255), -1)

    def _add_turbidity(self, canvas: np.ndarray, strength: float = 0.4):
        overlay = np.full_like(canvas, (120, 140, 100))
        alpha = np.random.uniform(0.1, strength)
        cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0, canvas)

    def _add_motion_blur(self, canvas: np.ndarray, ksize: int = 7):
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = 1.0 / ksize
        return cv2.filter2D(canvas, -1, kernel)

    # ── sample builder ────────────────────────────────────────────────────────

    def _make_sample(self, n_fish: int, aeration_level: float = 0.2):
        canvas = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        canvas[:] = (30, 60, 80)      # dark-blue background

        boxes, labels = [], []
        for _ in range(n_fish):
            w = random.randint(40, 90)
            h = random.randint(20, 45)
            cx = random.randint(w, self.IMG_SIZE - w)
            cy = random.randint(h, self.IMG_SIZE - h)
            angle = random.uniform(-30, 30)
            color = random.choice(self.FISH_COLORS)
            x1, y1, x2, y2 = self._draw_fish(canvas, cx, cy, w, h, angle, color)
            boxes.append([x1, y1, x2, y2])
            labels.append(0)          # single class: "fish"

        self._add_turbidity(canvas, strength=random.uniform(0.1, 0.5))
        self._add_bubbles(canvas, intensity=aeration_level)
        if random.random() < 0.3:
            canvas = self._add_motion_blur(canvas)

        return canvas, np.array(boxes, dtype=np.float32), np.array(labels)

    # ── public API ────────────────────────────────────────────────────────────

    def build(self, output_dir: str):
        out = Path(output_dir)
        for split in ["train", "val", "test"]:
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)

        counts = {"train": int(self.n_samples * 0.7),
                  "val":   int(self.n_samples * 0.15),
                  "test":  int(self.n_samples * 0.15)}

        density_ranges = {
            "low":     (5,  15),
            "medium":  (15, 30),
            "high":    (30, 50),
            "extreme": (50, 80),
        }

        idx = 0
        annotations = {"train": [], "val": [], "test": []}
        for split, n in counts.items():
            for i in range(n):
                level = random.choices(
                    list(self.density_split.keys()),
                    weights=list(self.density_split.values()))[0]
                lo, hi = density_ranges[level]
                n_fish = random.randint(lo, hi)
                aeration = random.uniform(0.1, 0.6)

                img, boxes, labels = self._make_sample(n_fish, aeration)
                occ_mask = compute_occlusion_mask(boxes)

                img_name = f"{idx:06d}.jpg"
                cv2.imwrite(str(out / split / "images" / img_name), img)

                # YOLO-format label: class cx cy w h  (normalised)
                lbl_lines = []
                for b, l in zip(boxes, labels):
                    x1, y1, x2, y2 = b
                    cx_n = ((x1 + x2) / 2) / self.IMG_SIZE
                    cy_n = ((y1 + y2) / 2) / self.IMG_SIZE
                    w_n  = (x2 - x1) / self.IMG_SIZE
                    h_n  = (y2 - y1) / self.IMG_SIZE
                    lbl_lines.append(f"{l} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
                lbl_name = img_name.replace(".jpg", ".txt")
                with open(out / split / "labels" / lbl_name, "w") as f:
                    f.write("\n".join(lbl_lines))

                annotations[split].append({
                    "image":         img_name,
                    "n_fish":        n_fish,
                    "density_level": level,
                    "aeration":      round(aeration, 3),
                    "occlusion_ids": occ_mask.tolist(),
                })
                idx += 1

            with open(out / split / "meta.json", "w") as f:
                json.dump(annotations[split], f, indent=2)

        print(f"[DenseAquaGenerator] Dataset written to {out} — {idx} samples total.")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Unified FishDataset
# ──────────────────────────────────────────────────────────────────────────────

class FishDataset(Dataset):
    """
    Unified loader for all three datasets in the paper.

    Expected on-disk layout (YOLO format):
        root/
          train/images/*.jpg
          train/labels/*.txt      # class cx cy w h  (normalised)
          train/meta.json         # optional density/aeration metadata
          val/...
          test/...

    Returns:
        image   : FloatTensor (3, H, W)  normalised to [0,1]
        target  : dict with keys
            boxes         FloatTensor (N,4)  [x1,y1,x2,y2] in pixel coords
            labels        LongTensor  (N,)
            occlusion_mask BoolTensor (N,)
            density_level str
            n_fish        int
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root: str, split: str = "train",
                 img_size: int = 640, augment: bool = True,
                 occlusion_iou_thresh: float = 0.5):
        self.root   = Path(root)
        self.split  = split
        self.img_size = img_size
        self.augment  = augment
        self.occ_thresh = occlusion_iou_thresh

        img_dir = self.root / split / "images"
        self.img_paths = sorted(img_dir.glob("*.jpg")) + \
                         sorted(img_dir.glob("*.png"))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")

        # optional per-image metadata
        meta_path = self.root / split / "meta.json"
        self.meta: Dict[str, dict] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                for rec in json.load(f):
                    self.meta[rec["image"]] = rec

        self._build_transforms()

    def _build_transforms(self):
        ops = [transforms.ToTensor()]
        if self.augment:
            ops += [
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
            ]
        ops.append(transforms.Normalize(self.MEAN, self.STD))
        self.tf = transforms.Compose(ops)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_boxes(self, img_path: Path, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        lbl_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
        boxes, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
                x1 = (cx_n - w_n / 2) * w
                y1 = (cy_n - h_n / 2) * h
                x2 = (cx_n + w_n / 2) * w
                y2 = (cy_n + h_n / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _augment_spatial(self, img: np.ndarray, boxes: np.ndarray):
        """Random horizontal flip + mosaic-style random crop."""
        h, w = img.shape[:2]
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            if len(boxes):
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        # random scale/crop
        scale = random.uniform(0.8, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        x0 = random.randint(0, w - new_w)
        y0 = random.randint(0, h - new_h)
        img = img[y0:y0 + new_h, x0:x0 + new_w]
        img = cv2.resize(img, (w, h))
        if len(boxes):
            boxes[:, [0, 2]] -= x0
            boxes[:, [1, 3]] -= y0
            boxes = boxes / [new_w / w, new_h / h, new_w / w, new_h / h]
            boxes = np.clip(boxes, 0, [w, h, w, h])
            # discard degenerate boxes
            valid = (boxes[:, 2] - boxes[:, 0] > 4) & (boxes[:, 3] - boxes[:, 1] > 4)
            boxes = boxes[valid]
        return img, boxes

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img_bgr  = cv2.imread(str(img_path))
        img_bgr  = cv2.resize(img_bgr, (self.img_size, self.img_size))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        boxes, labels = self._load_boxes(img_path, h, w)

        if self.augment and len(boxes):
            img_rgb, boxes = self._augment_spatial(img_rgb, boxes)

        occ_mask = compute_occlusion_mask(boxes, self.occ_thresh) \
                   if len(boxes) else np.zeros(0, bool)

        meta = self.meta.get(img_path.name, {})
        density_level = meta.get("density_level",
                                 assign_density_level(len(boxes)))

        tensor_img = self.tf(img_rgb)

        target = {
            "boxes":          torch.as_tensor(boxes, dtype=torch.float32),
            "labels":         torch.as_tensor(labels, dtype=torch.long),
            "occlusion_mask": torch.as_tensor(occ_mask, dtype=torch.bool),
            "density_level":  density_level,
            "n_fish":         len(boxes),
            "img_id":         img_path.stem,
        }
        return tensor_img, target

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        return torch.stack(images), list(targets)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloader(root: str, split: str, batch_size: int = 32,
                     num_workers: int = 4, img_size: int = 640,
                     augment: Optional[bool] = None) -> DataLoader:
    aug = (split == "train") if augment is None else augment
    ds  = FishDataset(root, split=split, img_size=img_size, augment=aug)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=(split == "train"),
                      num_workers=num_workers,
                      collate_fn=FishDataset.collate_fn,
                      pin_memory=True, drop_last=(split == "train"))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Quick sanity test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic Dense-Aqua dataset …")
    gen = DenseAquaGenerator(n_samples=200, seed=42)
    gen.build("data/dense_aqua")

    loader = build_dataloader("data/dense_aqua", split="train",
                              batch_size=4, num_workers=0)
    imgs, targets = next(iter(loader))
    print(f"  Image batch: {imgs.shape}")
    print(f"  First sample — fish count: {targets[0]['n_fish']}, "
          f"density: {targets[0]['density_level']}, "
          f"occluded: {targets[0]['occlusion_mask'].sum().item()}")
    print("Dataset sanity check passed ✓")
