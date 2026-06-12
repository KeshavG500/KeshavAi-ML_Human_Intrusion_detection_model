"""
augmentor.py
────────────
Albumentations-based augmentation pipeline for intrusion detection training.
Includes separate pipelines for training (heavy augmentation) and
validation/inference (light transforms only).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch


# ─────────────────────────── Pipeline Builders ────────────────────────────

def get_train_transforms(
    img_size: int = 640,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Heavy augmentation pipeline used during model training.
    Bounding boxes are transformed alongside the image.
    """
    return A.Compose(
        [
            # ── Spatial ──────────────────────────────────────────────────
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.5,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=0.7,
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),

            # ── Color / Photometric ──────────────────────────────────────
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0,
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),

            # ── Noise / Blur ─────────────────────────────────────────────
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=1.0),
            ], p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),     # simulate camera motion
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),

            # ── Dropout / Occlusion ──────────────────────────────────────
            A.CoarseDropout(
                max_holes=8,
                max_height=img_size // 10,
                max_width=img_size // 10,
                fill_value=(114, 114, 114),
                p=0.3,
            ),

            # ── Weather Simulation ───────────────────────────────────────
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=20, drop_width=1,
                    drop_color=(200, 200, 200), blur_value=5,
                    brightness_coefficient=0.7, rain_type="default", p=1.0
                ),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=1.0),
                A.RandomShadow(p=1.0),
            ], p=0.2),

            # ── Normalize & Convert ──────────────────────────────────────
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",           # [cx, cy, w, h] normalized
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(
    img_size: int = 640,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Minimal transforms for validation and inference — no random augmentation."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )


def get_inference_transforms(
    img_size: int = 640,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Transforms for single-frame inference (no bbox params needed)."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114),
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ─────────────────────────── Mosaic Augmentation ─────────────────────────

class MosaicAugmentation:
    """
    4-image mosaic augmentation — combine 4 training images into one.
    Significantly improves small-object detection performance.
    """

    def __init__(self, img_size: int = 640, p: float = 0.5):
        self.img_size = img_size
        self.p = p

    def __call__(
        self,
        images: List[np.ndarray],
        bboxes_list: List[List],
        labels_list: List[List],
    ) -> Tuple[np.ndarray, List, List]:
        """
        Args:
            images      : List of 4 BGR images.
            bboxes_list : List of 4 bbox lists (YOLO format).
            labels_list : List of 4 class label lists.
        Returns:
            Mosaic image, combined bboxes, combined labels.
        """
        assert len(images) == 4, "MosaicAugmentation requires exactly 4 images."

        if np.random.random() > self.p:
            return images[0], bboxes_list[0], labels_list[0]

        s = self.img_size
        # Random center point
        cx = int(np.random.uniform(s * 0.25, s * 0.75))
        cy = int(np.random.uniform(s * 0.25, s * 0.75))

        mosaic = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        corners = [(0, 0), (s, 0), (0, s), (s, s)]

        all_bboxes, all_labels = [], []

        for i, (img, bboxes, labels) in enumerate(
            zip(images, bboxes_list, labels_list)
        ):
            h, w = img.shape[:2]
            x1, y1 = corners[i]
            x2, y2 = x1 + s, y1 + s
            mosaic[y1:y2, x1:x2] = cv2.resize(img, (s, s))

            for bbox, label in zip(bboxes, labels):
                # Convert YOLO → absolute, offset, convert back
                bx, by, bw, bh = bbox
                abs_x = (bx * s + x1) / (s * 2)
                abs_y = (by * s + y1) / (s * 2)
                abs_w = bw / 2
                abs_h = bh / 2
                all_bboxes.append([abs_x, abs_y, abs_w, abs_h])
                all_labels.append(label)

        # Crop back to img_size
        mosaic = mosaic[s // 2: s + s // 2, s // 2: s + s // 2]
        return mosaic, all_bboxes, all_labels
