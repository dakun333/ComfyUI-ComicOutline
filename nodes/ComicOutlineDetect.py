"""
🎨 ComfyUI Comic / Manga Outline Detection
- Clear black strokes on white background
- Alpha/background foreground masking + bilateral denoise + luminance/color edges
- Connected-component speckle removal
- 纯 OpenCV 实现，无模型依赖
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


# ======================================================================
# 工具函数（从原脚本移植，保持不变）
# ======================================================================

def keep_large_components(mask: np.ndarray, min_area: int = 50, keep_largest: bool = False) -> np.ndarray:
    """Remove tiny isolated foreground components."""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m, dtype=np.uint8)
    if n <= 1:
        return out

    areas = stats[1:, cv2.CC_STAT_AREA]
    if keep_largest:
        idx = int(np.argmax(areas)) + 1
        out[labels == idx] = 255
    else:
        for i, area in enumerate(areas, start=1):
            if area >= min_area:
                out[labels == i] = 255
    return out


def estimate_foreground_mask(
    rgba: np.ndarray,
    alpha_thr: int = 24,
    bg_threshold: float = 18.0,
) -> np.ndarray:
    """
    Foreground mask:
    1. Transparent PNG: use alpha and remove alpha speckles.
    2. RGB/no alpha: estimate border background color in LAB and threshold color distance.
    """
    rgb = rgba[..., :3].astype(np.uint8)
    alpha = rgba[..., 3].astype(np.uint8)
    h, w = alpha.shape

    has_useful_alpha = alpha.min() < 250
    if has_useful_alpha:
        mask = (alpha > alpha_thr).astype(np.uint8) * 255
        min_area = max(20, int(h * w * 0.00003))
        mask = keep_large_components(mask, min_area=min_area, keep_largest=False)
    else:
        border = np.concatenate(
            [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]],
            axis=0,
        ).astype(np.uint8)
        bg_rgb = np.median(border, axis=0).astype(np.uint8).reshape(1, 1, 3)

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]

        diff = lab - bg_lab
        dist = np.sqrt((0.65 * diff[..., 0]) ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)
        dist8 = np.clip(dist, 0, 255).astype(np.uint8)

        otsu_thr, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(bg_threshold, float(otsu_thr) * 0.55)
        mask = (dist > thr).astype(np.uint8) * 255

        min_area = max(50, int(h * w * 0.00015))
        mask = keep_large_components(mask, min_area=min_area, keep_largest=False)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
    mask = cv2.dilate(mask, k3, iterations=1)
    return mask


def auto_canny(gray: np.ndarray, mask: Optional[np.ndarray] = None, sigma: float = 0.28) -> np.ndarray:
    """Median-based Canny thresholds computed inside the foreground mask."""
    if mask is not None and np.any(mask):
        vals = gray[mask > 0]
    else:
        vals = gray.reshape(-1)

    med = float(np.median(vals))
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    if upper <= lower + 10:
        lower = max(0, upper - 40)

    return cv2.Canny(gray, lower, upper, L2gradient=True)


def remove_edge_noise(edges: np.ndarray, min_area: int = 10, min_length: int = 12) -> np.ndarray:
    """Remove small edge speckles while keeping short but meaningful strokes."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    out = np.zeros_like(edges, dtype=np.uint8)

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area or max(w, h) >= min_length:
            out[labels == i] = 255

    return out


def tensor_to_rgba(tensor: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE (BHWC, 0-1 float) → RGBA uint8 numpy (取 batch 第一张)。"""
    img = tensor[0].cpu().numpy()
    # 处理单通道灰度图
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.float32)
    elif img.shape[-1] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.float32)
    else:
        # 4 通道（RGBA）
        alpha = img[..., 3:4] * 255.0

    rgb = (img[..., :3] * 255.0).clip(0, 255)
    alpha = alpha.clip(0, 255)

    rgba = np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)
    return rgba


def outline_to_tensor(outline: np.ndarray) -> torch.Tensor:
    """灰度轮廓图 → ComfyUI IMAGE (1HWC, 0-1) float32 RGB。"""
    # outline: HxW uint8, 0=黑线 255=白底
    rgb = np.stack([outline] * 3, axis=-1).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


# ======================================================================
# 核心算法（输入改为 RGBA numpy 数组，不再读文件）
# ======================================================================

def comic_outline_from_rgba(
    rgba: np.ndarray,
    alpha_thr: int = 24,
    line_width: int = 3,
    edge_percentile: float = 90.0,
    min_edge_area: Optional[int] = None,
    bg_threshold: float = 18.0,
    add_outer_contour: bool = True,
    invert: bool = False,
) -> np.ndarray:
    """从 RGBA numpy 数组提取漫画轮廓，返回 HxW uint8（0=黑线 255=白底）。"""
    rgb = rgba[..., :3].astype(np.uint8)
    alpha = rgba[..., 3].astype(np.uint8)
    h, w = alpha.shape

    # 透明图复合到白底
    a = alpha.astype(np.float32) / 255.0
    comp = (rgb.astype(np.float32) * a[..., None] + 255.0 * (1.0 - a[..., None])).astype(np.uint8)

    mask = estimate_foreground_mask(rgba, alpha_thr=alpha_thr, bg_threshold=bg_threshold)
    mask_edge_area = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    # 双边滤波 + 中值去噪
    smooth = cv2.bilateralFilter(comp, d=7, sigmaColor=45, sigmaSpace=7)
    smooth = cv2.medianBlur(smooth, 3)

    # 1) 亮度边缘
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    edges_luma = auto_canny(gray, mask_edge_area, sigma=0.28)

    # 2) LAB 颜色边界边缘
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    mag = np.zeros((h, w), np.float32)
    for c in range(3):
        gx = cv2.Scharr(lab[..., c], cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(lab[..., c], cv2.CV_32F, 0, 1)
        mag += gx * gx + gy * gy
    mag = np.sqrt(mag)

    valid = mag[mask_edge_area > 0]
    if valid.size > 0:
        color_thr = np.percentile(valid, edge_percentile)
    else:
        color_thr = np.percentile(mag, edge_percentile)
    edges_color = (mag > color_thr).astype(np.uint8) * 255

    # 3) 外部轮廓（来自 mask 梯度）
    if add_outer_contour:
        outer = cv2.morphologyEx(
            mask,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
    else:
        outer = np.zeros_like(mask)

    edges = cv2.bitwise_or(edges_luma, edges_color)
    edges = cv2.bitwise_or(edges, outer)

    # 限制在前景内
    edges = cv2.bitwise_and(edges, mask_edge_area)

    # 连接微小断口
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    if min_edge_area is None:
        min_edge_area = max(8, int(h * w * 0.000006))
    edges = remove_edge_noise(edges, min_area=min_edge_area, min_length=12)

    # 线条粗细
    if line_width > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width, line_width))
        edges = cv2.dilate(edges, k, iterations=1)

    if invert:
        return edges

    out = np.full_like(edges, 255)
    out[edges > 0] = 0
    return out


# ======================================================================
# ComfyUI Node
# ======================================================================


class ComicOutlineDetect:
    """🎨 漫画轮廓检测 | 多通道边缘融合 + 连通域去噪"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "line_width": (
                    "INT",
                    {"default": 3, "min": 1, "max": 8, "step": 1},
                ),
                "edge_percentile": (
                    "FLOAT",
                    {"default": 90.0, "min": 75.0, "max": 99.0, "step": 0.5},
                ),
                "alpha_thr": (
                    "INT",
                    {"default": 24, "min": 0, "max": 255, "step": 1},
                ),
                "bg_threshold": (
                    "FLOAT",
                    {"default": 18.0, "min": 1.0, "max": 100.0, "step": 1.0},
                ),
                "invert": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "add_outer_contour": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "min_edge_area": (
                    "INT",
                    {"default": -1, "min": -1, "max": 10000, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("轮廓图",)
    FUNCTION = "detect"
    CATEGORY = "🎨 漫画轮廓"

    def detect(
        self,
        图像: torch.Tensor,
        line_width: int = 3,
        edge_percentile: float = 90.0,
        alpha_thr: int = 24,
        bg_threshold: float = 18.0,
        invert: bool = False,
        add_outer_contour: bool = True,
        min_edge_area: int = -1,
    ) -> Tuple[torch.Tensor]:
        min_edge = None if min_edge_area < 0 else min_edge_area

        print(
            f"[ComicOutline] line_width={line_width} | edge_percentile={edge_percentile} | "
            f"min_edge_area={min_edge} | invert={invert}"
        )

        rgba = tensor_to_rgba(图像)

        result = comic_outline_from_rgba(
            rgba,
            alpha_thr=alpha_thr,
            line_width=line_width,
            edge_percentile=edge_percentile,
            min_edge_area=min_edge,
            bg_threshold=bg_threshold,
            add_outer_contour=add_outer_contour,
            invert=invert,
        )

        return (outline_to_tensor(result),)
