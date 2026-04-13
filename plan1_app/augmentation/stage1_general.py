"""Stage 1 — general image degradations (OpenCV)."""

from __future__ import annotations

import random

import cv2
import numpy as np


def apply_gaussian_blur(bgr: np.ndarray, kernel_max: int = 9, sigma_max: float = 2.5) -> np.ndarray:
    """Apply random odd kernel Gaussian blur; kernel in [3, kernel_max]."""
    k = random.randrange(3, kernel_max + 1, 2)
    sigma = random.uniform(0.5, sigma_max)
    return cv2.GaussianBlur(bgr, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_gaussian_noise(bgr: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Additive Gaussian noise in [0,255] space; clipped."""
    if sigma is None:
        sigma = random.uniform(2.0, 18.0)
    noise = np.random.randn(*bgr.shape).astype(np.float32) * sigma
    out = bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_general(bgr: np.ndarray, p_blur: float = 0.5, p_noise: float = 0.5) -> np.ndarray:
    """Compose random blur and/or noise."""
    out = bgr.copy()
    if random.random() < p_blur:
        out = apply_gaussian_blur(out)
    if random.random() < p_noise:
        out = apply_gaussian_noise(out)
    return out
