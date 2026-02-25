"""Morphology-based utilities for removing OMR grid lines.

This module is intentionally lightweight and OpenCV-only so it can run on CPU
without adding heavy dependencies.
"""

from __future__ import annotations

import cv2
import numpy as np


def remove_grid_lines(image_gray: np.ndarray) -> np.ndarray:
    """Remove long vertical grid lines from a grayscale/binary digit tile.

    The function targets separators that span most of the tile height, which are
    common in OMR layouts and frequently confuse digit OCR models. Thin digit
    strokes (including vertical strokes such as digit "1") are preserved by
    using a long vertical kernel that only matches near full-height structures.

    Args:
        image_gray: Input image in grayscale or thresholded format.

    Returns:
        Gridline-reduced image with the same shape and dtype as input.
    """
    if image_gray is None or image_gray.size == 0:
        return image_gray

    if image_gray.ndim == 3:
        gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_gray.copy()

    h = int(gray.shape[0])
    if h < 4:
        return gray.astype(image_gray.dtype, copy=False)

    # Keep operation stable for both grayscale and thresholded images.
    work = gray.astype(np.uint8, copy=False)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, int(h * 0.75))))
    vertical_lines = cv2.morphologyEx(work, cv2.MORPH_OPEN, vertical_kernel)
    cleaned = cv2.subtract(work, vertical_lines)

    # Preserve tiny high-frequency strokes that may be weakened by subtraction.
    # This selectively restores very thin components from the original image.
    residue = cv2.subtract(work, cleaned)
    restore_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    thin_restore = cv2.erode(residue, restore_kernel, iterations=1)
    cleaned = cv2.max(cleaned, cv2.subtract(work, cv2.subtract(residue, thin_restore)))

    return cleaned.astype(image_gray.dtype, copy=False)

