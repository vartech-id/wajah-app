import cv2
import numpy as np


def boost_saturation(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    """Increase or reduce saturation while keeping luminance stable."""
    hsv = cv2.cvtColor((np.clip(img_bgr, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    return out


def apply_unsharp_mask(img_bgr: np.ndarray, amount: float = 0.05, radius: float = 1.0) -> np.ndarray:
    """Light unsharp mask to restore subtle texture."""
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=radius, sigmaY=radius)
    enhanced = img_bgr + amount * (img_bgr - blur)
    return np.clip(enhanced, 0.0, 1.0)


__all__ = ["boost_saturation", "apply_unsharp_mask"]
