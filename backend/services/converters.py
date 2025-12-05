from PIL import Image
import numpy as np


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR float32 in range [0, 1]."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr[:, :, ::-1]  # RGB -> BGR


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR float32 [0, 1] to PIL RGB."""
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr[:, :, ::-1])  # BGR -> RGB


__all__ = ["pil_to_cv", "cv_to_pil"]
