import cv2
import dlib
import numpy as np

from backend.config import PREDICTOR_PATH

# Load dlib detector and shape predictor once
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
except RuntimeError:
    raise RuntimeError(
        f"Tidak bisa load shape predictor. Pastikan file ada di: {PREDICTOR_PATH}"
    )


def detect_face_and_landmarks(img_bgr: np.ndarray):
    """
    Detect face and 68 landmarks with dlib.
    Returns (landmarks, rect) or (None, None) if no face is found.
    """
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)

    scale = 1.0
    target = 1600.0
    if max_side > target:
        scale = target / max_side
        img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr.copy()

    img_small_gray = (img_small * 255).astype(np.uint8)
    img_small_gray = cv2.cvtColor(img_small_gray, cv2.COLOR_BGR2GRAY)

    faces = detector(img_small_gray, 1)
    if len(faces) == 0:
        return None, None

    faces_sorted = sorted(faces, key=lambda r: r.width() * r.height(), reverse=True)
    face = faces_sorted[0]

    shape = predictor(img_small_gray, face)
    pts = []
    for i in range(68):
        x = int(shape.part(i).x / scale)
        y = int(shape.part(i).y / scale)
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32)

    rect_full = dlib.rectangle(
        int(face.left() / scale),
        int(face.top() / scale),
        int(face.right() / scale),
        int(face.bottom() / scale),
    )

    return pts, rect_full


__all__ = ["detect_face_and_landmarks", "detector", "predictor"]
