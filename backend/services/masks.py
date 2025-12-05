import cv2
import numpy as np


def create_skin_mask(img_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Build a soft skin mask:
    - Includes jawline up to the forehead above the brows.
    - Eyes and lips are softened instead of cut out to keep blending natural.
    Returns float32 mask [0, 1].
    """
    h, w = img_bgr.shape[:2]
    mask_face = np.zeros((h, w), dtype=np.float32)

    jaw = landmarks[0:17]

    left_brow = landmarks[17:22]
    right_brow = landmarks[22:27]
    brow_all = np.vstack([left_brow, right_brow])

    brow_center_y = float(brow_all[:, 1].mean())
    chin_y = float(landmarks[8, 1])
    face_height = max(chin_y - brow_center_y, 1.0)

    forehead_height = 0.40 * face_height

    forehead_points = []
    for (x, y) in brow_all:
        fx = int(x)
        fy = int(max(0, y - forehead_height))
        forehead_points.append((fx, fy))
    forehead_points = np.array(forehead_points, dtype=np.int32)

    poly_pts = np.concatenate([jaw, forehead_points[::-1]], axis=0)
    cv2.fillConvexPoly(mask_face, poly_pts, 1.0)

    mask_face = cv2.GaussianBlur(mask_face, (0, 0), sigmaX=5, sigmaY=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_face = cv2.erode(mask_face, kernel, iterations=1)
    mask_face = cv2.dilate(mask_face, kernel, iterations=1)

    feature_mask = np.zeros((h, w), dtype=np.float32)

    def fill_region(indices):
        pts = landmarks[indices]
        cv2.fillConvexPoly(feature_mask, pts, 1.0)

    fill_region(list(range(36, 42)))  # left eye polygon
    fill_region(list(range(42, 48)))  # right eye polygon
    fill_region(list(range(60, 68)))  # inner mouth

    feature_soft = cv2.GaussianBlur(feature_mask, (0, 0), sigmaX=7, sigmaY=7)

    alpha = 0.8

    skin_mask = mask_face * (1.0 - alpha * feature_soft)
    skin_mask = np.clip(skin_mask, 0.0, 1.0)

    return skin_mask.astype(np.float32)


def create_under_eye_mask(img_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Mask area beneath the eyes for the 'kerutan' preset.
    Focused on under-eye bags, not the eyeball.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    def add_under_region(start_idx: int):
        eye_pts = landmarks[start_idx:start_idx + 6]
        x, y, ew, eh = cv2.boundingRect(eye_pts)

        y1 = y + int(eh * 1.0)
        y2 = y + int(eh * 2.2)

        x1 = x - int(ew * 0.4)
        x2 = x + ew + int(ew * 0.4)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, thickness=-1)

    add_under_region(36)  # left eye
    add_under_region(42)  # right eye

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10, sigmaY=10)
    return mask.astype(np.float32)


def compute_edge_preserve_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mask to preserve edges so smoothing does not blur detailed areas."""
    gray = (np.clip(img_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=1.4, sigmaY=1.4)
    mask = 1.0 - edges.astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return np.clip(mask, 0.25, 1.0).astype(np.float32)


def create_cheek_highlight_mask(img_shape, landmarks: np.ndarray) -> np.ndarray:
    """Soft mask around cheeks to apply local glow for the 'lembab' preset."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    left_eye_outer = landmarks[36]
    right_eye_outer = landmarks[45]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]

    left_center = (0.55 * left_mouth + 0.45 * left_eye_outer).astype(np.int32)
    right_center = (0.55 * right_mouth + 0.45 * right_eye_outer).astype(np.int32)

    eye_width = np.linalg.norm(right_eye_outer - left_eye_outer)
    radius = int(max(12, eye_width * 0.18))

    for cx, cy in [left_center, right_center]:
        cv2.circle(mask, (int(cx), int(cy)), radius, 1.0, thickness=-1)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.6, sigmaY=radius * 0.6)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def create_wrinkle_mask(img_shape, landmarks: np.ndarray, under_eye_mask: np.ndarray) -> np.ndarray:
    """
    Aggressive mask for wrinkle-prone areas:
    - Under-eye bags (reuses under_eye_mask)
    - Forehead above brows
    - Nasolabial folds (nose to mouth corners)
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask = np.maximum(mask, under_eye_mask)

    brow_pts = landmarks[17:27]
    x, y, bw, bh = cv2.boundingRect(brow_pts)
    y1 = max(0, y - int(bh * 0.9))
    y2 = min(h, y + int(bh * 0.4))
    x1 = max(0, x - int(bw * 0.1))
    x2 = min(w, x + bw + int(bw * 0.1))
    cv2.rectangle(mask, (x1, y1), (x2, y2), 0.6, thickness=-1)

    left_poly = np.array([landmarks[31], landmarks[33], landmarks[48], landmarks[3]], dtype=np.int32)
    right_poly = np.array([landmarks[35], landmarks[33], landmarks[54], landmarks[13]], dtype=np.int32)
    cv2.fillConvexPoly(mask, left_poly, 0.7)
    cv2.fillConvexPoly(mask, right_poly, 0.7)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8, sigmaY=8)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


__all__ = [
    "create_skin_mask",
    "create_under_eye_mask",
    "compute_edge_preserve_mask",
    "create_cheek_highlight_mask",
    "create_wrinkle_mask",
]
