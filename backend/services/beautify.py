import cv2
import numpy as np
from PIL import Image

from backend.models.presets import PRESET_CONFIGS
from backend.services.converters import pil_to_cv, cv_to_pil
from backend.services.face_detection import detect_face_and_landmarks
from backend.services.filters import apply_unsharp_mask, boost_saturation
from backend.services.masks import (
    compute_edge_preserve_mask,
    create_cheek_highlight_mask,
    create_skin_mask,
    create_under_eye_mask,
    create_wrinkle_mask,
)


def beautify_image(img_pil: Image.Image, preset: str) -> Image.Image:
    """
    Core retouch pipeline:
    - Face detection and landmarks
    - Skin and under-eye masking
    - Frequency separation smoothing
    - Preset-specific adjustments
    """
    config = PRESET_CONFIGS.get(preset)
    if config is None:
        return img_pil

    img_bgr = pil_to_cv(img_pil)  # float32 [0,1], BGR
    h, w = img_bgr.shape[:2]

    landmarks, _ = detect_face_and_landmarks(img_bgr)
    if landmarks is None:
        print("Tidak ada wajah terdeteksi, mengembalikan gambar asli.")
        return img_pil

    skin_mask = create_skin_mask(img_bgr, landmarks)
    under_eye_mask = create_under_eye_mask(img_bgr, landmarks)

    img_bgr_8u = (np.clip(img_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)

    img_lab = cv2.cvtColor(img_bgr_8u, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)

    L = L.astype(np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    skin_mask_u8 = (skin_mask * 255).astype(np.uint8)
    skin_pixels = L[skin_mask_u8 > 0]
    mean_L = float(np.mean(skin_pixels)) if len(skin_pixels) > 0 else 128.0

    edge_preserve_mask = compute_edge_preserve_mask(img_bgr)
    cheek_mask = create_cheek_highlight_mask(img_bgr.shape, landmarks)
    wrinkle_mask = create_wrinkle_mask(img_bgr.shape, landmarks, under_eye_mask)

    delta_L = np.clip(config.target_L - mean_L, 0, config.max_delta_L)
    L = L + delta_L * (skin_mask)
    L = np.clip(L, 0, 255)

    L_out = L.astype(np.uint8)
    A_out = np.clip(A, 0, 255).astype(np.uint8)
    B_out = np.clip(B, 0, 255).astype(np.uint8)

    assert L_out.shape == A_out.shape == B_out.shape, f"Shape LAB tidak sama: {L_out.shape}, {A_out.shape}, {B_out.shape}"

    lab_merged = cv2.merge([L_out, A_out, B_out])
    img_bgr_tone = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    sigma = max(int(max(h, w) * 0.01), 3)
    lf = cv2.GaussianBlur(img_bgr_tone, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hf = img_bgr_tone - lf

    base = lf
    orig = img_bgr_tone
    smooth = base * config.smooth_strength + orig * (1.0 - config.smooth_strength)

    edge_skin_mask = np.clip(skin_mask * edge_preserve_mask, 0.0, 1.0)
    skin_mask_3c = np.dstack([skin_mask] * 3)
    edge_skin_mask_3c = np.dstack([edge_skin_mask] * 3)
    img_smooth_skin = orig * (1.0 - edge_skin_mask_3c) + smooth * edge_skin_mask_3c

    if config.eye_smooth_strength > 0:
        sigma_eye = sigma * 0.7
        lf_eye = cv2.GaussianBlur(img_bgr_tone, (0, 0), sigmaX=sigma_eye, sigmaY=sigma_eye)
        extra_smooth = lf_eye * config.eye_smooth_strength + img_bgr_tone * (1.0 - config.eye_smooth_strength)

        under_eye_3c = np.dstack([under_eye_mask] * 3)
        img_smooth_skin = img_smooth_skin * (1.0 - under_eye_3c) + extra_smooth * under_eye_3c

    result = img_smooth_skin

    if config.glow_strength > 0:
        blur_glow = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma * 0.8, sigmaY=sigma * 0.8)
        glow_layer = blur_glow
        result = result * (1.0 - config.glow_strength * skin_mask_3c) + glow_layer * (config.glow_strength * skin_mask_3c)

    if config.hydration_highlight > 0:
        cheek_3c = np.dstack([cheek_mask] * 3)
        lifted = np.clip(result + config.hydration_highlight * 0.4, 0.0, 1.0)
        result = result * (1.0 - config.hydration_highlight * cheek_3c) + lifted * (config.hydration_highlight * cheek_3c)

    if config.wrinkle_soften > 0:
        wrinkle_3c = np.dstack([wrinkle_mask] * 3)
        wrinkle_blur = cv2.bilateralFilter((result * 255).astype(np.uint8), 9, 30, 9)
        wrinkle_blur = wrinkle_blur.astype(np.float32) / 255.0
        result = result * (1.0 - config.wrinkle_soften * wrinkle_3c) + wrinkle_blur * (config.wrinkle_soften * wrinkle_3c)

    if abs(config.saturation_boost - 1.0) > 1e-3:
        result = boost_saturation(result, config.saturation_boost)

    if config.detail_mix > 0:
        detail_mask = np.dstack([edge_preserve_mask * skin_mask] * 3)
        result = np.clip(result + hf * config.detail_mix * detail_mask, 0.0, 1.0)

    if config.edge_enhance_mix > 0 and config.unsharp_amount > 0:
        sharp = apply_unsharp_mask(result, amount=config.unsharp_amount, radius=config.unsharp_radius)
        edge_mask_3c = np.dstack([edge_preserve_mask] * 3)
        result = result * (1.0 - config.edge_enhance_mix * edge_mask_3c) + sharp * (config.edge_enhance_mix * edge_mask_3c)

    result = np.clip(result, 0.0, 1.0)
    return cv_to_pil(result)


__all__ = ["beautify_image"]
