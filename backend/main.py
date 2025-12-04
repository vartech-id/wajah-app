from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import numpy as np
import cv2
import dlib
import io
from pathlib import Path

app = FastAPI()

# Sesuaikan origin kalau perlu (misal http://localhost:5173 untuk Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Konfigurasi dlib
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PREDICTOR_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
except RuntimeError:
    raise RuntimeError(
        f"Tidak bisa load shape predictor. Pastikan file ada di: {PREDICTOR_PATH}"
    )

VALID_PRESETS = {"cerah", "lembab", "kerutan"}


# -------------------------------------------------------------------
# Util: konversi gambar
# -------------------------------------------------------------------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR float32 [0,1]."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return arr


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR float [0,1] -> PIL RGB."""
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = arr[:, :, ::-1]  # BGR -> RGB
    return Image.fromarray(arr)


# -------------------------------------------------------------------
# Face & mask utilities
# -------------------------------------------------------------------
def detect_face_and_landmarks(img_bgr: np.ndarray):
    """
    Deteksi wajah dan landmark dengan dlib.
    Mengembalikan (landmarks, rect) atau (None, None) jika tidak ada wajah.
    """
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)

    # Supaya deteksi tidak terlalu berat, resize kalau gambar sangat besar
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

    # Ambil wajah paling besar
    faces_sorted = sorted(faces, key=lambda r: r.width() * r.height(), reverse=True)
    face = faces_sorted[0]

    shape = predictor(img_small_gray, face)
    pts = []
    for i in range(68):
        x = int(shape.part(i).x / scale)
        y = int(shape.part(i).y / scale)
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32)

    # Konversi rect ke koordinat full-res
    rect_full = dlib.rectangle(
        int(face.left() / scale),
        int(face.top() / scale),
        int(face.right() / scale),
        int(face.bottom() / scale),
    )

    return pts, rect_full

def create_skin_mask(img_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Buat mask kulit:
    - Include: rahang sampai jidat (di atas alis).
    - Mata & bibir TIDAK dibolongin keras, hanya dikurangi kekuatan efek
      supaya blending natural (tidak seperti topeng).
    Hasil: mask float32 [0,1].
    """
    h, w = img_bgr.shape[:2]
    mask_face = np.zeros((h, w), dtype=np.float32)

    # ---------- 1) POLYGON WAJAH (RAHANG + JIDAT) ----------

    # Jaw line: 0-16
    jaw = landmarks[0:17]

    # Eyebrows: 17-21 (kiri), 22-26 (kanan)
    left_brow = landmarks[17:22]
    right_brow = landmarks[22:27]
    brow_all = np.vstack([left_brow, right_brow])

    # Estimasi tinggi wajah: brow -> dagu
    brow_center_y = float(brow_all[:, 1].mean())
    chin_y = float(landmarks[8, 1])
    face_height = max(chin_y - brow_center_y, 1.0)

    # Tinggi jidat yang kamu suka
    forehead_height = 0.57 * face_height

    # Titik jidat di atas alis
    forehead_points = []
    for (x, y) in brow_all:
        fx = int(x)
        fy = int(max(0, y - forehead_height))
        forehead_points.append((fx, fy))
    forehead_points = np.array(forehead_points, dtype=np.int32)

    # Polygon: rahang kiri → rahang kanan → jidat kanan → jidat kiri
    poly_pts = np.concatenate([jaw, forehead_points[::-1]], axis=0)
    cv2.fillConvexPoly(mask_face, poly_pts, 1.0)

    # Sedikit blur supaya tepi wajah halus
    mask_face = cv2.GaussianBlur(mask_face, (0, 0), sigmaX=5, sigmaY=5)

    # ---------- 2) MASK FITUR (MATA + DALAM BIBIR) ----------

    feature_mask = np.zeros((h, w), dtype=np.float32)

    def fill_region(indices):
        pts = landmarks[indices]
        cv2.fillConvexPoly(feature_mask, pts, 1.0)

    # MATA (kelopak + bola mata)
    fill_region(list(range(36, 42)))  # left eye polygon
    fill_region(list(range(42, 48)))  # right eye polygon

    # HANYA bagian dalam bibir (jadi kulit sekitar bibir tetap bisa kena efek)
    fill_region(list(range(60, 68)))  # inner mouth

    # Blur supaya transisi soft (bukan shape keras)
    feature_soft = cv2.GaussianBlur(feature_mask, (0, 0), sigmaX=7, sigmaY=7)

    # ---------- 3) KOMBINASI: KURANGI EFEK DI FITUR ----------

    # alpha = seberapa kuat kita mengurangi efek di mata/bibir
    # 0.8 berarti: di tengah mata/bibir, efek hampir hilang, pinggiran halus.
    alpha = 0.8

    skin_mask = mask_face * (1.0 - alpha * feature_soft)
    skin_mask = np.clip(skin_mask, 0.0, 1.0)

    return skin_mask.astype(np.float32)


def create_under_eye_mask(img_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Mask khusus area bawah mata untuk preset 'kerutan':
    - Fokus di kantong mata (under-eye), bukan sampai ke bola mata.
    - Dibangun dari bounding box mata lalu diturunkan ke bawah,
      dengan feather halus.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    def add_under_region(start_idx: int):
        eye_pts = landmarks[start_idx:start_idx + 6]  # 6 titik mata
        x, y, ew, eh = cv2.boundingRect(eye_pts)

        # mulai area di bawah mata (sekitar 1x tinggi mata)
        y1 = y + int(eh * 1.0)
        # selesai sedikit lebih ke bawah pipi atas
        y2 = y + int(eh * 2.2)

        # melebar sedikit ke kiri/kanan
        x1 = x - int(ew * 0.4)
        x2 = x + ew + int(ew * 0.4)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, thickness=-1)

    # kedua mata
    add_under_region(36)  # kiri
    add_under_region(42)  # kanan

    # feather halus
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10, sigmaY=10)
    return mask.astype(np.float32)


# -------------------------------------------------------------------
# Core beautify engine
# -------------------------------------------------------------------
def beautify_image(img_pil: Image.Image, preset: str) -> Image.Image:
    """
    Algoritma retouch:
    - Deteksi wajah & landmark
    - Buat mask kulit + mask bawah mata
    - Frequency separation sederhana
    - Terapkan preset cerah / lembab / kerutan
    """
    img_bgr = pil_to_cv(img_pil)  # float32 [0,1], BGR
    h, w = img_bgr.shape[:2]

    # Deteksi wajah
    landmarks, rect = detect_face_and_landmarks(img_bgr)
    if landmarks is None:
        # Tidak ada wajah, kembalikan gambar original
        print("Tidak ada wajah terdeteksi, mengembalikan gambar asli.")
        return img_pil

    # Mask kulit & bawah mata
    skin_mask = create_skin_mask(img_bgr, landmarks)  # [0,1], shape (h, w)
    under_eye_mask = create_under_eye_mask(img_bgr, landmarks)

    # -----------------------------
    # KONVERSI KE LAB (rapi)
    # -----------------------------
    # Pastikan ke uint8 dulu
    img_bgr_8u = (np.clip(img_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)

    img_lab = cv2.cvtColor(img_bgr_8u, cv2.COLOR_BGR2LAB)  # uint8
    L, A, B = cv2.split(img_lab)  # masing2 shape (h, w), uint8

    # Ubah ke float32 untuk perhitungan
    L = L.astype(np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # Hitung brightness rata-rata kulit di channel L
    skin_mask_u8 = (skin_mask * 255).astype(np.uint8)
    skin_pixels = L[skin_mask_u8 > 0]
    if len(skin_pixels) > 0:
        mean_L = float(np.mean(skin_pixels))
    else:
        mean_L = 128.0

    # Parameter preset
    if preset == "cerah":
        target_L = 160.0
        delta_L = np.clip(target_L - mean_L, 0, 20)
        smooth_strength = 0.3
        eye_smooth_strength = 0.2
        glow_strength = 0.05
    elif preset == "lembab":
        target_L = 155.0
        delta_L = np.clip(target_L - mean_L, 0, 18)
        smooth_strength = 0.5
        eye_smooth_strength = 0.35
        glow_strength = 0.12
    elif preset == "kerutan":
        target_L = 150.0
        delta_L = np.clip(target_L - mean_L, 0, 12)
        smooth_strength = 0.35
        eye_smooth_strength = 0.6
        glow_strength = 0.03
    else:
        # Fallback: tidak diubah
        return img_pil

    # 1) Brighten kulit di channel L
    L = L + delta_L * (skin_mask)  # skin_mask [0,1]
    L = np.clip(L, 0, 255)

    # Kembalikan ke uint8 sebelum merge
    L_out = L.astype(np.uint8)
    A_out = np.clip(A, 0, 255).astype(np.uint8)
    B_out = np.clip(B, 0, 255).astype(np.uint8)

    # Pastikan ukuran sama
    assert L_out.shape == A_out.shape == B_out.shape, f"Shape LAB tidak sama: {L_out.shape}, {A_out.shape}, {B_out.shape}"

    lab_merged = cv2.merge([L_out, A_out, B_out])  # semua uint8, shape (h, w, 3)
    img_bgr_tone = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    img_bgr_tone = img_bgr_tone.astype(np.float32) / 255.0

    # -----------------------------
    # FREQUENCY SEPARATION
    # -----------------------------
    sigma = max(int(max(h, w) * 0.01), 3)
    lf = cv2.GaussianBlur(img_bgr_tone, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hf = img_bgr_tone - lf

    # Smoothing via LF + original
    base = lf
    orig = img_bgr_tone
    smooth = base * smooth_strength + orig * (1.0 - smooth_strength)

    skin_mask_3c = np.dstack([skin_mask] * 3)
    img_smooth_skin = orig * (1.0 - skin_mask_3c) + smooth * skin_mask_3c

    # Extra smoothing bawah mata
    if eye_smooth_strength > 0:
        sigma_eye = sigma * 0.7
        lf_eye = cv2.GaussianBlur(img_bgr_tone, (0, 0), sigmaX=sigma_eye, sigmaY=sigma_eye)
        extra_smooth = lf_eye * eye_smooth_strength + img_bgr_tone * (1.0 - eye_smooth_strength)

        under_eye_3c = np.dstack([under_eye_mask] * 3)
        img_smooth_skin = img_smooth_skin * (1.0 - under_eye_3c) + extra_smooth * under_eye_3c

    result = img_smooth_skin

    # Glow halus
    if glow_strength > 0:
        blur_glow = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma * 0.8, sigmaY=sigma * 0.8)
        glow_layer = blur_glow
        result = result * (1.0 - glow_strength * skin_mask_3c) + glow_layer * (glow_strength * skin_mask_3c)

    result = np.clip(result, 0.0, 1.0)
    out_pil = cv_to_pil(result)
    return out_pil

# -------------------------------------------------------------------
# FastAPI endpoint
# -------------------------------------------------------------------
@app.post("/api/beauty")
async def beauty_endpoint(
    image: UploadFile = File(...),
    preset: str = Form(...),
):
    preset = preset.lower()
    if preset not in VALID_PRESETS:
        raise HTTPException(status_code=400, detail="Preset tidak dikenal")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File gambar tidak valid")

    # Jalankan algoritma beautify
    out_img = beautify_image(img, preset)

    # Simpan ke buffer sebagai JPEG high-quality
    buffer = io.BytesIO()
    out_img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    return Response(
        content=buffer.read(),
        media_type="image/jpeg"
    )
