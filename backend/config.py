from pathlib import Path

# Directories and resource paths
BASE_DIR = Path(__file__).resolve().parent
PREDICTOR_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

# CORS configuration (adjust origins as needed, e.g., http://localhost:5173 for Vite)
CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
