
from pathlib import Path

# ---------- المسارات ----------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

DETECTOR_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
ENROLL_DIR = BASE_DIR / "enrolled"

# ---------- الكاميرا ----------
CAMERA_INDEX = 0

# ---------- الكشف ----------
DETECTION_SCORE = 0.8

# ---------- التعرف ----------
MATCH_THRESHOLD = 0.363
ENROLL_SAMPLES = 15

# ---------- قرار "غريب" (تصويت) ----------
VOTE_WINDOW = 15
VOTE_UNKNOWN_MIN = 10

# ---------- التنبيه ----------
ALERT_HOLD_SECONDS = 5

# ---------- الأردوينو ----------
SERIAL_PORT = "COM3"
BAUD_RATE = 9600