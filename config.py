from __future__ import annotations
import os
from pathlib import Path

# Jalur proyek
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("CHATBOT_DATA_FILE", "/data/data.txt"))  # baca langsung dari jalur lampiran
MODELS_DIR = BASE_DIR / "models"

# Bahasa yang didukung dan pemetaannya
# Kode bahasa dinormalisasi ke huruf besar
SUPPORTED_LANGUAGES = ["ID", "EN", "JP"]
DEFAULT_LANG = "EN"

# Konfigurasi pelatihan
MODEL_TYPE = os.environ.get("CHATBOT_MODEL_TYPE", "sklearn")  # "sklearn" | "transformers"
RANDOM_SEED = int(os.environ.get("CHATBOT_RANDOM_SEED", 42))
TEST_SIZE = float(os.environ.get("CHATBOT_TEST_SIZE", 0.0))  # tidak digunakan untuk baseline retrieval, disimpan untuk masa depan
MIN_SAMPLES_PER_LANG = int(os.environ.get("CHATBOT_MIN_SAMPLES_PER_LANG", 10))

# Model retrieval Sklearn
TFIDF_MAX_FEATURES = int(os.environ.get("CHATBOT_TFIDF_MAX_FEATURES", 50000))
NGRAM_RANGE = (1, 3)  # n-gram kata untuk EN/ID; n-gram karakter akan digunakan untuk JP
RETRIEVAL_TOP_K = int(os.environ.get("CHATBOT_RETRIEVAL_TOP_K", 3))

# Transformers (opsional, dipercepat GPU jika tersedia)
USE_TRANSFORMERS = bool(int(os.environ.get("CHATBOT_USE_TRANSFORMERS", "0")))  # atur "1" untuk mengaktifkan
HF_MODEL_NAME = os.environ.get("CHATBOT_HF_MODEL_NAME", "google/mt5-small")  # varian T5 multibahasa
HF_MAX_INPUT_LENGTH = int(os.environ.get("CHATBOT_HF_MAX_INPUT_LENGTH", 256))
HF_MAX_OUTPUT_LENGTH = int(os.environ.get("CHATBOT_HF_MAX_OUTPUT_LENGTH", 128))
HF_TRAIN_EPOCHS = int(os.environ.get("CHATBOT_HF_TRAIN_EPOCHS", 1))
HF_BATCH_SIZE = int(os.environ.get("CHATBOT_HF_BATCH_SIZE", 4))
HF_LR = float(os.environ.get("CHATBOT_HF_LR", 5e-5))

# Konfigurasi database
# Prioritaskan MySQL melalui DATABASE_URL='mysql+pymysql://user:pass@host:port/dbname'
DATABASE_URL = os.environ.get("DATABASE_URL")  # jika tidak disediakan, fallback ke sqlite
SQLITE_FALLBACK = BASE_DIR / "chatbot.db"

# Konfigurasi API
API_HOST = os.environ.get("CHATBOT_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("CHATBOT_API_PORT", 8000))
API_DEBUG = bool(int(os.environ.get("CHATBOT_API_DEBUG", "0")))

# Opsi inferensi
DEFAULT_TONE = os.environ.get("CHATBOT_DEFAULT_TONE", "neutral")  # "neutral" | "polite" | "friendly"