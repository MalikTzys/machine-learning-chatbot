# Machine Learning (Chatbot) - Easy to Train

Sistem chatbot multibahasa berbasis machine learning yang mendukung Bahasa Indonesia, Inggris, dan Jepang dengan retrieval TF-IDF dan kemampuan fine-tuning transformer.

## Fitur Utama

- **Dukungan Multibahasa**: Indonesia, Inggris, dan Jepang dengan deteksi bahasa otomatis
- **Arsitektur Dual-Mode**: Sistem retrieval TF-IDF + transformer opsional
- **RESTful API**: Dibangun dengan FastAPI untuk performa tinggi
- **Fleksibel**: Mendukung MySQL dan SQLite dengan tracking riwayat percakapan
- **Konfigurasi Tone**: Netral, sopan, atau ramah

## Instalasi

### Persyaratan Sistem

- Python 3.8+
- RAM minimal 4GB (8GB untuk training transformer)
- GPU CUDA (opsional, untuk akselerasi)

### Langkah Instalasi

#### 1. Setup Environment

```bash
mkdir multilingual-chatbot && cd multilingual-chatbot
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

#### 2. Install Dependencies

Buat file `requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
pymysql==1.1.0
scikit-learn==1.3.2
joblib==1.3.2
numpy==1.24.3
transformers==4.35.2
torch==2.1.1
```

Install:
```bash
pip install -r requirements.txt
```

#### 3. Persiapan Dataset

Buat direktori data dan file dataset:

```bash
mkdir -p data
```

Format data yang didukung (pilih salah satu):

**JSONL** (Direkomendasikan):
```json
{"lang":"ID","input":"Halo, apa kabar?","response":"Halo! Saya baik-baik saja, terima kasih."}
{"lang":"EN","input":"Hello, how are you?","response":"Hello! I'm doing well, thank you."}
{"lang":"JP","input":"こんにちは","response":"こんにちは！元気です。"}
```

**TSV**:
```
ID	Halo, apa kabar?	Halo! Saya baik-baik saja.
EN	Hello, how are you?	Hello! I'm doing well, thank you.
```

**CSV**:
```csv
ID,"Halo, apa kabar?","Halo! Saya baik-baik saja."
EN,"Hello, how are you?","Hello! I'm doing well, thank you."
```

**Panduan Dataset**:
- Minimal 10 sampel per bahasa
- Semakin banyak data, semakin baik performa model
- Seimbangkan jumlah sampel antar bahasa

#### 4. Konfigurasi

Buat file `.env` (opsional):

```bash
# Konfigurasi Dataset
CHATBOT_DATA_FILE=data/data.txt

# Konfigurasi Model
CHATBOT_MODEL_TYPE=sklearn
CHATBOT_MIN_SAMPLES_PER_LANG=10

# Parameter TF-IDF
CHATBOT_TFIDF_MAX_FEATURES=50000
CHATBOT_RETRIEVAL_TOP_K=3

# Konfigurasi API
CHATBOT_API_HOST=0.0.0.0
CHATBOT_API_PORT=8000

# Database (opsional)
# DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/chatbot

# Transformer (opsional)
CHATBOT_USE_TRANSFORMERS=0
CHATBOT_HF_MODEL_NAME=google/mt5-small
```

## Deployment

### Mode Development

Jalankan training dan server sekaligus:

```bash
python main.py
```

### Training Saja

```bash
python train_model.py
```

### Server Saja

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload
```

## Dokumentasi API

### Health Check

**Endpoint**: `GET /health`

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

### Chat

**Endpoint**: `POST /chat`

**Parameter**:
| Parameter | Tipe | Wajib | Keterangan |
|-----------|------|-------|------------|
| user_id | string | Ya | ID unik user |
| message | string | Ya | Pesan dari user |
| lang | string | Tidak | Kode bahasa (ID/EN/JP) |
| tone | string | Tidak | Tone respons (neutral/polite/friendly) |

**Contoh Request**:

Bahasa Indonesia:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Halo, apa kabar?",
    "lang": "ID",
    "tone": "friendly"
  }'
```

Bahasa Inggris:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user1","message":"Who are you?"}'
```

Bahasa Jepang:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user1","message":"こんにちは"}'
```

**Response**:
```json
{
  "lang": "ID",
  "response": "Halo! Saya baik-baik saja, terima kasih."
}
```

### Training

**Endpoint**: `POST /train`

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"data_file": "data/data.txt"}'
```

Response:
```json
{"status": "trained"}
```

## Contoh Integrasi

### Python Client

```python
import requests

class ChatbotClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def chat(self, user_id, message, lang=None, tone=None):
        payload = {"user_id": user_id, "message": message}
        if lang:
            payload["lang"] = lang
        if tone:
            payload["tone"] = tone
            
        response = requests.post(f"{self.base_url}/chat", json=payload)
        return response.json()

# Penggunaan
client = ChatbotClient()

# Chat dalam bahasa Indonesia
result = client.chat(
    user_id="user123",
    message="Siapa kamu?",
    lang="ID",
    tone="friendly"
)
print(f"Bot: {result['response']}")

# Chat dengan deteksi bahasa otomatis
result = client.chat(
    user_id="user123",
    message="Hello, how can you help me?"
)
print(f"Bot [{result['lang']}]: {result['response']}")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class ChatbotClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }

  async chat(userId, message, lang = null, tone = null) {
    const payload = { user_id: userId, message };
    if (lang) payload.lang = lang;
    if (tone) payload.tone = tone;

    const response = await axios.post(`${this.baseURL}/chat`, payload);
    return response.data;
  }
}

// Penggunaan
const client = new ChatbotClient();

// Chat dalam bahasa Indonesia
client.chat('user123', 'Halo, apa kabar?', 'ID', 'friendly')
  .then(result => console.log(`Bot: ${result.response}`))
  .catch(error => console.error(error));
```

## Deployment Produksi

### Docker

Buat `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /data

EXPOSE 8000

CMD ["python", "main.py"]
```

Build dan jalankan:

```bash
docker build -t chatbot:latest .

docker run -d \
  --name chatbot \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -v $(pwd)/models:/app/models \
  chatbot:latest
```

### Docker Compose

Buat `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=mysql+pymysql://chatbot:password@db:3306/chatbot
    volumes:
      - ./data:/data
      - ./models:/app/models
    depends_on:
      - db

  db:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=rootpassword
      - MYSQL_DATABASE=chatbot
      - MYSQL_USER=chatbot
      - MYSQL_PASSWORD=password
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  mysql_data:
```

Deploy:
```bash
docker-compose up -d
```

## Troubleshooting

### Model Tidak Ditemukan

```bash
# Train model terlebih dahulu
python train_model.py

# Cek folder models
ls -R models/
```

### Sampel Training Kurang

```bash
# Tambah data atau kurangi threshold
export CHATBOT_MIN_SAMPLES_PER_LANG=5
```

### Out of Memory (Transformer)

```bash
# Kurangi batch size
export CHATBOT_HF_BATCH_SIZE=2

# Atau gunakan model lebih kecil
export CHATBOT_HF_MODEL_NAME=google/mt5-small

# Atau nonaktifkan transformer
export CHATBOT_USE_TRANSFORMERS=0
```

### Koneksi Database Gagal

```bash
# Untuk MySQL, cek koneksi
mysql -u username -p -h host database

# Atau gunakan SQLite (hapus DATABASE_URL)
unset DATABASE_URL
```

## Konfigurasi Lanjutan

### Aktifkan Transformer

```bash
export CHATBOT_USE_TRANSFORMERS=1
export CHATBOT_HF_MODEL_NAME=google/mt5-small
python train_model.py
```

### Optimasi TF-IDF

Edit di `config.py`:

```python
TFIDF_MAX_FEATURES = 50000  # Ukuran vocabulary
NGRAM_RANGE = (1, 3)        # Unigram sampai trigram
RETRIEVAL_TOP_K = 3         # Jumlah kandidat respons
```

## Testing

Buat `tests/test_api.py`:

```python
from fastapi.testclient import TestClient
from web_app import app

client = TestClient(app)

def test_chat():
    response = client.post("/chat", json={
        "user_id": "test",
        "message": "Hello"
    })
    assert response.status_code == 200
    assert "response" in response.json()
```

Jalankan:
```bash
pip install pytest
pytest tests/ -v
```

## Roadmap Pengembangan

### Fase 1: Peningkatan Core
- [ ] Dukungan bahasa tambahan (Mandarin, Korea, Arab)
- [ ] Percakapan multi-turn dengan context
- [ ] Deteksi bahasa menggunakan ML

### Fase 2: Performa
- [ ] Caching dengan Redis
- [ ] Inference asynchronous
- [ ] Model quantization

### Fase 3: Fitur Lanjutan
- [ ] Fine-tuning dengan LoRA
- [ ] Integrasi RAG (Retrieval-Augmented Generation)
- [ ] A/B testing untuk kualitas respons

### Fase 4: Integrasi Platform
- [ ] Bot Telegram
- [ ] WhatsApp Business API
- [ ] Slack & Discord bot

## Lisensi & Dukungan

- **Status Proyek**: Aktif
- **Python Minimum**: 3.8
- **Sistem Direkomendasikan**: Linux/Unix dengan GPU

---

**Versi Dokumentasi**: 8.7.1
**Terakhir Diperbarui**: November 2025