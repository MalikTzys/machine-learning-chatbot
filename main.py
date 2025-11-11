from __future__ import annotations
from utils import log_info
from train_model import run_training
from web_app import app
from config import API_HOST, API_PORT, API_DEBUG

if __name__ == "__main__":
    # Contoh: Melatih lalu melayani
    log_info("Starting training...")
    run_training()
    log_info("Launching API...")
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=API_DEBUG)
