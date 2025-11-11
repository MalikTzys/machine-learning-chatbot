from __future__ import annotations
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import API_HOST, API_PORT, API_DEBUG, DEFAULT_TONE
from database import init_db, SessionLocal, get_or_create_user, add_message
from model_loader import ChatbotModels
from language_selector import select_language
from utils import log_info, log_error
from train_model import run_training

app = FastAPI(title="Multilingual ML Chatbot", version="8.7.1")
models = ChatbotModels()

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="External user id")
    message: str = Field(..., description="User message text")
    lang: Optional[str] = Field(None, description="Preferred language: ID|EN|JP")
    tone: Optional[str] = Field(None, description="Tone: neutral|polite|friendly")

class ChatResponse(BaseModel):
    lang: str
    response: str

class TrainRequest(BaseModel):
    data_file: Optional[str] = Field(None, description="Path to dataset, defaults to /data/data.txt")

@app.on_event("startup")
def on_startup():
    init_db()
    models.load()
    log_info("API started")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = SessionLocal()
    try:
        user = get_or_create_user(session, req.user_id, preferred_lang=req.lang)
        lang = select_language(req.message, req.lang)
        add_message(session, user, "user", req.message, lang)
        tone = req.tone or DEFAULT_TONE

        reply = models.infer(lang, req.message, tone=tone)
        add_message(session, user, "assistant", reply, lang)

        return ChatResponse(lang=lang, response=reply)
    except Exception as e:
        log_error("Chat error", error=str(e))
        return ChatResponse(lang=req.lang or "EN", response="Sorry, something went wrong.")
    finally:
        session.close()

@app.post("/train")
def train(req: TrainRequest):
    """
    Memicu pelatihan untuk me-refresh model. Gunakan dengan hati-hati di production.
    """
    try:
        run_training(data_path=None if not req.data_file else Path(req.data_file))
        # Muat ulang model setelah pelatihan
        models.load()
        return {"status": "trained"}
    except Exception as e:
        log_error("Training failed", error=str(e))
        return {"status": "error", "detail": str(e)}