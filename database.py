from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config import DATABASE_URL, SQLITE_FALLBACK
from utils import log_info, log_warn

Base = declarative_base()

def _db_url():
    if DATABASE_URL:
        return DATABASE_URL
    # sqlite fallback
    return f"sqlite:///{SQLITE_FALLBACK}"

engine = create_engine(_db_url(), echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    external_id = Column(String(128), unique=True, index=True, nullable=False)
    preferred_lang = Column(String(4), default="EN")
    messages = relationship("Message", back_populates="user")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String(16))  # "user" or "assistant"
    text = Column(Text)
    lang = Column(String(4), default="EN")
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="messages")

def init_db():
    Base.metadata.create_all(engine)
    log_info("Database initialized", url=_db_url())

def get_or_create_user(session, external_id: str, preferred_lang: Optional[str] = None):
    u = session.query(User).filter(User.external_id == external_id).one_or_none()
    if u:
        if preferred_lang and u.preferred_lang != preferred_lang:
            u.preferred_lang = preferred_lang
            session.add(u)
            session.commit()
        return u
    u = User(external_id=external_id, preferred_lang=preferred_lang or "EN")
    session.add(u)
    session.commit()
    return u

def add_message(session, user: User, role: str, text: str, lang: str):
    m = Message(user_id=user.id, role=role, text=text, lang=lang)
    session.add(m)
    session.commit()
    return m
