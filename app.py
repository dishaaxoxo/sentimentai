"""
AI Sentiment Analysis — FastAPI Backend (HF Inference API version)
===================================================================
HOW TO RUN LOCALLY:
  1. pip install fastapi uvicorn requests python-multipart
  2. Set env var: export HF_TOKEN=hf_yourtoken
  3. uvicorn app:app --reload
  4. Server: http://localhost:8000
  5. Docs:   http://localhost:8000/docs

DEPLOY ON RENDER:
  - Build command: pip install -r requirements.txt
  - Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
  - Add env var:   HF_TOKEN = hf_yourtoken
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import re
import sqlite3
from datetime import datetime, timedelta

# ── CONFIG ───────────────────────────────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN   = os.getenv("HF_TOKEN", "")
DB_PATH    = "./feedback.db"

# ── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="SentimentAI API",
    description="Real-time customer feedback sentiment analysis using DistilBERT via HF API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DATABASE SETUP ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text   TEXT    NOT NULL,
            clean_text TEXT,
            label      TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            source     TEXT    DEFAULT 'api',
            created_at TEXT    NOT NULL
        )
    """)
    conn.commit(); conn.close()

def save_to_db(raw_text, clean_text, label, confidence, source="api"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO feedback (raw_text, clean_text, label, confidence, source, created_at) VALUES (?,?,?,?,?,?)",
        (raw_text, clean_text, label, confidence, source, datetime.utcnow().isoformat())
    )
    conn.commit(); conn.close()

# ── TEXT CLEANING ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── INFERENCE via HF API ──────────────────────────────────────────────────────
def run_inference(text: str):
    cleaned = clean_text(text)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    try:
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": cleaned},
            timeout=15
        )
        resp.raise_for_status()
        scores = resp.json()

        # HF returns [[{label, score}, ...]] — unwrap if nested
        if isinstance(scores, list) and isinstance(scores[0], list):
            scores = scores[0]

        top = max(scores, key=lambda x: x["score"])
        raw_label = top["label"].upper()

        # SST-2 gives POSITIVE/NEGATIVE; map to 3-class with neutral band
        score_map = {s["label"].upper(): s["score"] for s in scores}
        pos = score_map.get("POSITIVE", 0)
        neg = score_map.get("NEGATIVE", 0)

        if pos > 0.75:
            label = "Positive"; conf = pos
        elif neg > 0.75:
            label = "Negative"; conf = neg
        else:
            label = "Neutral"; conf = max(pos, neg)

        return label, round(conf * 100, 2), cleaned

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=503, detail="HF API timed out — model may be loading, retry in ~20s")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"HF API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    source: Optional[str] = "api"

class BatchRequest(BaseModel):
    texts: List[str]
    source: Optional[str] = "batch"

class AnalysisResponse(BaseModel):
    text: str
    label: str
    confidence: float
    timestamp: str

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    token_status = "✅ HF_TOKEN set" if HF_TOKEN else "⚠️  HF_TOKEN not set — rate limits may apply"
    print(token_status)
    print("✅ SentimentAI ready (HF Inference API mode — no local model needed)")

@app.get("/")
def root():
    if os.path.exists("dashboard.html"):
        return FileResponse("dashboard.html")
    return {
        "message": "SentimentAI API is running",
        "endpoints": ["/analyze", "/batch", "/trends", "/history", "/stats", "/docs"]
    }

@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: TextRequest):
    """Analyze sentiment of a single text."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    label, conf, cleaned = run_inference(req.text)
    ts = datetime.utcnow().isoformat()
    save_to_db(req.text, cleaned, label, conf, req.source)
    return AnalysisResponse(text=req.text, label=label, confidence=conf, timestamp=ts)

@app.post("/batch")
def batch_analyze(req: BatchRequest):
    """Analyze multiple texts at once (max 50)."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    results = []
    for text in req.texts[:50]:
        label, conf, cleaned = run_inference(text)
        ts = datetime.utcnow().isoformat()
        save_to_db(text, cleaned, label, conf, req.source)
        results.append({"text": text, "label": label, "confidence": conf, "timestamp": ts})
    summary = {
        "Positive": sum(1 for r in results if r["label"] == "Positive"),
        "Negative": sum(1 for r in results if r["label"] == "Negative"),
        "Neutral":  sum(1 for r in results if r["label"] == "Neutral"),
    }
    return {"total": len(results), "summary": summary, "results": results}

@app.get("/trends")
def get_trends(days: int = 7):
    """Get sentiment volume grouped by day for the past N days."""
    conn = sqlite3.connect(DB_PATH)
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT date(created_at) as day, label, COUNT(*) as cnt "
        "FROM feedback WHERE created_at > ? GROUP BY day, label ORDER BY day",
        (since,)
    ).fetchall()
    conn.close()
    trend = {}
    for day, label, cnt in rows:
        if day not in trend:
            trend[day] = {"Positive": 0, "Negative": 0, "Neutral": 0}
        trend[day][label] = cnt
    return {"days": days, "trend": trend}

@app.get("/history")
def get_history(limit: int = 20, label: Optional[str] = None):
    """Fetch recent inference history."""
    conn = sqlite3.connect(DB_PATH)
    if label:
        rows = conn.execute(
            "SELECT raw_text, label, confidence, created_at FROM feedback "
            "WHERE label=? ORDER BY id DESC LIMIT ?", (label, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT raw_text, label, confidence, created_at FROM feedback "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [{"text": r[0], "label": r[1], "confidence": r[2], "timestamp": r[3]} for r in rows]

@app.get("/stats")
def get_stats():
    """Aggregate statistics for dashboard KPIs."""
    conn = sqlite3.connect(DB_PATH)
    total    = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    by_label = conn.execute("SELECT label, COUNT(*) FROM feedback GROUP BY label").fetchall()
    avg_conf = conn.execute("SELECT AVG(confidence) FROM feedback").fetchone()[0]
    conn.close()
    counts = {row[0]: row[1] for row in by_label}
    return {
        "total": total,
        "positive": counts.get("Positive", 0),
        "negative": counts.get("Negative", 0),
        "neutral":  counts.get("Neutral", 0),
        "avg_confidence": round(avg_conf or 0, 2)
    }

@app.delete("/clear")
def clear_db():
    """Clear all feedback data (for testing)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM feedback")
    conn.commit(); conn.close()
    return {"message": "Database cleared"}
