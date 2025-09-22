
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from serve.infer import load_sentiment_bundle, predict_sentiment_text, load_emotions_runtime, predict_emotions_text

app = FastAPI(title="Amazon Reviews — Sentiment & Emotions API", version="1.0")

class SentimentIn(BaseModel):
    texts: List[str]

class EmotionsIn(BaseModel):
    texts: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/sentiment")
def predict_sentiment(payload: SentimentIn):
    bundle = load_sentiment_bundle()
    y, proba = predict_sentiment_text(payload.texts, bundle=bundle)
    return {"labels": y, "probas": proba}

@app.post("/predict/emotions")
def predict_emotions(payload: EmotionsIn):
    runtime = load_emotions_runtime()
    labels, Y, proba = predict_emotions_text(payload.texts, runtime=runtime)
    return {"labels": labels, "multi_hot": Y, "probas": proba}
