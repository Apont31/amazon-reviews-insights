from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI(title="Amazon Reviews Insights API")
ART_PATH = "models/sentiment_v1.joblib"
_art = None

class InText(BaseModel):
    text: str

@app.on_event("startup")
def _load():
    global _art
    try:
        _art = load(ART_PATH)
    except Exception as e:
        _art = None
        print("WARN: artefact non chargé :", e)

@app.get("/health")
def health():
    return {"ok": _art is not None}

@app.post("/predict")
def predict(inp: InText):
    assert _art is not None, "Artefact manquant"
    vec, clf = _art["vectorizer"], _art["model"]
    X = vec.transform([inp.text])
    proba = float(clf.predict_proba(X)[0,1])
    return {"proba_pos": proba, "label": int(proba >= 0.5)}
