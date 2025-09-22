
from pathlib import Path
import joblib, numpy as np
from typing import List, Tuple, Optional

# =========================
# SENTIMENT (TF-IDF -> LR)
# =========================

def _find_sentiment_artifact() -> Path:
    cands = []
    cands += sorted(Path("models").glob("clf_logreg_chi2_gridcal_final_*.joblib"))
    cands += [Path("artifacts")/"sentiment_grid_best_calibrated.joblib"]
    cands += sorted(Path("models").glob("clf_logreg_chi2_final_*.joblib"))
    cands = [p for p in cands if p.exists()]
    if not cands:
        raise FileNotFoundError("No sentiment artifact found.")
    return cands[-1]

def load_sentiment_bundle(path: Optional[str]=None):
    p = Path(path) if path else _find_sentiment_artifact()
    obj = joblib.load(p)
    # bundle may be dict or pipeline
    bundle = {"path": str(p)}
    if isinstance(obj, dict):
        bundle.update(obj)
    else:
        bundle["model"] = obj
    # Optional: load TF-IDF vectorizer if saved
    tfidf = None
    for cand in ["models/tfidf_vectorizer.joblib", "artifacts/tfidf_vectorizer.joblib"]:
        q = Path(cand)
        if q.exists():
            tfidf = joblib.load(q)
            break
    bundle["tfidf"] = tfidf  # may be None if your pipeline embeds it
    return bundle

def predict_sentiment_text(texts: List[str], bundle=None):
    """Predict from raw texts. If bundle.model is a full Pipeline it will handle vectorization.
       Else we require a saved TF-IDF vectorizer (bundle['tfidf'])."""
    b = bundle or load_sentiment_bundle()
    clf = b.get("model_cal") or b.get("model_uncal") or b.get("model")
    if hasattr(clf, "predict") and hasattr(clf, "predict_proba") and not b.get("tfidf"):
        # assume pipeline includes vectorizer
        y = clf.predict(texts)
        proba = clf.predict_proba(texts)
        return y.tolist(), proba[:,1].tolist()
    # otherwise we need the vectorizer
    tfidf = b.get("tfidf", None)
    if tfidf is None:
        raise RuntimeError("No TF-IDF in bundle and model is not a full pipeline. Save your vectorizer as models/tfidf_vectorizer.joblib")
    X = tfidf.transform(texts)
    y = clf.predict(X)
    proba = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else None
    return y.tolist(), (proba.tolist() if proba is not None else None)

# =========================
# EMOTIONS (SBERT -> OVR)
# =========================

class OVRListWrapper:
    def __init__(self, models):
        self.models = models
    def predict_proba(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        n, L = X.shape[0], len(self.models)
        out = np.empty((n, L), dtype=np.float32)
        for i in range(0, n, batch):
            j = min(n, i+batch)
            Xb = np.asarray(X[i:j], dtype=np.float32)
            for k, m in enumerate(self.models):
                try:
                    out[i:j, k] = m.predict_proba(Xb)[:,1]
                except Exception:
                    out[i:j, k] = m.decision_function(Xb)
        return out

def _find_emotions_artifact() -> Path:
    c = []
    c += [Path("artifacts")/"emo_grid_best_bundle.joblib"]
    c += sorted(Path("artifacts").glob("emo_sgd_partial_models_resumed_final.joblib"))
    c += sorted(Path("artifacts").glob("emo_sgd_partial_models_final.joblib"))
    c = [p for p in c if p.exists()]
    if not c:
        raise FileNotFoundError("No emotions artifact found.")
    return c[0]

def load_emotions_runtime(path: Optional[str]=None):
    import joblib, json
    p = Path(path) if path else _find_emotions_artifact()
    obj = joblib.load(p)
    thresholds = None
    labels = None
    if isinstance(obj, dict):
        est = obj.get("best_estimator", obj.get("estimator", None))
        thresholds = obj.get("thresholds", thresholds)
        labels = obj.get("label_names_kept", labels)
        if est is None:
            raise ValueError(f"Bundle {p.name} has no best_estimator")
    elif isinstance(obj, list):
        est = OVRListWrapper(obj)
        # try thresholds file
        from glob import glob
        cand_thr = glob("artifacts/emo_thr_mean_floor*.joblib")
        if cand_thr:
            thresholds = joblib.load(cand_thr[-1])
    else:
        est = obj
    return est, thresholds, labels

def encode_sbert(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer(model_name, device=dev)
    X = sbert.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return X

def predict_emotions_text(texts: List[str], runtime=None, thresholds=None, label_names=None):
    est, thr, labels = runtime if runtime else load_emotions_runtime()
    thr = thresholds if thresholds is not None else thr
    labels = label_names if label_names is not None else labels
    X = encode_sbert(texts)
    proba = est.predict_proba(X)
    if thr is None:
        thr = 0.5
    thr_arr = np.asarray(thr) if not np.isscalar(thr) else np.full(proba.shape[1], thr, dtype=float)
    Y = (proba >= thr_arr.reshape(1,-1)).astype(int)
    return (labels or [f"label_{i}" for i in range(proba.shape[1])]), Y.tolist(), proba.tolist()
