import os
from pathlib import Path
from joblib import load
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

def _latest_bundle():
    cand = sorted(MODELS.glob("sentiment_bundle_runtime_*.joblib"))
    return cand[-1] if cand else None

def test_bundle_exists():
    p = _latest_bundle()
    assert p is not None, "Aucun bundle sentiment trouvé dans models/"

def test_bundle_structure_and_infer():
    p = _latest_bundle()
    art = load(p)
    # clés minimales
    for k in ["model", "selector_chi2", "vectorizer_word", "vectorizer_char", "threshold"]:
        assert k in art, f"Clé manquante dans le bundle: {k}"

    vec_w = art["vectorizer_word"]
    vec_c = art["vectorizer_char"]
    sel   = art["selector_chi2"]
    clf   = art["model"]

    texts = [
        "Great battery life but the screen is dim.",
        "Terrible packaging, but excellent sound.",
        "Worst purchase ever, completely broken!"
    ]
    Xw = vec_w.transform(texts)
    Xc = vec_c.transform(texts)
    from scipy import sparse
    X = sparse.hstack([Xw, Xc]).tocsr()
    Xsel = sel.transform(X)

    proba = clf.predict_proba(Xsel)[:,1]
    assert proba.shape == (3,), "La sortie proba doit avoir 3 éléments"
    assert np.isfinite(proba).all(), "Probabilités non finies"
