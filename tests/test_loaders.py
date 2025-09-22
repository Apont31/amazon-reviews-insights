# tests/test_loaders.py
from pathlib import Path
from joblib import dump, load
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

MODELS = Path("models")

def _latest_bundle():
    if not MODELS.exists():
        return None
    cands = sorted(list(MODELS.glob("sentiment_bundle_runtime*.joblib")))
    return cands[-1] if cands else None

def _ensure_ci_bundle():
    """
    Retourne le chemin d’un bundle de sentiment.
    S’il n’y en a pas dans models/, on en génère un minuscule pour le CI.
    """
    MODELS.mkdir(parents=True, exist_ok=True)
    p = _latest_bundle()
    if p:
        return p

    # Données minuscules pour un bundle de démonstration
    texts = [
        "good phone with excellent battery",
        "bad phone, terrible battery life",
        "excellent sound quality, great device",
        "screen is dim and the battery is awful",
    ]
    y = np.array([1, 0, 1, 0], dtype=int)

    # Vectorizer (word uni/bi-gram)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)

    # Sélecteur chi² (k réduit)
    k = min(10, X.shape[1])
    sel = SelectKBest(score_func=chi2, k=k).fit(X, y)
    Xs = sel.transform(X)

    # LR rapide
    clf = LogisticRegression(max_iter=1000).fit(Xs, y)

    bundle = {
        "model": clf,
        "selector_chi2": sel,
        "vectorizer_word": vec,   # clé alignée avec ton code
        "vectorizer_char": None,  # pas nécessaire ici
        "threshold": 0.5,
        "meta": {"ci_stub": True, "n_rows": int(len(texts))},
    }
    out = MODELS / "sentiment_bundle_runtime_CI.joblib"
    dump(bundle, out, compress=3)
    return out

def _infer_simple(texts, bundle):
    vec = bundle.get("vectorizer_word") or bundle.get("vectorizer")
    assert vec is not None, "Vectorizer manquant dans le bundle."
    X = vec.transform(texts)

    sel = bundle.get("selector_chi2")
    if sel is not None:
        X = sel.transform(X)

    clf = bundle.get("model")
    assert clf is not None, "Model manquant dans le bundle."

    proba = clf.predict_proba(X)[:, 1]
    return proba

def test_bundle_structure_and_infer():
    p = _ensure_ci_bundle()
    art = load(p)

    # Clés minimales
    assert "model" in art
    assert ("vectorizer_word" in art) or ("vectorizer" in art)
    # selector_chi2 peut être None, mais on vérifie la clé dans notre stub
    assert "selector_chi2" in art

    # Inférence simple
    texts = [
        "great battery life but the screen is dim",
        "terrible packaging but excellent sound",
        "awful phone",
    ]
    proba = _infer_simple(texts, art)
    assert proba.shape == (len(texts),)
    # On vérifie juste que c'est bien des proba dans [0,1]
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0

def test_bundle_exists():
    # On s’assure qu’un bundle existe (réel ou stub CI)
    p = _ensure_ci_bundle()
    assert p is not None and p.exists(), "Aucun bundle sentiment trouvé dans models/"
