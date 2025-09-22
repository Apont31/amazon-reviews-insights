# streamlit_app.py
import glob
from pathlib import Path

import numpy as np
import streamlit as st
from joblib import load
from scipy import sparse

# =========================
#   PAGE / STYLING
# =========================
st.set_page_config(page_title="Amazon Reviews — Démo", page_icon="💬", layout="wide")
st.title("💬 Amazon Reviews — Démo")

# =========================
#   SENTIMENT — LOADING
# =========================
@st.cache_resource
def load_sentiment_bundle():
    """
    Charge le dernier bundle entraîné contenant :
      - vectorizer_word
      - vectorizer_char
      - selector_chi2  (n_in=300k -> n_out=50k)
      - model          (LogisticRegression)
      - threshold      (t*)
    """
    paths = sorted(glob.glob("models/sentiment_bundle_runtime_*.joblib"))
    if not paths:
        raise FileNotFoundError(
            "Aucun bundle 'sentiment_bundle_runtime_*.joblib' trouvé dans models/.\n"
            "Entraîne-le puis relance l'appli."
        )
    bpath = paths[-1]
    b = load(bpath)
    b["_path"] = bpath
    # Sanity keys
    for key in ["vectorizer_word", "vectorizer_char", "selector_chi2", "model"]:
        if key not in b:
            raise KeyError(f"Clé manquante dans le bundle: {key}")
    return b

def _stack_word_char(texts, vw, vc):
    """TF-IDF word + char -> matrice 300k colonnes (200k + 100k)."""
    Xw = vw.transform(texts)
    Xc = vc.transform(texts)
    return sparse.hstack([Xw, Xc], format="csr")

def predict_sentiment_texts(texts, bundle):
    """
    Pipeline d'inférence identique au training :
      TF-IDF(word+char) -> SelectKBest(chi2, k=50k) -> LogReg
    Renvoie: y_pred, proba, thr, debug_dict
    """
    vw = bundle["vectorizer_word"]
    vc = bundle["vectorizer_char"]
    sel = bundle["selector_chi2"]
    clf = bundle["model"]
    thr = float(bundle.get("threshold", 0.5))

    # 1) TF-IDF concat
    X = _stack_word_char(texts, vw, vc)                 # (n, 300000)
    # 2) Reduction chi²
    Xr = sel.transform(X)                                # (n, 50000)
    # 3) Proba + label
    proba = clf.predict_proba(Xr)[:, 1]
    yhat = (proba >= thr).astype(int)

    dbg = {
        "bundle_path": bundle.get("_path"),
        "tfidf_shape": (X.shape[0], X.shape[1]),
        "reduced_shape": (Xr.shape[0], Xr.shape[1]),
        "lr_coef_shape": getattr(getattr(clf, "coef_", None), "shape", None),
        "threshold": thr,
    }
    return yhat, proba, thr, dbg

# =========================
#   EMOTIONS — (OPTIONNEL)
# =========================
@st.cache_resource
def load_emotions_runtime():
    """
    Optionnel : si tu as un runtime émotions, charge-le ici et
    renvoie un callable `predict(texts, topk, thr)` qui renvoie
    (labels, topk_indices, topk_scores, binarized_per_text).

    Si rien n'est dispo, on renvoie None.
    """
    try:
        # Exemple : si tu as déjà un module interne de service
        # from serve.infer import load_emotions_runtime as _load
        # return _load()
        return None
    except Exception:
        return None

# =========================
#   UI
# =========================
placeholder = "Great battery life but the screen is dim.\nTerrible packaging, but excellent sound."
txt = st.text_area("Collez un ou plusieurs avis (1 par ligne) :", placeholder, height=180)

cols = st.columns([1, 1])
with cols[0]:
    st.subheader("Sentiment")
with cols[1]:
    st.subheader("Émotions")
    thr_override = st.slider("Seuil global (override)", 0.0, 0.9, 0.30, 0.01, help="Seuil affichage émotions (optionnel)")
    topk = st.selectbox("Top-k à afficher", [1, 2, 3, 5, 10], index=2)
    always_show = st.checkbox("Toujours afficher Top-k (même si aucune n’atteint le seuil)", value=True)

analyze = st.button("Analyser")

if analyze and txt.strip():
    texts = [t.strip() for t in txt.splitlines() if t.strip()]

    # ---------- SENTIMENT ----------
    with cols[0]:
        try:
            sb = load_sentiment_bundle()
            yhat, proba, thr, dbg = predict_sentiment_texts(texts, sb)
            for t, p, y in zip(texts, proba, yhat):
                lbl = "Positif" if y == 1 else "Négatif"
                st.write(f"- **{t[:80]}…** → {lbl}  (p={float(p):.3f}, thr={thr:.2f})")
        except Exception as e:
            st.error(f"Erreur sentiment : {e}")

    # ---------- EMOTIONS (OPTIONNEL) ----------
    with cols[1]:
        runtime = load_emotions_runtime()
        if runtime is None:
            st.info("Modèle émotions non trouvé (optionnel). Ajoute ton runtime plus tard.")
        else:
            try:
                # Exemple d’API attendue si tu branches ton runtime :
                # labels, top_idx, top_scores, binarized = runtime(texts, topk=topk, thr=thr_override)
                # for i, t in enumerate(texts):
                #     if always_show:
                #         tops = [f"{labels[j]} ({top_scores[i][k]:.2f})" for k, j in enumerate(top_idx[i])]
                #         st.write(f"- **{t[:80]}…** → " + (", ".join(tops) if tops else "aucune"))
                #     else:
                #         active = [labels[j] for j, v in enumerate(binarized[i]) if v == 1]
                #         st.write(f"- **{t[:80]}…** → " + (", ".join(active) if active else "aucune"))
                st.info("Brancher ici l’inférence émotions quand ton runtime sera prêt.")
            except Exception as e:
                st.error(f"Erreur émotions : {e}")

    # ---------- DEBUG ----------
    with st.expander("🔧 Infos chargement / Debug"):
        try:
            st.json(dbg)
        except Exception:
            st.write("Aucune info debug.")
