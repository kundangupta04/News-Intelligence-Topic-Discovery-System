"""All the machine-learning code of the project """
import io
import os
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = os.path.join("data", "bbc_news.csv")
STOP_WORDS = set(ENGLISH_STOP_WORDS) | {"said", "says", "say", "mr", "mrs", "ms", "told",
                                        "bbc", "don", "didn", "doesn", "isn", "wasn"}


# ---------------------------------------------------------------- 1. data
def load_data(path=DATA_FILE):
    """Read the CSV, remove empty rows and duplicates, split title and text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python download_data.py")
    raw = open(path, "rb").read().decode("utf-8", errors="replace")  # file mixes encodings
    df = pd.read_csv(io.StringIO(raw))
    if "news" not in df.columns:
        raise ValueError(f"Column 'news' not found. Your columns are: {list(df.columns)}")
    n_raw = len(df)
    df = df.dropna(subset=["news"]).drop_duplicates(subset=["news"]).reset_index(drop=True)
    parts = df["news"].str.strip().str.split("\n", n=1, expand=True)
    df["title"] = parts[0].str.strip()
    df["text"] = parts[1].fillna("").str.strip()
    if "type" in df.columns:
        df = df.rename(columns={"type": "category"})   # only used for evaluation
    df.attrs["n_raw"] = n_raw
    return df


# ---------------------------------------------------------- 2. preprocessing
def clean_text(text):
    """lowercase -> remove HTML and punctuation -> remove stop words and numbers."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split()
             if len(w) > 1 and not w.isdigit() and w not in STOP_WORDS]
    return " ".join(words)


# --------------------------------------------------- 3. topic names (labels)
def make_label(words):
    """Join the first 3 keywords that do not look alike, e.g. 'Film / Music / Star'."""
    chosen = []
    for w in words:
        if not any(w[:4] == c[:4] for c in chosen):
            chosen.append(w)
        if len(chosen) == 3:
            break
    return " / ".join(chosen).title()


# ------------------------------------------------------------ 4. the model
def build_model(df, k=5):
    """TF-IDF -> K-Means -> keywords -> PCA -> metrics. Returns a dictionary."""
    if k < 2 or k >= len(df):
        raise ValueError(f"k must be between 2 and {len(df) - 1}, got {k}.")
    df = df.copy()
    df["clean"] = (df["title"] + " " + df["text"]).map(clean_text)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    vec = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85,
                          ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(df["clean"])                      # fit ONCE
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    df["cluster"] = km.labels_

    terms = np.array(vec.get_feature_names_out())
    overall = np.asarray(X.mean(axis=0)).ravel()
    keywords, labels = {}, {}
    for c in range(k):
        centre = np.asarray(X[(km.labels_ == c)].mean(axis=0)).ravel()
        top = (centre - overall).argsort()[::-1][:8]        # words that stand out in cluster c
        keywords[c] = list(terms[top])
        labels[c] = make_label(keywords[c])

    pca = PCA(n_components=2, random_state=42)              # only for drawing the map
    coords = pca.fit_transform(X.toarray())
    df["x"], df["y"] = coords[:, 0], coords[:, 1]

    result = dict(df=df, vec=vec, km=km, X=X, keywords=keywords, labels=labels,
                  silhouette=float(silhouette_score(X, km.labels_, metric="cosine")),
                  variance=pca.explained_variance_ratio_, ari=None, nmi=None)
    if "category" in df.columns:                            # labels used ONLY to check results
        result["ari"] = float(adjusted_rand_score(df["category"], km.labels_))
        result["nmi"] = float(normalized_mutual_info_score(df["category"], km.labels_))
    return result


def compare_k(X, ks=(5, 6, 8, 10)):
    """Silhouette score for several cluster counts."""
    rows = []
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        rows.append({"k": k, "silhouette": round(float(silhouette_score(X, labels, metric="cosine")), 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------ 5. similar articles
def find_similar(model, text=None, index=None, n=5):
    """Top-n similar articles to an existing article (index) or to new text."""
    if index is not None:
        query = model["X"][index]
    else:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Please enter some text.")
        query = model["vec"].transform([cleaned])           # transform only, never re-fit
        if query.nnz == 0:
            raise ValueError("None of these words are known to the model. Try a longer news text.")
    scores = cosine_similarity(query, model["X"]).ravel()
    if index is not None:
        scores[index] = -1                                  # exclude the article itself
    best = scores.argsort()[::-1][:n]
    out = model["df"].iloc[best][["title", "text", "cluster"]].copy()
    out["similarity"] = scores[best].round(3)
    return out