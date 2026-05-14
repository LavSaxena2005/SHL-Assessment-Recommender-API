"""
Hybrid Retrieval System: BM25 + Semantic (FAISS or TF-IDF fallback)
Implements fusion of keyword and semantic retrieval for SHL catalog.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).parent.parent / "catalog" / "catalog.json"
INDEX_DIR = Path(__file__).parent.parent / "catalog"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
BM25_PATH = INDEX_DIR / "bm25.pkl"
TFIDF_PATH = INDEX_DIR / "tfidf.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10


def _try_load_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        logger.warning(f"sentence-transformers unavailable: {e}")
        return None


class TFIDFSemanticEncoder:
    """TF-IDF based semantic encoder used as offline fallback."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=8000,
            sublinear_tf=True, stop_words="english",
        )
        self._fitted = False

    def fit(self, texts: list):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            mat = mat / norms
        return mat


class HybridRetriever:
    """
    Hybrid retriever combining BM25 (keyword) + semantic search.
    Scores fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self.assessments: list = []
        self.texts: list = []
        self.bm25 = None
        self.faiss_index = None
        self.embeddings: Optional[np.ndarray] = None
        self.model = None
        self.use_faiss = False
        self._loaded = False

    def _load_catalog(self):
        with open(CATALOG_PATH, encoding="utf-8") as f:
            return json.load(f)

    def _build_texts(self, assessments: list) -> list:
        texts = []
        for a in assessments:
            skills = ", ".join(a.get("skills", []))
            dur = f"{a.get('duration', '')} minutes" if a.get("duration") else ""
            texts.append(f"{a.get('name','')} {a.get('category','')} {a.get('description','')} {skills} {dur}".strip())
        return texts

    def _init_model(self, texts: list):
        """Try sentence-transformers, fall back to TF-IDF."""
        self.model = _try_load_sentence_transformers()
        if self.model is None:
            logger.info("Using TF-IDF semantic encoder (offline fallback)")
            enc = TFIDFSemanticEncoder()
            enc.fit(texts)
            self.model = enc
            self.use_faiss = False
        else:
            self.use_faiss = True

    def build_index(self):
        from rank_bm25 import BM25Okapi
        logger.info("Building retrieval indexes...")
        self.assessments = self._load_catalog()
        self.texts = self._build_texts(self.assessments)

        # BM25
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])
        with open(BM25_PATH, "wb") as f:
            pickle.dump(self.bm25, f)

        # Semantic
        self._init_model(self.texts)
        embeddings = self.model.encode(self.texts, normalize_embeddings=True).astype(np.float32)
        self.embeddings = embeddings
        np.save(EMBEDDINGS_PATH, embeddings)

        if isinstance(self.model, TFIDFSemanticEncoder):
            with open(TFIDF_PATH, "wb") as f:
                pickle.dump(self.model, f)

        if self.use_faiss:
            import faiss
            idx = faiss.IndexFlatIP(embeddings.shape[1])
            emb = embeddings.copy()
            faiss.normalize_L2(emb)
            idx.add(emb)
            self.faiss_index = idx
            faiss.write_index(idx, str(FAISS_INDEX_PATH))

        self._loaded = True
        logger.info(f"Indexes built: {len(self.assessments)} assessments")

    def load_index(self):
        if self._loaded:
            return
        self.assessments = self._load_catalog()
        self.texts = self._build_texts(self.assessments)

        if BM25_PATH.exists() and EMBEDDINGS_PATH.exists():
            logger.info("Loading cached indexes...")
            with open(BM25_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
            self.embeddings = np.load(EMBEDDINGS_PATH)

            # Try FAISS + sentence-transformers
            if FAISS_INDEX_PATH.exists():
                try:
                    import faiss
                    self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
                    m = _try_load_sentence_transformers()
                    if m:
                        self.model = m
                        self.use_faiss = True
                except Exception:
                    pass

            # TF-IDF fallback
            if not self.use_faiss:
                if TFIDF_PATH.exists():
                    with open(TFIDF_PATH, "rb") as f:
                        self.model = pickle.load(f)
                else:
                    enc = TFIDFSemanticEncoder()
                    enc.fit(self.texts)
                    self.model = enc
                    with open(TFIDF_PATH, "wb") as f:
                        pickle.dump(enc, f)
                self.use_faiss = False

            self._loaded = True
        else:
            self.build_index()

    def _bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(query.lower().split()).astype(np.float32)

    def _semantic_scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        if self.use_faiss and self.faiss_index:
            scores = np.zeros(len(self.assessments), dtype=np.float32)
            dists, idxs = self.faiss_index.search(q, len(self.assessments))
            for d, i in zip(dists[0], idxs[0]):
                if 0 <= i < len(scores):
                    scores[i] = d
            return scores
        else:
            return (self.embeddings @ q.T).squeeze().astype(np.float32)

    def _rrf(self, s1: np.ndarray, s2: np.ndarray, k=60, w1=0.4, w2=0.6) -> np.ndarray:
        n = len(s1)
        r1 = np.zeros(n); r2 = np.zeros(n)
        for rank, i in enumerate(np.argsort(-s1)):
            r1[i] = 1.0 / (k + rank + 1)
        for rank, i in enumerate(np.argsort(-s2)):
            r2[i] = 1.0 / (k + rank + 1)
        return w1 * r1 + w2 * r2

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        if not self._loaded:
            self.load_index()
        bm25 = self._bm25_scores(query)
        sem = self._semantic_scores(query)
        fused = self._rrf(bm25, sem)
        results = []
        for i in np.argsort(-fused)[:top_k]:
            if fused[i] > 0:
                a = dict(self.assessments[i])
                a["_score"] = float(fused[i])
                results.append(a)
        return results

    def get_all(self) -> list:
        if not self.assessments:
            self.assessments = self._load_catalog()
        return self.assessments


_retriever: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        _retriever.load_index()
    return _retriever
