"""Chroma repository and embedding adapter."""

import math
import re
from collections import Counter
from threading import Lock

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .retrieval import _terms


class CheckedEmbeddings(Embeddings):
    def __init__(self, delegate, dimension):
        self.delegate, self.dimension = delegate, dimension

    def _check(self, vectors):
        if any(len(v) != self.dimension for v in vectors):
            raise ValueError("API 返回的嵌入维度与配置不一致")
        return vectors

    def embed_documents(self, texts):
        vectors = self.delegate.embed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError("API 返回的嵌入数量不一致")
        return self._check(vectors)

    def embed_query(self, text):
        return self._check([self.delegate.embed_query(text)])[0]


class VectorIndex:
    def __init__(self, settings):
        self.settings = settings
        self._store = None
        self._init_lock = Lock()

    def store(self):
        if self._store is None:
            with self._init_lock:
                if self._store is None:
                    s = self.settings
                    embedding = OpenAIEmbeddings(
                        api_key=s.api_key or "not-configured",
                        base_url=s.base_url or None,
                        model=s.embedding_model,
                        dimensions=s.embedding_dim,
                        check_embedding_ctx_length=False,
                        request_timeout=s.timeout,
                        max_retries=s.max_retries,
                    )
                    self._store = Chroma(
                        collection_name="carbon_policy_v1",
                        embedding_function=CheckedEmbeddings(embedding, s.embedding_dim),
                        persist_directory=str(s.data_dir / "chroma"),
                        collection_metadata={"hnsw:space": "cosine"},
                    )
        return self._store

    def add(self, chunks):
        self.settings.require_api()
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            self.store().add_documents(batch, ids=[c.id for c in batch])

    def delete(self, doc_id):
        self.store().delete(where={"doc_id": doc_id})

    def count(self, doc_id):
        return len(self.store().get(where={"doc_id": doc_id})["ids"])

    def search(self, query, ids, k):
        self.settings.require_api()
        if not ids:
            return []
        return self.store().similarity_search_with_score(
            query, k=k, filter={"doc_id": {"$in": ids}}
        )

    def keyword_search(self, query, ids, k=2):
        """Find literal term overlap in the allowed local chunks without an API call."""
        if not ids:
            return []
        rows = self.store().get(where={"doc_id": {"$in": ids}}, include=["documents", "metadatas"])
        texts = rows["documents"]
        if not texts:
            return []
        terms = [_terms(text) for text in texts]
        frequencies = Counter(term for group in terms for term in group)

        def score(group, query_terms):
            return sum(
                math.log((len(texts) + 1) / (frequencies[term] + 1)) + 1
                for term in query_terms & group
            )

        query_terms = _terms(query)
        ranked = sorted(range(len(texts)), key=lambda i: -score(terms[i], query_terms))
        chosen = ranked[:1]
        list_target = re.search(r"(?:哪些|哪几种|哪几类)([\u4e00-\u9fff]{2,12})[？?]?$", query)
        if list_target:
            focus_terms = _terms(f"{list_target.group(1)}主要包括")
            focused = max(range(len(texts)), key=lambda i: score(terms[i], focus_terms))
            if score(terms[focused], focus_terms) > 0 and focused not in chosen:
                chosen.append(focused)
        selected = set(chosen)
        chosen.extend(i for i in ranked if i not in selected)
        return [
            Document(page_content=texts[i], metadata=rows["metadatas"][i], id=rows["ids"][i])
            for i in chosen[:k]
            if score(terms[i], query_terms) > 0
        ]
