"""Optional reranking and extractive context compression."""

import re
from threading import Lock


def _terms(text):
    words = set(re.findall(r"[a-z0-9]+", text.casefold()))
    han = re.findall(r"[\u4e00-\u9fff]", text)
    return words | {"".join(han[i : i + 2]) for i in range(len(han) - 1)}


def compress_evidence(question, evidence, max_chars):
    """Keep high-overlap sentences; retain evidence numbers and source text only."""
    if not max_chars or sum(len(e["excerpt"]) for e in evidence) <= max_chars:
        return evidence
    terms = _terms(question)
    slots = max(1, max_chars // max(1, len(evidence)))
    compressed = []
    for item in evidence:
        text = item["excerpt"]
        if len(text) <= slots:
            compressed.append(item)
            continue
        parts = [p for p in re.split(r"(?<=[。；！？.!?])|\n", text) if p.strip()]
        ranked = sorted(
            enumerate(parts),
            key=lambda pair: (-len(terms & _terms(pair[1])), pair[0]),
        )
        selected, used = [], 0
        for index, part in ranked:
            if used + len(part) <= slots:
                selected.append((index, part))
                used += len(part)
        excerpt = "".join(part for _, part in sorted(selected)) or text[:slots]
        compressed.append({**item, "excerpt": excerpt[:slots]})
    return compressed


class CrossEncoderReranker:
    """Load a local sentence-transformers CrossEncoder only when configured."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None
        self._lock = Lock()

    def rerank(self, query, found):
        if not found:
            return []
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, doc.page_content) for doc, _ in found])
        return [item for _, item in sorted(zip(scores, found), key=lambda pair: -float(pair[0]))]
