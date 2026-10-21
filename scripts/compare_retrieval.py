"""Compare vector-only and CrossEncoder top-k on the same labelled questions."""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_rag.config import Settings
from carbon_rag.retrieval import CrossEncoderReranker, compress_evidence
from carbon_rag.service import RAGService


def matches(found, expected):
    actual = {(doc.metadata["doc_id"], int(doc.metadata["page"])) for doc, _ in found}
    return len(actual & expected) / len(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--rerank-model", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/retrieval-compare.json")
    )
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    labelled = [c for c in cases if c.get("question", "").strip() and c.get("expected_evidence")]
    if not labelled:
        parser.error("需要至少一道已标注 question 和 expected_evidence；没有发起 API 请求")
    settings = Settings.from_env()
    settings.require_api()
    service = RAGService(settings)
    ranker = CrossEncoderReranker(args.rerank_model)
    results = []
    for case in labelled:
        rows, stopped = service.scope_documents(
            case["question"], scope=case.get("scope", "all"), province=case.get("province", "")
        )
        if stopped:
            results.append({"id": case["id"], "status": stopped.status})
            continue
        expected = {(e["doc_id"], int(e["page"])) for e in case["expected_evidence"]}
        ids = [row["doc_id"] for row in rows]
        started = perf_counter()
        baseline = service.index.search(case["question"], ids, settings.top_k)
        baseline_ms = round((perf_counter() - started) * 1000, 2)
        started = perf_counter()
        candidates = service.index.search(case["question"], ids, settings.rerank_candidates)
        reranked = ranker.rerank(case["question"], candidates)[: settings.top_k]
        rerank_ms = round((perf_counter() - started) * 1000, 2)
        evidence = [
            {"number": i + 1, "excerpt": doc.page_content} for i, (doc, _) in enumerate(reranked)
        ]
        compressed = compress_evidence(case["question"], evidence, settings.max_context_chars)
        results.append(
            {
                "id": case["id"],
                "baseline_recall": matches(baseline, expected),
                "rerank_recall": matches(reranked, expected),
                "baseline_ms": baseline_ms,
                "rerank_ms": rerank_ms,
                "context_chars_before": sum(len(x["excerpt"]) for x in evidence),
                "context_chars_after": sum(len(x["excerpt"]) for x in compressed),
            }
        )
    valid = [r for r in results if "baseline_recall" in r]
    report = {
        "model": args.rerank_model,
        "top_k": settings.top_k,
        "rerank_candidates": settings.rerank_candidates,
        "labelled_count": len(valid),
        "baseline_mean_recall": mean(r["baseline_recall"] for r in valid) if valid else None,
        "rerank_mean_recall": mean(r["rerank_recall"] for r in valid) if valid else None,
        "baseline_mean_ms": mean(r["baseline_ms"] for r in valid) if valid else None,
        "rerank_mean_ms": mean(r["rerank_ms"] for r in valid) if valid else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
