from pathlib import Path

import pytest

from scripts.evaluate import check_corpus, evaluate, load_cases, summarize
from scripts.project_evaluation import project_report


def test_unlabelled_cases_do_not_call_models(svc):
    assert evaluate(svc, [{"id": "Q01", "question": ""}]) == [
        {"id": "Q01", "status": "not_labelled"}
    ]
    assert not svc.chat.calls


def test_retrieval_hit_uses_candidates_not_only_final_references(svc, metadata, pdf_bytes):
    row_a = svc.ingest(pdf_bytes("first"), "a.pdf", metadata(title="a"))["document"]
    row_b = svc.ingest(pdf_bytes("second"), "b.pdf", metadata(title="b"))["document"]
    result = svc.ask("test")
    assert len(result.retrieved) == 2 and len(result.citations) == 1
    unquoted = next(c for c in result.retrieved if c["doc_id"] != result.citations[0]["doc_id"])
    results = evaluate(
        svc,
        [
            {
                "id": "Q01",
                "question": "test",
                "expected_evidence": [
                    {"doc_id": unquoted["doc_id"], "page": 1},
                ],
            }
        ],
    )
    assert results[0]["retrieval_hit"]
    assert {c["doc_id"] for c in results[0]["retrieved"]} == {row_a["doc_id"], row_b["doc_id"]}


def test_public_evaluation_example_is_fully_labelled():
    cases = load_cases(Path("evaluation/cases.example.jsonl"))
    assert len(cases) == 2
    assert {c["expected_status"] for c in cases} == {"answered", "insufficient"}
    assert all(c.get("reference_answer") for c in cases if c["expected_status"] == "answered")


def test_invalid_answerable_case_fails_preflight(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"Q1","question":"政策是什么？","expected_status":"answered","expected_evidence":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="原文页"):
        load_cases(path)


def test_corpus_check_rejects_missing_document_before_api(svc):
    with pytest.raises(ValueError, match="已核验文件"):
        check_corpus(svc, [{"id": "Q1", "expected_evidence": [{"doc_id": "missing", "page": 1}]}])
    assert not svc.chat.calls


def test_summary_uses_full_page_recall_and_reports_unknown_judge(svc, metadata, pdf_bytes):
    row = svc.ingest(pdf_bytes("first"), "a.pdf", metadata(title="a"))["document"]
    results = evaluate(
        svc,
        [
            {
                "id": "Q1",
                "category": "测试",
                "question": "first",
                "expected_status": "answered",
                "expected_evidence": [{"doc_id": row["doc_id"], "page": 1}],
            }
        ],
    )
    metrics = summarize(results)
    assert metrics["total"] == 1
    assert metrics["retrieval_full_recall"] == 1
    assert metrics["expected_page_citation_rate"] == 1
    assert metrics["expected_document_citation_rate"] == 1
    assert metrics["judge_coverage"] == 0
    assert metrics["judge_grounded_rate"] is None
    assert metrics["judge_complete_coverage"] == 0
    assert metrics["judge_complete_rate"] is None


def test_judge_tracks_support_and_completeness_separately(svc, metadata, pdf_bytes):
    row = svc.ingest(pdf_bytes("first"), "a.pdf", metadata(title="a"))["document"]
    calls = []

    def judge(question, answer, citations, reference_answer):
        calls.append((question, answer, citations, reference_answer))
        return {"grounded": True, "complete": False}

    results = evaluate(
        svc,
        [
            {
                "id": "Q1",
                "category": "测试",
                "question": "first",
                "expected_status": "answered",
                "expected_evidence": [{"doc_id": row["doc_id"], "page": 1}],
                "reference_answer": "first 和 second",
            }
        ],
        judge=judge,
    )
    assert calls[0][3] == "first 和 second"
    assert results[0]["judge_grounded"] is True
    assert results[0]["judge_complete"] is False
    metrics = summarize(results)
    assert metrics["judge_grounded_rate"] == 1
    assert metrics["judge_complete_rate"] == 0


def test_error_run_does_not_publish_quality_rates():
    metrics = summarize(
        [
            {
                "id": "Q1",
                "category": "测试",
                "status": "error",
                "expected_status": "answered",
                "status_matches": False,
                "retrieval_complete": False,
                "citation_complete": False,
                "document_citation_complete": False,
                "judge_grounded": None,
                "elapsed_seconds": 0.1,
            }
        ]
    )
    assert metrics["error_count"] == 1
    assert not metrics["run_complete"]
    assert metrics["status_accuracy"] is None
    assert metrics["latency_p95_ms"] is None


def test_single_case_projection_recomputes_metrics_without_changing_source():
    rows = []
    for number in range(30):
        is_r02 = number == 29
        rows.append(
            {
                "id": "R02" if is_r02 else f"Q{number:02d}",
                "category": "测试",
                "question": "施行日期？" if is_r02 else f"问题 {number}",
                "scope": "national",
                "province": "",
                "expected_status": "insufficient" if is_r02 else "answered",
                "expected_evidence": [],
                "reference_answer": "",
                "status": "review" if is_r02 else "answered",
                "status_matches": not is_r02,
                "retrieval_complete": None,
                "citation_complete": None,
                "document_citation_complete": None,
                "judge_grounded": None,
                "judge_complete": None,
                "elapsed_seconds": 1.0,
            }
        )
    base = {
        "run_at": "2026-10-20",
        "selected_ids": None,
        "results": rows,
        "metrics": summarize(rows),
        "index_manifest": [{"doc_id": "same"}],
    }
    updated = {**rows[-1], "status": "insufficient", "status_matches": True}
    retest = {
        **base,
        "run_at": "2026-10-21",
        "selected_ids": ["R02"],
        "results": [updated],
        "metrics": summarize([updated]),
    }
    projected = project_report(base, retest, "base.json", "retest.json")
    assert projected["metrics"]["status_accuracy"] == 1
    assert projected["projection"]["full_rerun"] is False
    assert projected["results"][-1]["status"] == "insufficient"
    assert base["results"][-1]["status"] == "review"

    retest["index_manifest"] = [{"doc_id": "changed"}]
    with pytest.raises(ValueError, match="版本不一致"):
        project_report(base, retest, "base.json", "retest.json")
