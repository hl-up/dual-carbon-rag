"""Run the fixed page-labelled policy evaluation set."""

import argparse
import json
import os
import statistics
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_rag.config import Settings
from carbon_rag.ledger import now


def load_cases(path):
    cases = [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    ids, questions = set(), set()
    for case in cases:
        case_id, question = case.get("id"), case.get("question", "").strip()
        if not case_id or case_id in ids or not question or question in questions:
            raise ValueError(f"题目 ID 或问题缺失、重复：{case_id}")
        ids.add(case_id)
        questions.add(question)
        if case.get("expected_status") not in {"answered", "insufficient", "out_of_scope"}:
            raise ValueError(f"{case_id}: expected_status 无效")
        if case.get("scope", "all") not in {"all", "national", "province"}:
            raise ValueError(f"{case_id}: scope 无效")
        if case.get("scope") == "province" and not case.get("province"):
            raise ValueError(f"{case_id}: 省级题缺少 province")
        evidence = case.get("expected_evidence")
        if not isinstance(evidence, list) or (
            case["expected_status"] == "answered" and not evidence
        ):
            raise ValueError(f"{case_id}: 答案题必须标注原文页")
        if case["expected_status"] == "answered" and not case.get("reference_answer", "").strip():
            raise ValueError(f"{case_id}: 答案题必须填写参考答案")
        for item in evidence:
            if (
                not isinstance(item.get("doc_id"), str)
                or not isinstance(item.get("page"), int)
                or item["page"] < 1
            ):
                raise ValueError(f"{case_id}: 原文标注无效")
    if not cases:
        raise ValueError("评测集为空")
    return cases


def check_corpus(service, cases):
    """Stop before paid calls if a labelled page or original is unavailable."""
    eligible = {r["doc_id"]: r for r in service.ledger.eligible()}
    for case in cases:
        for item in case["expected_evidence"]:
            row = eligible.get(item["doc_id"])
            if row is None or item["page"] > row["page_count"]:
                raise ValueError(f"{case['id']}: 已核验文件或 PDF 页不存在：{item}")
            if not (service.raw_dir / f"{item['doc_id']}.pdf").is_file():
                raise ValueError(f"{case['id']}: PDF 原件缺失：{item['doc_id']}")


def evaluate(service, cases, judge=None, mode="direct"):
    if mode not in {"direct", "agent"}:
        raise ValueError("mode 只能是 direct 或 agent")
    if mode == "agent":
        from carbon_rag.agent import PolicyAgent

        worker = PolicyAgent(service)
    else:
        worker = service
    results = []
    for case in cases:
        if not case.get("question", "").strip():
            results.append({"id": case["id"], "status": "not_labelled"})
            continue
        result = worker.ask(
            case["question"], scope=case.get("scope", "all"), province=case.get("province", "")
        )
        expected = {(e["doc_id"], e["page"]) for e in case.get("expected_evidence", [])}
        retrieved = {(c["doc_id"], c["page"]) for c in result.retrieved}
        cited = {(c["doc_id"], c["page"]) for c in result.citations}
        judgment = (
            judge(case["question"], result.text, result.citations, case.get("reference_answer", ""))
            if judge and result.status == "answered"
            else {"grounded": None, "complete": None}
        )
        results.append(
            {
                "id": case["id"],
                "category": case.get("category", "未分类"),
                "question": case["question"],
                "scope": case.get("scope", "all"),
                "province": case.get("province", ""),
                "expected_status": case.get("expected_status"),
                "expected_evidence": case.get("expected_evidence", []),
                "reference_answer": case.get("reference_answer", ""),
                "status": result.status,
                "answer": result.text,
                "retrieved": result.retrieved,
                "citations": result.citations,
                "steps": result.steps,
                "retrieval_hit": bool(expected & retrieved) if expected else None,
                "retrieval_complete": expected <= retrieved if expected else None,
                "citation_complete": expected <= cited and result.status == "answered"
                if expected
                else None,
                "document_citation_complete": (
                    {doc_id for doc_id, _ in expected} <= {doc_id for doc_id, _ in cited}
                    and result.status == "answered"
                )
                if expected
                else None,
                "status_matches": result.status == case.get("expected_status"),
                "refusal_matches": (result.status == "insufficient")
                == (
                    case.get("expected_status") == "insufficient"
                    or case.get("should_refuse", False)
                ),
                "judge_grounded": judgment["grounded"],
                "judge_complete": judgment["complete"],
                "elapsed_seconds": result.elapsed,
                "manual_citation_review": "待人工核查" if result.status == "answered" else "不适用",
                "manual_region_review": "待人工核查" if result.status == "answered" else "不适用",
            }
        )
    return results


def summarize(results):
    rows = [r for r in results if r["status"] != "not_labelled"]
    errors = [r for r in rows if r["status"] == "error"]
    answerable = [r for r in rows if r.get("expected_status") == "answered"]
    grounded = [r for r in rows if r.get("judge_grounded") is not None]
    complete = [r for r in answerable if r.get("judge_complete") is not None]
    times = sorted(r["elapsed_seconds"] * 1000 for r in rows)

    def rate(items, key):
        return round(sum(r[key] is True for r in items) / len(items), 4) if items else None

    by_category = {}
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        by_category[category] = {
            "count": len(group),
            "errors": sum(r["status"] == "error" for r in group),
            "status_accuracy": None
            if any(r["status"] == "error" for r in group)
            else rate(group, "status_matches"),
        }
    return {
        "total": len(rows),
        "answerable": len(answerable),
        "run_complete": not errors,
        "error_count": len(errors),
        "status_accuracy": None if errors else rate(rows, "status_matches"),
        "retrieval_full_recall": None if errors else rate(answerable, "retrieval_complete"),
        "expected_page_citation_rate": None if errors else rate(answerable, "citation_complete"),
        "expected_document_citation_rate": None
        if errors
        else rate(answerable, "document_citation_complete"),
        "judge_coverage": len(grounded),
        "judge_grounded_rate": rate(grounded, "judge_grounded"),
        "judge_complete_coverage": len(complete),
        "judge_complete_rate": rate(complete, "judge_complete"),
        "latency_p50_ms": round(statistics.median(times)) if times and not errors else None,
        "latency_p95_ms": round(times[ceil(len(times) * 0.95) - 1])
        if times and not errors
        else None,
        "by_category": by_category,
    }


def make_judge(settings):
    """Optional same-provider second opinion; parse failures remain unknown."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from carbon_rag.providers import make_chat_model

    model = make_chat_model(settings, json_output=True)
    system = SystemMessage(
        content=(
            "你是政策问答评判员。只依据被引用原文判断回答中每个事实是否得到支持，"
            "并与参考答案比较是否覆盖其全部关键事实（允许同义改写）。"
            "参考答案不是证据；不得用其替代被引用原文判断事实支持。"
            "若参考答案为空，complete 返回 null。只返回 JSON："
            '{"grounded":true/false,"complete":true/false/null}。'
        )
    )

    def judge(question, answer, citations, reference_answer):
        payload = {
            "question": question,
            "answer": answer,
            "reference_answer": reference_answer,
            "cited_evidence": [
                {key: item[key] for key in ("title", "page", "excerpt") if key in item}
                for item in citations
            ],
        }
        try:
            response = model.invoke(
                [system, HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
            )
            values = json.loads(response.content)
            return {
                key: values.get(key) if isinstance(values.get(key), bool) else None
                for key in ("grounded", "complete")
            }
        except Exception:
            return {"grounded": None, "complete": None}

    return judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, nargs="?", default=Path("evaluation/cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/results.json"))
    parser.add_argument("--check", action="store_true", help="只检查题集格式")
    parser.add_argument("--check-corpus", action="store_true", help="检查已核验原件，不调用 API")
    parser.add_argument("--judge", action="store_true", help="对回答再调用一次模型评判")
    parser.add_argument("--trace", action="store_true", help="允许 LangSmith 追踪评测问题与证据")
    parser.add_argument("--mode", choices=("direct", "agent"), default="direct")
    parser.add_argument("--ids", nargs="+", help="只运行指定题目 ID，便于诊断")
    parser.add_argument("--judge-existing", type=Path, help="评判已有报告，不重新检索或生成答案")
    args = parser.parse_args()
    if not args.trace:
        os.environ["LANGSMITH_TRACING"] = "false"
    if args.judge_existing:
        settings = Settings.from_env()
        settings.require_api(chat=True)
        report = json.loads(args.judge_existing.read_text(encoding="utf-8"))
        judge = make_judge(settings)
        for row in report["results"]:
            if row["status"] == "answered" and (
                row.get("judge_grounded") is None
                or (row.get("reference_answer") and row.get("judge_complete") is None)
            ):
                judgment = judge(
                    row["question"],
                    row["answer"],
                    row["citations"],
                    row.get("reference_answer", ""),
                )
                row["judge_grounded"] = judgment["grounded"]
                row["judge_complete"] = judgment["complete"]
        report["metrics"] = summarize(report["results"])
        args.judge_existing.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
        return 0
    cases = load_cases(args.cases)
    if args.ids:
        requested = set(args.ids)
        cases = [case for case in cases if case["id"] in requested]
        if {case["id"] for case in cases} != requested:
            raise ValueError("--ids 包含题集中不存在的 ID")
    if args.check:
        print(f"题集格式通过：{len(cases)} 题")
        return 0
    from carbon_rag.service import RAGService

    settings = Settings.from_env()
    service = RAGService(settings)
    check_corpus(service, cases)
    if args.check_corpus:
        print(f"题集原件检查通过：{len(cases)} 题")
        return 0
    settings.require_api(chat=True)
    results = evaluate(service, cases, make_judge(settings) if args.judge else None, args.mode)
    report = {
        "run_at": now(),
        "dataset": str(args.cases),
        "mode": args.mode,
        "selected_ids": args.ids,
        "chat_model": settings.chat_model,
        "chat_base_url": settings.chat_base_url or settings.base_url,
        "embedding_model": settings.embedding_model,
        "embedding_base_url": settings.base_url,
        "embedding_dim": settings.embedding_dim,
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "index_manifest": [
            {"doc_id": r["doc_id"], "sha256": r["sha256"], "ingested_at": r["ingested_at"]}
            for r in service.ledger.eligible()
        ],
        "metrics": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"逐题结果：{args.output}。事实支持与地区适用性仍需人工核查。")
    return 0 if report["metrics"]["run_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
