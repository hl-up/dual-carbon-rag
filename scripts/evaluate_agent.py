"""真实国家资料智能体试用题。自动检查与人工语义复核分别记录。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_rag.agent import PolicyAgent
from carbon_rag.config import Settings
from carbon_rag.ledger import now
from carbon_rag.service import RAGService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--output", type=Path, default=Path("docs/national-agent-results.json"))
    args = parser.parse_args()
    svc = RAGService(Settings.from_env())
    seeds = json.loads(Path("docs/verified_seed_files.json").read_text(encoding="utf-8"))
    ids = {r["record_id"]: r["doc_id"] for r in seeds}
    cases = json.loads(Path("docs/national_agent_cases.json").read_text(encoding="utf-8"))
    cases = [c for c in cases if not args.ids or c["id"] in args.ids][: args.limit]
    results = []
    for case in cases:
        result = PolicyAgent(svc).ask(case["question"], scope="national")
        expected = {(ids[e["record_id"]], e["page"]) for e in case["evidence"]}
        retrieved = {(c["doc_id"], c["page"]) for c in result.retrieved}
        cited_ids = {c["doc_id"] for c in result.citations}
        checks = {
            "status": result.status == case["expected_status"],
            "required_evidence_retrieved": expected <= retrieved,
            "required_documents_cited": {ids[r] for r in case.get("required_citation_records", [])}
            <= cited_ids,
            "keywords_all": all(k in result.text for k in case.get("keywords_all", [])),
            "keywords_any": not case.get("keywords_any")
            or any(k in result.text for k in case["keywords_any"]),
        }
        row = {
            "id": case["id"],
            "question": case["question"],
            "status": result.status,
            "answer": result.text,
            "citations": result.citations,
            "retrieved": result.retrieved,
            "steps": result.steps,
            "elapsed_seconds": result.elapsed,
            "checks": checks,
            "automatic_pass": all(checks.values()),
            "manual_citation_review": "待核查"
            if result.status == "answered"
            else "无需事实引用复核",
        }
        results.append(row)
        report = {
            "run_at": now(),
            "dataset": "two-verified-national-documents",
            "scope": "national",
            "chat_model": svc.settings.chat_model,
            "embedding_model": svc.settings.embedding_model,
            "embedding_dim": svc.settings.embedding_dim,
            "skipped_pages": {r["record_id"]: r.get("skipped_pages", []) for r in seeds},
            "index_manifest": [
                {"record_id": r["record_id"], "doc_id": r["doc_id"], "sha256": r["sha256"]}
                for r in seeds
            ],
            "results": results,
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{case['id']}: {'PASS' if row['automatic_pass'] else 'REVIEW'} ({result.status}) {result.elapsed:.1f}s; tools={len(result.steps)}",
            flush=True,
        )
    return 0 if all(r["automatic_pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
