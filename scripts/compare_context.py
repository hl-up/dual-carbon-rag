"""Compare real LLM usage with and without extractive context compression."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_rag.config import Settings
from carbon_rag.service import RAGService


def run_case(settings, question, budget):
    service = RAGService(replace(settings, max_context_chars=budget))
    with service.ledger.connection() as db:
        before = db.execute("SELECT COALESCE(MAX(rowid),0) FROM operations").fetchone()[0]
    answer = service.ask(question, scope="national")
    with service.ledger.connection() as db:
        row = db.execute(
            "SELECT elapsed,details FROM operations WHERE rowid>? AND action='chat_usage' "
            "ORDER BY rowid DESC LIMIT 1",
            (before,),
        ).fetchone()
    usage = json.loads(row[1]) if row else {}
    return {
        "budget_chars": budget,
        "status": answer.status,
        "citation_pages": [(c["doc_id"], c["page"]) for c in answer.citations],
        "llm_seconds": row[0] if row else None,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "context_chars_before": usage.get("context_chars_before"),
        "context_chars_after": usage.get("context_chars_after"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="国家应对气候变化标准体系包含哪些一级子体系？")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/context-compare.json"))
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("budget 必须大于零")
    settings = Settings.from_env()
    settings.require_api(chat=True)
    report = {
        "question": args.question,
        "cases": [run_case(settings, args.question, budget) for budget in (0, args.budget)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
