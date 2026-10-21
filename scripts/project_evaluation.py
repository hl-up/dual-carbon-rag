"""Recompute 30-question metrics after replacing one case with a verified retest."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_rag.ledger import now
from scripts.evaluate import summarize

CONFIG_FIELDS = (
    "dataset",
    "mode",
    "chat_model",
    "chat_base_url",
    "embedding_model",
    "embedding_base_url",
    "embedding_dim",
    "top_k",
    "chunk_size",
    "chunk_overlap",
    "index_manifest",
)
CASE_FIELDS = (
    "id",
    "category",
    "question",
    "scope",
    "province",
    "expected_status",
    "expected_evidence",
    "reference_answer",
)


def project_report(base, retest, base_path, retest_path):
    if base.get("projection") or base.get("selected_ids") is not None:
        raise ValueError("基线必须是未拼接的完整运行报告")
    if len(base.get("results", [])) != 30 or len(retest.get("results", [])) != 1:
        raise ValueError("需要完整 30 题基线和单题复测报告")
    if not base.get("metrics", {}).get("run_complete") or not retest.get("metrics", {}).get(
        "run_complete"
    ):
        raise ValueError("不能拼接含接口错误的报告")
    if any(base.get(key) != retest.get(key) for key in CONFIG_FIELDS):
        raise ValueError("题集、模型、检索配置或索引文件版本不一致")
    rows = deepcopy(base["results"])
    ids = [row["id"] for row in rows]
    if len(set(ids)) != 30:
        raise ValueError("基线题目 ID 缺失或重复")
    update = retest["results"][0]
    if retest.get("selected_ids") != [update["id"]] or update["id"] not in ids:
        raise ValueError("复测题目 ID 与基线不匹配")
    position = ids.index(update["id"])
    if any(rows[position].get(key) != update.get(key) for key in CASE_FIELDS):
        raise ValueError("复测题目或标注与基线不一致")
    rows[position] = deepcopy(update)
    report = {
        key: deepcopy(value) for key, value in base.items() if key not in {"metrics", "results"}
    }
    report["results"] = rows
    report["metrics"] = summarize(rows)
    report["projection"] = {
        "method": "replace_one_retested_case",
        "full_rerun": False,
        "projected_at": now(),
        "base_report": str(base_path),
        "base_run_at": base["run_at"],
        "retest_report": str(retest_path),
        "retest_run_at": retest["run_at"],
        "replaced_id": update["id"],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("retest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    retest = json.loads(args.retest.read_text(encoding="utf-8"))
    report = project_report(base, retest, args.base, args.retest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
