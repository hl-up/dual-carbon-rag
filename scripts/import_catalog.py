"""Import the user's reviewed catalog PDFs, recording every skipped row and page.

The manifest is an exact snapshot of the supplied spreadsheet. Re-running this
script is safe: already indexed PDF hashes are reported as duplicates.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_rag.config import Settings
from carbon_rag.models import Metadata
from carbon_rag.pdf import parse_pdf
from carbon_rag.service import RAGService

ROOT = Path("pdf")
MANIFEST = Path("docs/catalog_manifest.json")
OUTPUT = Path("docs/catalog_import_results.json")
CITY_PROVINCES = {"深圳市": "广东省", "武汉市": "湖北省"}
TRANSLATION_ROWS = {50, 51, 53, 54}
RESEARCH_ROWS = {13, 25, 26}
OTHER_REFERENCE_ROWS = {8, 19, 27}
STANDARD_ROWS = set(range(50, 60))
MATCHED_PDF_ALTERNATIVES = {
    # The local PDF and catalog DOCX have 99.5% normalized text similarity.
    45: (
        ROOT / "江苏省工业领域节能技改行动计划（2022-2025年）.pdf",
        "8771ced105491876e7d0bf61961a8589282cf5ff47f16ff613b6b9a89c48216c",
    )
}


def normalized_name(name):
    return re.sub(r"\s+", "", name).casefold()


def resolve_file(filename, files):
    exact = [path for path in files if path.name == filename]
    matches = exact or [
        path for path in files if normalized_name(path.name) == normalized_name(filename)
    ]
    if not matches:
        return None, "文件未找到"
    hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in matches}
    if len(hashes) != 1:
        return None, "同名文件内容不一致，需选择版本"
    return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0], (
        "完全匹配" if exact else "忽略空格后匹配"
    )


def classify(record):
    row = record["sheet_row"]
    region = record["region"]
    if region == "全国":
        jurisdiction, province = "国家", ""
    elif region == "国际":
        jurisdiction, province = "国际", ""
    else:
        jurisdiction, province = "省级", CITY_PROVINCES.get(region, region)
    if row in STANDARD_ROWS:
        doc_type = "标准"
    elif row in RESEARCH_ROWS:
        doc_type = "研究报告"
    elif row in OTHER_REFERENCE_ROWS:
        doc_type = "其他参考"
    elif jurisdiction == "国际":
        doc_type = "国际资料"
    else:
        doc_type = "中国政策"
    return jurisdiction, province, doc_type


def metadata_for(record, failed_pages):
    jurisdiction, province, doc_type = classify(record)
    note_parts = [record["note"]] if record["note"] else []
    if record["region"] in CITY_PROVINCES:
        note_parts.append(f"适用城市：{record['region']}")
    if failed_pages:
        note_parts.append(
            f"PDF 共 {len(failed_pages)} 页无法提取正文，按用户 2026-10-19 确认跳过；"
            "具体 PDF 页序见台账异常页"
        )
    note_parts.append(f"资料清单第 {record['sheet_row']} 行")
    return Metadata(
        title=record["title"],
        issuer=record["issuer"],
        publish_date=record["publish_date"],
        source_url=record["source_url"],
        doc_type=doc_type,
        jurisdiction=jurisdiction,
        province=province,
        verification_status="已核验",
        is_translation=record["sheet_row"] in TRANSLATION_ROWS,
        language="中文",
        source_note="；".join(note_parts),
    )


def save(records, workbook_sha):
    payload = {
        "source_workbook": str(ROOT / "双碳RAG知识库资料清单.xlsx"),
        "source_sha256": workbook_sha,
        "updated_date": date.today().isoformat(),
        "summary": dict(Counter(row["status"] for row in records)),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-row", type=int, default=2)
    parser.add_argument("--to-row", type=int, default=66)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    workbook = ROOT / "双碳RAG知识库资料清单.xlsx"
    workbook_sha = hashlib.sha256(workbook.read_bytes()).hexdigest()
    if workbook_sha != manifest["source_sha256"]:
        raise ValueError("资料清单已修改，请先重新生成目录快照")
    settings = Settings.from_env()
    service = RAGService(settings)
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    previous_payload = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    previous = (
        previous_payload.get("records", [])
        if previous_payload.get("source_sha256") == workbook_sha
        else []
    )
    active_rows = {item["sheet_row"] for item in manifest["records"]}
    results = {item["sheet_row"]: item for item in previous if item["sheet_row"] in active_rows}
    for record in manifest["records"]:
        row = record["sheet_row"]
        if row < args.from_row or row > args.to_row:
            continue
        result = {"sheet_row": row, "filename": record["filename"]}
        try:
            path, match_note = resolve_file(record["filename"], files)
            result["match_note"] = match_note
            if path is not None and row in MATCHED_PDF_ALTERNATIVES:
                alternative, expected_hash = MATCHED_PDF_ALTERNATIVES[row]
                if (
                    alternative.exists()
                    and hashlib.sha256(alternative.read_bytes()).hexdigest() == expected_hash
                ):
                    result["original_path"] = str(path)
                    path = alternative
                    result["match_note"] = "使用已核对同文 PDF（与清单 DOCX 文字相似度 99.5%）"
            if path is None:
                result["status"] = "文件待补齐"
            elif record["check_result"] != "发布页与 PDF 一致":
                result["status"] = "来源待核对"
                result["path"] = str(path)
            elif path.suffix.lower() != ".pdf":
                result["status"] = "格式待转换"
                result["path"] = str(path)
            else:
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                result.update(path=str(path), sha256=digest, size_bytes=len(content))
                existing = service.ledger.find_hash(digest)
                if existing and existing["status"] == "ready":
                    text_pages = existing["page_count"] - len(existing["failed_pages"])
                    if text_pages * 2 < existing["page_count"]:
                        existing_meta = Metadata(**existing["metadata"])
                        existing_meta.verification_status = "待核验"
                        if "正文不足半数页面可提取" not in existing_meta.source_note:
                            existing_meta.source_note += "；正文不足半数页面可提取，需 OCR 后复核"
                        service.update_metadata(existing["doc_id"], existing_meta)
                        result["status"] = "需 OCR"
                    else:
                        result["status"] = "重复已入库"
                    result.update(
                        doc_id=existing["doc_id"],
                        pages=existing["page_count"],
                        chunks=existing["chunk_count"],
                        skipped_pages=existing["failed_pages"],
                    )
                else:
                    parsed = parse_pdf(content, path.name, settings.max_upload_mb)
                    if len(parsed.pages) * 2 < parsed.page_count:
                        result.update(
                            status="需 OCR",
                            pages=parsed.page_count,
                            text_pages=len(parsed.pages),
                            skipped_pages=parsed.failed_pages,
                        )
                    else:
                        meta = metadata_for(record, parsed.failed_pages)
                        meta.validate()
                        ingested = service.ingest(
                            content, path.name, meta, allow_partial=bool(parsed.failed_pages)
                        )["document"]
                        result.update(
                            status="已入库",
                            doc_id=ingested["doc_id"],
                            pages=ingested["page_count"],
                            chunks=ingested["chunk_count"],
                            skipped_pages=ingested["failed_pages"],
                            doc_type=meta.doc_type,
                            jurisdiction=meta.jurisdiction,
                            province=meta.province,
                        )
        except Exception as exc:
            # Do not persist provider error text: an HTTP exception can contain
            # request payload or credentials. PDF parser messages are fixed.
            scan_only = isinstance(exc, ValueError) and str(exc).startswith("没有可提取文字")
            result["status"] = "需 OCR" if scan_only else "处理失败"
            result["error_type"] = type(exc).__name__
            result["error"] = (
                str(exc)
                if isinstance(exc, ValueError)
                and str(exc).startswith(("没有可提取文字", "文件超过", "PDF 解析失败"))
                else "需查看本地日志后重试"
            )
        results[row] = result
        ordered = [results[key] for key in sorted(results)]
        save(ordered, workbook_sha)
        print(
            f"清单第 {row} 行：{result['status']}"
            + (
                f"，{result.get('pages')} 页，{result.get('chunks')} 片段"
                if result["status"] == "已入库"
                else ""
            ),
            flush=True,
        )
        if result["status"] == "处理失败" and result.get("error_type") == "RuntimeError":
            # Avoid sending repeated API requests on a service failure.
            break
    print(json.dumps(dict(Counter(row["status"] for row in results.values())), ensure_ascii=False))


if __name__ == "__main__":
    main()
