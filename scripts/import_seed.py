"""仅导入用户指定且核对过的首批官方资料；可安全重复执行。"""

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carbon_rag.config import Settings
from carbon_rag.ledger import now
from carbon_rag.models import Metadata
from carbon_rag.pdf import parse_pdf
from carbon_rag.service import RAGService


def normalize(value):
    return "".join(value.split())


def download_pdf(url, target, max_mb):
    if urlparse(url).scheme != "https" or urlparse(url).hostname != "www.mee.gov.cn":
        raise ValueError("首批资料下载仅允许生态环境部官方 HTTPS 地址")
    if target.exists():
        return target.read_bytes()
    with urlopen(Request(url, headers={"User-Agent": "DualCarbonRAG/0.1"}), timeout=60) as response:
        if urlparse(response.url).hostname != "www.mee.gov.cn":
            raise ValueError("下载发生跨站重定向，请核对来源")
        data = response.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024 or not data.startswith(b"%PDF-"):
        raise ValueError("下载内容并非有效大小的PDF")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    records = json.loads(Path("docs/seed_documents.json").read_text(encoding="utf-8"))
    service = None if args.download_only else RAGService(settings)
    output = []
    for record in records:
        path = Path("pdf") / record["filename"]
        data = download_pdf(record["pdf_url"], path, settings.max_upload_mb)
        parsed = parse_pdf(data, path.name, settings.max_upload_mb)
        # 封面会换行或包含空格，以去空白后的正式标题核对下载文件。
        if normalize(record["title"]) not in normalize("\n".join(t for _, t in parsed.pages[:3])):
            raise ValueError(f"{record['record_id']}：PDF 标题与核验记录不符")
        if set(parsed.failed_pages) - set(record.get("allowed_missing_pages", [])):
            raise ValueError(f"{record['record_id']} 存在异常页 {parsed.failed_pages}，未自动跳过")
        row = {
            "record_id": record["record_id"],
            "filename": record["filename"],
            "sha256": hashlib.sha256(data).hexdigest(),
            "pages": parsed.page_count,
            "source_url": record["source_url"],
            "pdf_url": record["pdf_url"],
            "verification_status": "已核验",
            "checked_at": now(),
            "skipped_pages": parsed.failed_pages,
        }
        print(
            f"{record['record_id']}: PDF {parsed.page_count} 页，可提取 {len(parsed.pages)} 页，已确认跳过 {parsed.failed_pages}",
            flush=True,
        )
        if service is not None:
            allowed = {f.name for f in fields(Metadata)}
            meta = Metadata(**{k: v for k, v in record.items() if k in allowed})
            result = service.ingest(data, path.name, meta, allow_partial=bool(parsed.failed_pages))
            document = result["document"]
            if result["duplicate"]:
                service.update_metadata(document["doc_id"], meta)
                if document["failed_pages"] != parsed.failed_pages:
                    result = service.reindex(
                        document["doc_id"], allow_partial=bool(parsed.failed_pages)
                    )
                    document = result["document"]
            row.update(
                doc_id=document["doc_id"],
                chunks=document["chunk_count"],
                ingest_status=document["status"],
                embedding_model=document["embedding_model"],
                embedding_dim=document["embedding_dim"],
            )
            print(
                f"{record['record_id']}: {document['status']}, {document['chunk_count']} 个片段",
                flush=True,
            )
        output.append(row)
    target = Path("docs/verified_seed_files.json")
    # 仅下载时不覆盖已有包含入库 ID 的完整记录。
    if service is not None or not target.exists():
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
