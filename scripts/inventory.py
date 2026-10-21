"""只读盘点；不联网、不向量化、不根据文件名认定政策。"""

import argparse
import csv
import hashlib
import logging
from pathlib import Path

import pdfplumber


def inventory(root, output):
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    rows = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        row = {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_mb": round(path.stat().st_size / 1024**2, 2),
            "page_count": "",
            "sample_pages": "",
            "sample_characters": "",
            "parse_status": "不支持的非PDF格式",
            "title": "",
            "issuer": "",
            "publish_date": "",
            "source_url": "",
            "doc_type": "",
            "jurisdiction": "",
            "province": "",
            "verification_status": "待核验",
            "source_note": "",
        }
        if path.suffix.lower() == ".pdf":
            try:
                with pdfplumber.open(path) as pdf:
                    row["page_count"] = len(pdf.pages)
                    samples = (
                        sorted({0, len(pdf.pages) // 2, len(pdf.pages) - 1}) if pdf.pages else []
                    )
                    chars = sum(len((pdf.pages[i].extract_text() or "").strip()) for i in samples)
                    row["sample_pages"] = ",".join(str(i + 1) for i in samples)
                    row["sample_characters"] = chars
                    row["parse_status"] = (
                        "抽样可提取（待逐页检查）" if chars else "抽样无文字（需检查OCR）"
                    )
            except Exception:
                row["parse_status"] = "解析失败（需检查损坏或加密）"
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else ["filename"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"盘点 {len(rows)} 份文件，写入 {output}；全部保留待核验，不自动入库。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("pdf"))
    parser.add_argument("--output", type=Path, default=Path("docs/material_inventory.csv"))
    args = parser.parse_args()
    inventory(args.root, args.output)
