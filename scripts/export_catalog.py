"""Read the supplied XLSX catalog into the final JSON import manifest.

Uses only the Python standard library so the application environment does not
need a spreadsheet authoring dependency. The original workbook is untouched;
rows marked 待核对 are excluded from the final catalog.
"""

import argparse
import hashlib
import json
import posixpath
import re
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
KEYS = [
    "filename",
    "source_url",
    "title",
    "issuer",
    "publish_date",
    "region",
    "check_result",
    "note",
]


def final_records(records):
    """Keep the reviewed catalog while excluding withdrawn source-check rows."""
    return [row for row in records if row["check_result"] != "待核对"]


def sheet_path(archive):
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationship_id = next(
        sheet.attrib[f"{REL}id"]
        for sheet in workbook.findall(f".//{MAIN}sheet")
        if sheet.attrib["name"] == "资料清单"
    )
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = next(
        item.attrib["Target"]
        for item in relationships.findall(f"{PKG_REL}Relationship")
        if item.attrib["Id"] == relationship_id
    )
    path = posixpath.normpath(
        target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
    )
    if not path.startswith("xl/worksheets/"):
        raise ValueError("资料清单工作表路径无效")
    return path


def read_catalog(source):
    with ZipFile(source) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join((t.text or "") for t in item.iter(f"{MAIN}t"))
                for item in strings.findall(f"{MAIN}si")
            ]
        sheet = ET.fromstring(archive.read(sheet_path(archive)))
        records = []
        for row in sheet.findall(f".//{MAIN}sheetData/{MAIN}row"):
            index = int(row.attrib["r"])
            if index == 1:
                continue
            cells = [""] * len(KEYS)
            for cell in row.findall(f"{MAIN}c"):
                match = re.match(r"([A-Z]+)", cell.attrib["r"])
                if not match or len(match[1]) != 1 or not "A" <= match[1] <= "H":
                    continue
                column = ord(match[1]) - ord("A")
                value = cell.find(f"{MAIN}v")
                if cell.attrib.get("t") == "s" and value is not None:
                    text = shared[int(value.text)]
                elif cell.attrib.get("t") == "inlineStr":
                    text = "".join((part.text or "") for part in cell.iter(f"{MAIN}t"))
                else:
                    text = value.text if value is not None and value.text else ""
                    if column == 4 and text and cell.attrib.get("t") != "str":
                        try:
                            serial = float(text)
                        except ValueError:
                            pass
                        else:
                            if serial > 30000:
                                text = (
                                    (datetime(1899, 12, 30) + timedelta(days=serial))
                                    .date()
                                    .isoformat()
                                )
                cells[column] = text.strip()
            if any(cells):
                records.append({"sheet_row": index, **dict(zip(KEYS, cells, strict=True))})
    if not records or any(not row["filename"] for row in records):
        raise ValueError("资料清单为空或文件名缺失")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("pdf/双碳RAG知识库资料清单.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("docs/catalog_manifest.json"))
    args = parser.parse_args()
    records = final_records(read_catalog(args.source))
    payload = {
        "source_workbook": str(args.source).replace("\\", "/"),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已整理 {len(records)} 条清单记录：{args.output}")


if __name__ == "__main__":
    main()
