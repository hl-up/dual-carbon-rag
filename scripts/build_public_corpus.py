"""Create the reviewed corpus snapshot for the public Streamlit app."""

import hashlib
import shutil
import sqlite3
from pathlib import Path

from carbon_rag.ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data"
TARGET = ROOT / "public_data"


def main():
    if TARGET.exists():
        raise SystemExit("public_data 已存在；请先人工检查并移走旧快照。")
    source = Ledger(SOURCE / "ledger.sqlite3")
    rows = source.list()
    if len(rows) != 57 or any(row["status"] != "ready" for row in rows):
        raise SystemExit("台账数量或入库状态与预期不符，停止生成。")
    for row in rows:
        path = SOURCE / "originals" / f"{row['doc_id']}.pdf"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"原件缺失或哈希不符：{row['doc_id']}")
    TARGET.mkdir()
    shutil.copytree(SOURCE / "originals", TARGET / "originals")
    shutil.copytree(
        SOURCE / "chroma",
        TARGET / "chroma",
        ignore=shutil.ignore_patterns("*.sqlite3-shm", "*.sqlite3-wal"),
    )
    with sqlite3.connect(SOURCE / "ledger.sqlite3") as original:
        with sqlite3.connect(TARGET / "ledger.sqlite3") as snapshot:
            original.backup(snapshot)
            snapshot.execute("DELETE FROM operations")
            snapshot.commit()
            snapshot.execute("VACUUM")
    public_rows = Ledger(TARGET / "ledger.sqlite3").list()
    if len(public_rows) != len(rows):
        raise SystemExit("快照数量核对失败。")
    print(f"公开快照已生成：{len(public_rows)} 份 PDF")


if __name__ == "__main__":
    main()
