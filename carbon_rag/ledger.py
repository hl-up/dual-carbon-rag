import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import PROVINCES


def now():
    return datetime.now(UTC).isoformat()


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL, metadata TEXT NOT NULL,
                status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '',
                page_count INTEGER DEFAULT 0, chunk_count INTEGER DEFAULT 0,
                failed_pages TEXT DEFAULT '[]', verified_at TEXT,
                embedding_model TEXT, embedding_dim INTEGER,
                created_at TEXT NOT NULL, ingested_at TEXT)""")
            db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("""CREATE TABLE IF NOT EXISTS operations (
                timestamp TEXT, action TEXT, doc_id TEXT, status TEXT,
                elapsed REAL, details TEXT)""")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def list(self):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item["metadata"])
            item["failed_pages"] = json.loads(item["failed_pages"])
            result.append(item)
        return result

    def get(self, doc_id):
        return next((r for r in self.list() if r["doc_id"] == doc_id), None)

    def find_hash(self, digest):
        return next((r for r in self.list() if r["sha256"] == digest), None)

    def update(self, doc_id, **values):
        allowed = {
            "metadata",
            "status",
            "error",
            "page_count",
            "chunk_count",
            "failed_pages",
            "verified_at",
            "embedding_model",
            "embedding_dim",
            "ingested_at",
        }
        if not values or set(values) - allowed:
            raise ValueError("无效台账字段")
        for field in ("metadata", "failed_pages"):
            if field in values:
                values[field] = json.dumps(values[field], ensure_ascii=False)
        with self.connection() as db:
            db.execute(
                "UPDATE documents SET "
                + ",".join(f"{key}=?" for key in values)
                + " WHERE doc_id=?",
                [*values.values(), doc_id],
            )

    def record(self, action, status, elapsed=0, doc_id="", details=None):
        # 调用方只传白名单统计；不保存异常原文、正文、问题或密钥。
        with self.connection() as db:
            db.execute(
                "INSERT INTO operations VALUES (?,?,?,?,?,?)",
                (
                    now(),
                    action,
                    doc_id,
                    status,
                    elapsed,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )

    def eligible(self, library="policy", scope="all", province=""):
        rows = [
            r
            for r in self.list()
            if r["status"] == "ready" and r["metadata"]["verification_status"] == "已核验"
        ]
        if library == "policy":
            rows = [
                r
                for r in rows
                if r["metadata"]["doc_type"] == "中国政策" and not r["metadata"]["is_translation"]
            ]
        elif library == "reference":
            rows = [r for r in rows if r["metadata"]["doc_type"] != "中国政策"]
        else:
            raise ValueError("资料库范围无效")
        if scope == "national":
            rows = [r for r in rows if r["metadata"]["jurisdiction"] == "国家"]
        elif scope == "province":
            if province not in PROVINCES:
                raise ValueError("请选择有效省级地区")
            rows = [
                r
                for r in rows
                if r["metadata"]["jurisdiction"] == "国家" or r["metadata"]["province"] == province
            ]
        elif scope != "all":
            raise ValueError("检索范围无效")
        return rows

    def coverage(self):
        rows = [r for r in self.list() if r["metadata"]["doc_type"] == "中国政策"]
        result = []
        for region in ["国家", *PROVINCES]:
            docs = [
                r
                for r in rows
                if (r["metadata"]["province"] or r["metadata"]["jurisdiction"]) == region
            ]
            verified = [r for r in docs if r["metadata"]["verification_status"] == "已核验"]
            ready = [r for r in verified if r["status"] == "ready"]
            result.append(
                {
                    "地区": region,
                    "文件数": len(docs),
                    "已核验": len(verified),
                    "可检索": len(ready),
                    "待核验": len(docs) - len(verified),
                    "状态": "已核验" if verified else "待核验" if docs else "未收录",
                    "最近核验时间": max((r["verified_at"] or "" for r in verified), default=""),
                }
            )
        return result
