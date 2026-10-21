import hashlib
import json
import re
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from uuid import uuid4

from langsmith import traceable

from .conversation import contextual_question, social_reply, validate_question
from .ledger import Ledger, now
from .models import PROVINCES, Metadata
from .pdf import parse_pdf, split_pages
from .providers import AnswerModel, safe_error
from .retrieval import CrossEncoderReranker, compress_evidence
from .vector_index import VectorIndex


class PartialPagesError(ValueError):
    def __init__(self, pages):
        self.pages = pages
        super().__init__(f"以下 PDF 页无法提取文字：{pages}。请先查看解析预览并确认继续。")


def exact_dates(text):
    """Normalize only explicit full dates; partial publication years stay partial."""
    chinese = re.findall(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    iso = re.findall(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", text)
    return {(int(year), int(month), int(day)) for year, month, day in chinese + iso}


@dataclass
class Answer:
    status: str
    text: str
    citations: list[dict] = field(default_factory=list)
    elapsed: float = 0
    retrieved: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)


class RAGService:
    def __init__(self, settings, index=None, chat=None):
        self.settings = settings
        self.lock = RLock()
        self.ledger = Ledger(settings.data_dir / "ledger.sqlite3")
        self.raw_dir = settings.data_dir / "originals"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index = index if index is not None else VectorIndex(settings)
        self.chat = chat if chat is not None else AnswerModel(settings)
        self.reranker = (
            CrossEncoderReranker(settings.rerank_model) if settings.rerank_model else None
        )
        signature = json.dumps({"model": settings.embedding_model, "dim": settings.embedding_dim})
        chunk_signature = json.dumps(
            {"size": settings.chunk_size, "overlap": settings.chunk_overlap}
        )
        with self.ledger.connection() as db:
            old = db.execute("SELECT value FROM settings WHERE key='embedding'").fetchone()
            if old and old[0] != signature:
                raise ValueError(
                    "嵌入模型或维度与现有索引不一致。请使用新的 DATA_DIR 重建；禁止混用旧向量。"
                )
            db.execute("INSERT OR IGNORE INTO settings VALUES ('embedding', ?)", (signature,))
            old_chunks = db.execute("SELECT value FROM settings WHERE key='chunking'").fetchone()
            if old_chunks and old_chunks[0] != chunk_signature:
                raise ValueError("切片参数与现有索引不一致。请使用新的 DATA_DIR 重建索引。")
            db.execute("INSERT OR IGNORE INTO settings VALUES ('chunking', ?)", (chunk_signature,))
        self.recover()

    def recover(self):
        # 已提交 ready 的才可检索；崩溃留下的中间状态清理后允许重试。
        for row in self.ledger.list():
            if row["status"] in {"ingesting", "deleting"}:
                try:
                    self.index.delete(row["doc_id"])
                    if row["status"] == "deleting":
                        self._finish_delete(row["doc_id"])
                    else:
                        self.ledger.update(
                            row["doc_id"],
                            status="failed",
                            chunk_count=0,
                            error="上次入库中断，已清理，请重新入库",
                        )
                except Exception:
                    # 保留不可检索的中间状态，下次恢复或由维护者重试删除。
                    self.ledger.update(row["doc_id"], error="清理未完成，请重启或重试删除")

    def ingest(self, content: bytes, filename: str, metadata: Metadata, allow_partial=False):
        with self.lock:
            return self._ingest(content, filename, metadata, allow_partial)

    def _ingest(self, content, filename, metadata, allow_partial, force=False):
        started = perf_counter()
        meta = metadata.validate()
        parsed = parse_pdf(content, filename, self.settings.max_upload_mb)
        if parsed.failed_pages and not allow_partial:
            raise PartialPagesError(parsed.failed_pages)
        digest = hashlib.sha256(content).hexdigest()
        existing = self.ledger.find_hash(digest)
        if existing and existing["status"] == "ready" and not force:
            return {"duplicate": True, "document": existing}
        doc_id = existing["doc_id"] if existing else uuid4().hex
        if existing:
            self.ledger.update(doc_id, status="ingesting", error="", chunk_count=0)
        else:
            with self.ledger.connection() as db:
                db.execute(
                    """INSERT INTO documents
                    (doc_id,sha256,filename,metadata,status,created_at)
                    VALUES (?,?,?,?,?,?)""",
                    (
                        doc_id,
                        digest,
                        filename.replace("\\", "/").split("/")[-1],
                        json.dumps(meta, ensure_ascii=False),
                        "ingesting",
                        now(),
                    ),
                )
        try:
            if existing:
                self.index.delete(doc_id)
            (self.raw_dir / f"{doc_id}.pdf").write_bytes(content)
            chunks = split_pages(
                parsed, doc_id, self.settings.chunk_size, self.settings.chunk_overlap
            )
            self.ledger.update(
                doc_id,
                metadata=meta,
                page_count=parsed.page_count,
                failed_pages=parsed.failed_pages,
                verified_at=now() if meta["verification_status"] == "已核验" else None,
                embedding_model=self.settings.embedding_model,
                embedding_dim=self.settings.embedding_dim,
            )
            self.index.add(chunks)
            if self.index.count(doc_id) != len(chunks):
                raise RuntimeError("索引片段数量核对失败")
            self.ledger.update(doc_id, status="ready", chunk_count=len(chunks), ingested_at=now())
            self.ledger.record(
                "ingest",
                "ready",
                perf_counter() - started,
                doc_id,
                {"chunks": len(chunks), "model": self.settings.embedding_model},
            )
            return {"duplicate": False, "document": self.ledger.get(doc_id)}
        except Exception as exc:
            try:
                self.index.delete(doc_id)
            except Exception:
                pass  # 即使供应商/索引清理失败，failed 状态也不可检索；重试先清残留。
            message = safe_error(exc)
            self.ledger.update(doc_id, status="failed", error=message, chunk_count=0)
            self.ledger.record(
                "ingest",
                "failed",
                perf_counter() - started,
                doc_id,
                {"error_type": type(exc).__name__},
            )
            raise RuntimeError(message) from None

    def reindex(self, doc_id, allow_partial=False):
        with self.lock:
            row = self.ledger.get(doc_id)
            if row is None:
                raise ValueError("文件不存在")
            return self._ingest(
                (self.raw_dir / f"{doc_id}.pdf").read_bytes(),
                row["filename"],
                Metadata(**row["metadata"]),
                allow_partial,
                force=True,
            )

    def update_metadata(self, doc_id, metadata):
        with self.lock:
            if not self.ledger.get(doc_id):
                raise ValueError("文件不存在")
            meta = metadata.validate()
            self.ledger.update(
                doc_id,
                metadata=meta,
                verified_at=now() if meta["verification_status"] == "已核验" else None,
            )
            # 查询在 SQLite 选出允许的 doc_id，引用元数据也从 SQLite 读取，无旧向量元数据。

    def delete(self, doc_id):
        with self.lock:
            if not self.ledger.get(doc_id):
                raise ValueError("文件不存在")
            self.ledger.update(doc_id, status="deleting")
            try:
                self.index.delete(doc_id)
                self._finish_delete(doc_id)
                self.ledger.record("delete", "deleted", doc_id=doc_id)
            except Exception as exc:
                self.ledger.update(doc_id, error=safe_error(exc))
                raise RuntimeError("删除未完成，文件已退出检索，可重试删除") from None

    def _finish_delete(self, doc_id):
        (self.raw_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
        with self.ledger.connection() as db:
            db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    @traceable(run_type="chain", name="rag_ask")
    def ask(self, question, library="policy", scope="all", province="", previous_question=""):
        started = perf_counter()
        issue = validate_question(question)
        reply = social_reply(question) if issue is None else None
        if issue is not None:
            result = Answer(*issue)
        elif reply is not None:
            result = Answer("social", reply)
        else:
            effective_question = contextual_question(question, previous_question)
            result = self._ask(effective_question, library, scope, province)
        result.elapsed = perf_counter() - started
        self.ledger.record(
            "ask", result.status, result.elapsed, details={"model": self.settings.chat_model}
        )
        return result

    def scope_documents(self, question, library="policy", scope="all", province=""):
        if not question.strip():
            return [], Answer("empty", "请输入有效问题")
        rows = self.ledger.eligible(library, scope, province)
        local = [r for r in rows if r["metadata"]["province"] == province]
        if scope == "province" and not local:
            return rows, Answer(
                "insufficient",
                "当前未收录资料：所选地区暂无已核验且已入库文件。可切换“仅国家”查询国家文件。",
            )
        # 明确提及未覆盖地区时先停止，避免从另一地区资料编出比较。
        if library == "policy":
            covered = {r["metadata"]["province"] for r in rows}
            for region in PROVINCES:
                alias = region[:2] if region not in {"黑龙江省", "内蒙古自治区"} else region[:3]
                if (region in question or alias in question) and region not in covered:
                    return rows, Answer(
                        "insufficient",
                        f"当前检索范围未收录 {region} 的已核验资料，无法回答该地区或完成比较。",
                    )
        if not rows:
            return rows, Answer(
                "insufficient", "当前已收录资料中未找到足够依据；本范围暂无已核验且已入库文件。"
            )
        return rows, None

    @traceable(run_type="retriever", name="policy_retrieval")
    def retrieve(self, question, rows):
        started = perf_counter()
        k = self.settings.rerank_candidates if self.reranker else self.settings.top_k
        found = self.index.search(question, [r["doc_id"] for r in rows], k)
        if self.reranker:
            found = self.reranker.rerank(question, found)
        if self.settings.max_cosine_distance is None:
            lexical = [
                (chunk, None)
                for chunk in self.index.keyword_search(question, [r["doc_id"] for r in rows])
            ]
            found = lexical + found
        found = list({chunk.id: (chunk, distance) for chunk, distance in found}.values())
        found = found[: self.settings.top_k]
        allowed = {r["doc_id"]: r for r in rows}
        citations = []
        for chunk, distance in found:
            row = allowed.get(chunk.metadata["doc_id"])
            if row is None or (
                self.settings.max_cosine_distance is not None
                and distance is not None
                and distance > self.settings.max_cosine_distance
            ):
                continue
            citations.append(
                {
                    "number": len(citations) + 1,
                    "doc_id": row["doc_id"],
                    **row["metadata"],
                    "chunk_id": chunk.id
                    or f"{row['doc_id']}:{chunk.metadata['page']}:{chunk.metadata['chunk']}",
                    "page": chunk.metadata["page"],
                    "excerpt": chunk.page_content,
                }
            )
        self.ledger.record(
            "retrieve",
            "ok",
            perf_counter() - started,
            details={
                "candidate_count": len(found),
                "result_count": len(citations),
                "reranker": self.settings.rerank_model or "none",
            },
        )
        return citations

    @traceable(run_type="chain", name="evidence_answer")
    def answer_evidence(self, question, citations):
        if not citations:
            return Answer(
                "insufficient", "我还没找到能确认这个问题的原文依据。可以换个问法或补充相关资料。"
            )
        try:
            context = compress_evidence(question, citations, self.settings.max_context_chars)
            for attempt in range(2):
                request = question
                if attempt:
                    request += "\n请只返回 JSON；每个非空段落都要标注有效的 [编号] 引用。"
                started = perf_counter()
                raw, usage = self.chat.invoke(request, context)
                self.ledger.record(
                    "chat_usage",
                    "received",
                    perf_counter() - started,
                    details={
                        "model": self.settings.chat_model,
                        "context_chars_before": sum(len(c["excerpt"]) for c in citations),
                        "context_chars_after": sum(len(c["excerpt"]) for c in context),
                        **{
                            k: usage[k]
                            for k in ("input_tokens", "output_tokens", "total_tokens")
                            if k in usage
                        },
                    },
                )
                try:
                    data = json.loads(raw)
                    if data.get("supported") is False:
                        return Answer(
                            "insufficient",
                            "我找到了相关资料，但现有原文还不足以确认这一点。你可以补充更明确的文件，或换个角度问我。",
                            retrieved=citations,
                        )
                    answer = data["answer"]
                    if data.get("supported") is not True or not isinstance(answer, str):
                        raise ValueError()
                    refs = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
                    paragraphs = [p for p in answer.split("\n") if p.strip()]
                    if not refs or refs - set(range(1, len(citations) + 1)):
                        raise ValueError()
                    if any(not re.search(r"\[\d+\]", p) for p in paragraphs):
                        raise ValueError()
                    break
                except (ValueError, KeyError, TypeError, AttributeError):
                    if attempt:
                        return Answer(
                            "review",
                            "回答待核查：模型输出格式不符或引用缺失、越界，已停止展示该回答。",
                            citations,
                            retrieved=citations,
                        )
            for paragraph in paragraphs:
                numbers = {int(n) for n in re.findall(r"\[(\d+)\]", paragraph)}
                cited_text = "\n".join(
                    item["excerpt"] for item in citations if item["number"] in numbers
                )
                if exact_dates(paragraph) - exact_dates(cited_text):
                    return Answer(
                        "insufficient",
                        "现有原文不足以确认回答中的精确日期，已停止展示该回答。",
                        retrieved=citations,
                    )
            return Answer(
                "answered",
                answer,
                [c for c in citations if c["number"] in refs],
                retrieved=citations,
            )
        except Exception as exc:
            self.ledger.record("api", "failed", details={"error_type": type(exc).__name__})
            return Answer("error", safe_error(exc), retrieved=citations)

    def _ask(self, question, library, scope, province):
        rows, stopped = self.scope_documents(question, library, scope, province)
        if stopped is not None:
            return stopped
        try:
            return self.answer_evidence(question, self.retrieve(question, rows))
        except Exception as exc:
            self.ledger.record("api", "failed", details={"error_type": type(exc).__name__})
            return Answer("error", safe_error(exc))
