"""Read-only policy tools and evidence collection for one Agent execution."""

from langchain_core.tools import tool

from .pdf import read_page_text


class PolicyTools:
    def __init__(self, service, rows, evidence, max_evidence, max_chars=20000):
        self.service = service
        self.rows = rows
        self.allowed = {row["doc_id"]: row for row in rows}
        self.evidence = evidence
        self.max_evidence = max_evidence
        self.max_chars = max_chars
        self.seen = {item["chunk_id"] for item in evidence}
        self.tools = self._build_tools()

    def collect(self, candidates):
        collected = []
        for candidate in candidates:
            key = candidate["chunk_id"]
            remaining = self.max_chars - sum(len(c["excerpt"]) for c in self.evidence)
            if key not in self.seen and len(self.evidence) < self.max_evidence and remaining > 0:
                self.seen.add(key)
                item = {
                    **candidate,
                    "number": len(self.evidence) + 1,
                    "excerpt": candidate["excerpt"][:remaining],
                }
                self.evidence.append(item)
            item = next((c for c in self.evidence if c["chunk_id"] == key), None)
            if item:
                collected.append(item)
        return collected

    def _build_tools(self):
        svc, rows, allowed = self.service, self.rows, self.allowed
        collect = self.collect

        @tool
        def list_documents() -> list[dict]:
            """列出当前允许检索且已核验的资料及其文档ID、标题和PDF总页数。"""
            return [
                {
                    "doc_id": row["doc_id"],
                    "title": row["metadata"]["title"],
                    "pages": row["page_count"],
                    "issuer": row["metadata"]["issuer"],
                    "region": row["metadata"]["province"] or row["metadata"]["jurisdiction"],
                }
                for row in rows
            ]

        @tool
        def search_documents(query: str, doc_id: str = "") -> list[dict]:
            """检索原文证据。query为具体问题/关键词；doc_id可指定list_documents中的文件，留空搜索当前范围。"""
            if not query.strip() or len(query) > 1000:
                return [{"error": "检索词应为1到1000字"}]
            if doc_id and doc_id not in allowed:
                return [{"error": "文档不在当前允许范围"}]
            selected = [allowed[doc_id]] if doc_id else rows
            return collect(svc.retrieve(query, selected))

        @tool
        def read_pdf_page(doc_id: str, page: int) -> list[dict]:
            """读取当前范围内已核验文件的指定PDF页序（从1开始），用于核对原文上下文。"""
            row = allowed.get(doc_id)
            if row is None or not 1 <= page <= row["page_count"]:
                return [{"error": "文档或PDF页码不在允许范围"}]
            if page in row["failed_pages"]:
                return [{"error": "该页为已确认跳过的图片或解析异常页，没有可用文字"}]
            text = read_page_text(svc.raw_dir / f"{doc_id}.pdf", page)
            if not text:
                return [{"error": "该页无法提取文字"}]
            return collect(
                [
                    {
                        "doc_id": doc_id,
                        **row["metadata"],
                        "page": page,
                        "chunk_id": f"{doc_id}:{page}:full",
                        "excerpt": text[:12000],
                    }
                ]
            )

        return {t.name: t for t in (list_documents, search_documents, read_pdf_page)}
