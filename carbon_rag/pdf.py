import io
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ParseResult:
    pages: list[tuple[int, str]]
    failed_pages: list[int]
    page_count: int


def read_page_text(path: Path, page: int) -> str:
    """Read one original PDF page by its 1-based PDF page number."""
    with pdfplumber.open(path) as pdf:
        return (pdf.pages[page - 1].extract_text() or "").strip()


def parse_pdf(content: bytes, filename: str, max_mb=50) -> ParseResult:
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
        raise ValueError("仅支持具有有效文件签名的 PDF")
    if len(content) > max_mb * 1024 * 1024:
        raise ValueError(f"文件超过 {max_mb} MB 上限")
    pages, failed = [], []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            count = len(pdf.pages)
            for number, page in enumerate(pdf.pages, 1):
                try:
                    text = (page.extract_text() or "").strip()
                except Exception:
                    text = ""
                # 图片页可能只提取出页码，不能当作正文解析成功。
                if text and any(char.isalpha() for char in text):
                    pages.append((number, text))
                else:
                    failed.append(number)
    except Exception as exc:
        raise ValueError("PDF 解析失败：文件损坏、加密或格式不受支持") from exc
    if not pages:
        raise ValueError("没有可提取文字，可能是扫描件，需要 OCR；未入库")
    return ParseResult(pages, failed, count)


def split_pages(parsed: ParseResult, doc_id: str, size=800, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = []
    for page, text in parsed.pages:
        for index, part in enumerate(splitter.split_text(text)):
            chunks.append(
                Document(
                    page_content=part,
                    metadata={"doc_id": doc_id, "page": page, "chunk": index},
                    id=f"{doc_id}:{page}:{index}",
                )
            )
    return chunks
