import pytest

from carbon_rag.pdf import parse_pdf
from carbon_rag.service import PartialPagesError


def test_page_number_only_is_not_text(pdf_bytes):
    parsed = parse_pdf(pdf_bytes("123", "Real text"), "a.pdf")
    assert parsed.failed_pages == [1]
    assert [page for page, _ in parsed.pages] == [2]


def test_reindex_keeps_ids_and_removes_skipped_page(svc, metadata, pdf_bytes):
    row = svc.ingest(pdf_bytes("123", "Real text"), "a.pdf", metadata(), allow_partial=True)[
        "document"
    ]
    with pytest.raises(PartialPagesError):
        svc.reindex(row["doc_id"])
    assert svc.ledger.get(row["doc_id"])["status"] == "ready"
    rebuilt = svc.reindex(row["doc_id"], allow_partial=True)["document"]
    assert rebuilt["doc_id"] == row["doc_id"]
    assert rebuilt["failed_pages"] == [1]
    assert svc.index.count(row["doc_id"]) == rebuilt["chunk_count"] == 1
