from dataclasses import replace

import pytest
from conftest import FakeChat, local_index

from carbon_rag.pdf import parse_pdf, split_pages
from carbon_rag.service import PartialPagesError, RAGService


def test_ingest_restart_deduplicate_delete_reupload(svc, settings, metadata, pdf_bytes):
    content = pdf_bytes("National test evidence.")
    result = svc.ingest(content, "one.pdf", metadata())
    doc_id = result["document"]["doc_id"]
    assert result["document"]["status"] == "ready"
    assert result["document"]["page_count"] == 1
    assert svc.index.count(doc_id) == 1
    assert svc.ingest(content, "copy.pdf", metadata())["duplicate"]
    assert len(svc.ledger.list()) == 1
    reopened = RAGService(settings, index=local_index(settings), chat=FakeChat())
    assert reopened.ask("test").status == "answered"
    reopened.delete(doc_id)
    assert not reopened.ledger.list()
    assert reopened.index.count(doc_id) == 0
    assert reopened.ask("test").status == "insufficient"
    new = reopened.ingest(content, "one.pdf", metadata())["document"]
    assert new["doc_id"] != doc_id
    assert reopened.index.count(new["doc_id"]) == 1


def test_same_name_different_content_and_path_traversal(svc, metadata, pdf_bytes):
    a = svc.ingest(pdf_bytes("first"), "../../same.pdf", metadata())["document"]
    b = svc.ingest(pdf_bytes("second"), "../../same.pdf", metadata())["document"]
    assert a["doc_id"] != b["doc_id"]
    assert a["filename"] == b["filename"] == "same.pdf"
    assert {p.name for p in svc.raw_dir.iterdir()} == {a["doc_id"] + ".pdf", b["doc_id"] + ".pdf"}


@pytest.mark.parametrize(
    "changes",
    [
        {"source_url": ""},
        {"source_url": "javascript:alert(1)"},
        {"issuer": ""},
        {"publish_date": "2022-13-01"},
        {"jurisdiction": "省级", "province": ""},
        {"jurisdiction": "国家", "province": "广东省"},
        {"doc_type": "中国政策", "is_translation": True},
    ],
)
def test_metadata_validation_no_writes(svc, metadata, pdf_bytes, changes):
    with pytest.raises(ValueError):
        svc.ingest(pdf_bytes("test"), "test.pdf", metadata(**changes))
    assert not svc.ledger.list()
    assert not list(svc.raw_dir.iterdir())


@pytest.mark.parametrize("published", ["2006", "2018-12", "2024-02-29"])
def test_publication_date_preserves_available_precision(metadata, published):
    assert metadata(publish_date=published).validate()["publish_date"] == published


@pytest.mark.parametrize(
    "filename,content",
    [("fake.pdf", b"not pdf"), ("x.txt", b"%PDF-1.7"), ("bad.pdf", b"%PDF-1.7\ngarbage")],
)
def test_invalid_files_no_writes(svc, metadata, filename, content):
    with pytest.raises(ValueError):
        svc.ingest(content, filename, metadata())
    assert not svc.ledger.list()


def test_size_limit():
    with pytest.raises(ValueError, match="上限"):
        parse_pdf(b"%PDF-" + b"x" * 1024**2, "x.pdf", 1)


def test_scan_partial_and_page_identity(svc, metadata, pdf_bytes):
    with pytest.raises(ValueError, match="OCR"):
        svc.ingest(pdf_bytes(""), "scan.pdf", metadata())
    content = pdf_bytes("", "Real evidence on page two.")
    with pytest.raises(PartialPagesError) as exc:
        svc.ingest(content, "partial.pdf", metadata())
    assert exc.value.pages == [1]
    assert not svc.ledger.list()
    row = svc.ingest(content, "partial.pdf", metadata(), allow_partial=True)["document"]
    assert row["failed_pages"] == [1]
    assert svc.ask("evidence").citations[0]["page"] == 2


def test_chunks_never_cross_pages(pdf_bytes):
    result = parse_pdf(pdf_bytes("A" * 300, "B" * 300), "x.pdf")
    chunks = split_pages(result, "stable", size=100, overlap=10)
    assert len(chunks) > 2
    assert all(len(c.page_content) <= 100 for c in chunks)
    assert all(set(c.page_content) <= ({"A"} if c.metadata["page"] == 1 else {"B"}) for c in chunks)
    assert [c.id for c in chunks] == [c.id for c in split_pages(result, "stable", 100, 10)]


def test_partial_vector_write_rollback_and_retry(svc, metadata, pdf_bytes, monkeypatch):
    original = svc.index.add

    def broken(chunks):
        original(chunks[:1])
        raise RuntimeError("sensitive-provider-body")

    monkeypatch.setattr(svc.index, "add", broken)
    content = pdf_bytes("rollback test")
    with pytest.raises(RuntimeError):
        svc.ingest(content, "test.pdf", metadata())
    row = svc.ledger.list()[0]
    assert row["status"] == "failed"
    assert svc.index.count(row["doc_id"]) == 0
    assert svc.ask("rollback").status == "insufficient"
    assert "sensitive" not in row["error"]
    monkeypatch.setattr(svc.index, "add", original)
    assert svc.ingest(content, "test.pdf", metadata())["document"]["status"] == "ready"
    assert len(svc.ledger.list()) == 1


def test_crash_recovery(svc, settings, metadata, pdf_bytes):
    row = svc.ingest(pdf_bytes("crash"), "x.pdf", metadata())["document"]
    svc.ledger.update(row["doc_id"], status="ingesting")
    restarted = RAGService(settings, index=local_index(settings), chat=FakeChat())
    assert restarted.ledger.get(row["doc_id"])["status"] == "failed"
    assert restarted.index.count(row["doc_id"]) == 0


def test_interrupted_delete_stays_unsearchable_and_recovers(
    svc, settings, metadata, pdf_bytes, monkeypatch
):
    row = svc.ingest(pdf_bytes("delete"), "x.pdf", metadata())["document"]

    def fail(*args):
        raise RuntimeError("delete error")

    monkeypatch.setattr(svc.index, "delete", fail)
    with pytest.raises(RuntimeError):
        svc.delete(row["doc_id"])
    assert svc.ask("delete").status == "insufficient"
    restarted = RAGService(settings, index=local_index(settings), chat=FakeChat())
    assert not restarted.ledger.list()
    assert restarted.index.count(row["doc_id"]) == 0


def test_model_and_dimension_mismatch(svc, settings):
    for changes in ({"embedding_model": "another"}, {"embedding_dim": 16}):
        with pytest.raises(ValueError, match="禁止混用"):
            RAGService(replace(settings, **changes), index=svc.index)
