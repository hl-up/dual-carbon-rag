import pytest
from langchain_core.documents import Document


def seed(svc, metadata, pdf_bytes):
    for text, changes in [
        ("national evidence", {}),
        ("Shanghai evidence", {"jurisdiction": "省级", "province": "上海市"}),
        ("Guangdong evidence", {"jurisdiction": "省级", "province": "广东省"}),
        ("reference evidence", {"doc_type": "研究报告"}),
        ("pending evidence", {"verification_status": "待核验"}),
    ]:
        svc.ingest(pdf_bytes(text), "x.pdf", metadata(**changes))


def test_national_province_reference_and_pending_isolation(svc, metadata, pdf_bytes):
    seed(svc, metadata, pdf_bytes)
    result = svc.ask("test", scope="national")
    assert result.status == "answered"
    evidence = svc.chat.calls[-1][1]
    assert len(evidence) == 1 and evidence[0]["jurisdiction"] == "国家"
    svc.ask("test", scope="province", province="上海市")
    evidence = svc.chat.calls[-1][1]
    assert len(evidence) == 2
    assert {c["province"] for c in evidence} == {"", "上海市"}
    assert all(c["doc_type"] == "中国政策" for c in evidence)
    svc.ask("test", library="reference")
    assert all(c["doc_type"] == "研究报告" for c in svc.chat.calls[-1][1])
    assert svc.ask("test", scope="province", province="海南省").status == "insufficient"
    assert svc.ask("上海与海南的差别").status == "insufficient"


def test_keyword_candidate_recovers_page_missed_by_vector_search(
    svc, metadata, pdf_bytes, monkeypatch
):
    row = svc.ingest(
        pdf_bytes("unrelated energy text", "other industries may follow this guide"),
        "guide.pdf",
        metadata(),
    )["document"]
    wrong = Document(
        page_content="unrelated energy text",
        metadata={"doc_id": row["doc_id"], "page": 1, "chunk": 0},
        id=f"{row['doc_id']}:1:0",
    )
    monkeypatch.setattr(svc.index, "search", lambda *_: [(wrong, 0.2)])
    evidence = svc.retrieve("other industries may follow this guide", [row])
    assert evidence[0]["page"] == 2
    assert "other industries" in evidence[0]["excerpt"]


def test_metadata_edit_changes_filter_immediately(svc, metadata, pdf_bytes):
    row = svc.ingest(pdf_bytes("test"), "x.pdf", metadata())["document"]
    svc.update_metadata(row["doc_id"], metadata(jurisdiction="省级", province="上海市"))
    assert svc.ask("test", scope="national").status == "insufficient"
    assert svc.ask("test", scope="province", province="上海市").citations[0]["province"] == "上海市"
    svc.update_metadata(row["doc_id"], metadata(doc_type="研究报告"))
    assert svc.ask("test").status == "insufficient"
    assert svc.ask("test", library="reference").status == "answered"
    svc.update_metadata(row["doc_id"], metadata(verification_status="待核验"))
    assert svc.ask("test").status == "insufficient"


@pytest.mark.parametrize(
    "raw,status",
    [
        ('{"supported":false,"answer":"缺少依据"}', "insufficient"),
        ('{"supported":true,"answer":"事实 [99]"}', "review"),
        ('{"supported":true,"answer":"事实没有引用"}', "review"),
        ('{"supported":true,"answer":"事实 [1]\\n另一个无引用结论"}', "review"),
        ('{"supported":true,"answer":null}', "review"),
        ("[]", "review"),
        ("not JSON", "review"),
        ('{"supported":true,"answer":"事实 [1]"}', "answered"),
    ],
)
def test_insufficient_and_invalid_citations(svc, metadata, pdf_bytes, raw, status):
    svc.ingest(pdf_bytes("test evidence"), "x.pdf", metadata())
    svc.chat.result = raw
    result = svc.ask("test")
    assert result.status == status
    assert all(c["number"] != 99 for c in result.citations)


def test_invalid_citation_format_gets_one_safe_retry(svc, metadata, pdf_bytes, monkeypatch):
    svc.ingest(pdf_bytes("test evidence"), "x.pdf", metadata())
    requests = []

    def invoke(question, evidence):
        requests.append(question)
        answer = "没有引用" if len(requests) == 1 else "test evidence [1]"
        return f'{{"supported":true,"answer":"{answer}"}}', {}

    monkeypatch.setattr(svc.chat, "invoke", invoke)
    assert svc.ask("test evidence").status == "answered"
    assert len(requests) == 2
    assert "每个非空段落" in requests[1]


def test_cross_document_references(svc, metadata, pdf_bytes):
    svc.ingest(pdf_bytes("first"), "a.pdf", metadata(title="first"))
    svc.ingest(pdf_bytes("second"), "b.pdf", metadata(title="second"))
    svc.chat.result = '{"supported":true,"answer":"分别依据两份文件。[1][2]"}'
    result = svc.ask("compare")
    assert result.status == "answered"
    assert {c["title"] for c in result.citations} == {"first", "second"}


def test_exact_date_must_be_visible_in_cited_pdf_excerpt(svc):
    evidence = [
        {
            "number": 1,
            "title": "巴黎协定",
            "page": 14,
            "excerpt": "本协定在生效条件达到后第三十天起生效。",
            "source_note": "2016年11月4日生效",
        }
    ]
    svc.chat.result = '{"supported":true,"answer":"2016年11月4日生效。[1]"}'
    result = svc.answer_evidence("何时生效", evidence)
    assert result.status == "insufficient"
    assert "2016年11月4日" not in result.text
    evidence[0]["excerpt"] = "本协定于2016年11月4日生效。"
    assert svc.answer_evidence("何时生效", evidence).status == "answered"


def test_unfilled_effective_date_is_insufficient(svc):
    evidence = [{"number": 1, "excerpt": "本指南自 2024年□□月□□日起实施。"}]
    svc.chat.result = '{"supported":true,"answer":"本指南自2024年1月30日起实施。[1]"}'
    result = svc.answer_evidence("本指南从2024年1月30日起实施吗？", evidence)
    assert result.status == "insufficient"
    assert "2024年1月30日" not in result.text


def test_empty_no_model_calls(svc):
    assert svc.ask("   ").status == "empty"
    assert not svc.chat.calls


def test_chat_failure_does_not_expose_secrets(svc, metadata, pdf_bytes, settings):
    svc.ingest(pdf_bytes("test"), "x.pdf", metadata())
    svc.chat.error = RuntimeError(settings.api_key + " sensitive text")
    result = svc.ask("private question")
    assert result.status == "error"
    assert len(result.retrieved) == 1
    with svc.ledger.connection() as db:
        operations = str([tuple(r) for r in db.execute("SELECT * FROM operations")])
    for forbidden in (settings.api_key, "sensitive text", "private question"):
        assert forbidden not in operations + result.text


def test_coverage(svc, metadata, pdf_bytes):
    seed(svc, metadata, pdf_bytes)
    coverage = {r["地区"]: r for r in svc.ledger.coverage()}
    assert coverage["上海市"]["可检索"] == 1
    assert coverage["国家"]["待核验"] == 1
    assert coverage["海南省"]["状态"] == "未收录"
    assert coverage["国家"]["最近核验时间"]
