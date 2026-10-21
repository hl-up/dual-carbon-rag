from carbon_rag.tools import PolicyTools


def test_tool_layer_scopes_documents_and_deduplicates(svc, metadata, pdf_bytes):
    national = svc.ingest(pdf_bytes("national evidence"), "a.pdf", metadata())["document"]
    other = svc.ingest(
        pdf_bytes("other evidence"),
        "b.pdf",
        metadata(jurisdiction="省级", province="广东省"),
    )["document"]
    rows = svc.ledger.eligible(scope="national")
    evidence = []
    toolkit = PolicyTools(svc, rows, evidence, max_evidence=2)

    listed = toolkit.tools["list_documents"].invoke({})
    assert [item["doc_id"] for item in listed] == [national["doc_id"]]
    assert toolkit.tools["search_documents"].invoke(
        {"query": "test", "doc_id": other["doc_id"]}
    ) == [{"error": "文档不在当前允许范围"}]

    first = toolkit.tools["search_documents"].invoke({"query": "national"})
    second = toolkit.tools["search_documents"].invoke({"query": "national"})
    assert first[0]["number"] == second[0]["number"] == 1
    assert len(evidence) == 1


def test_tool_layer_validates_page_and_query(svc, metadata, pdf_bytes):
    row = svc.ingest(pdf_bytes("page one", "page two"), "a.pdf", metadata())["document"]
    toolkit = PolicyTools(svc, svc.ledger.eligible(), [], max_evidence=3)
    assert toolkit.tools["search_documents"].invoke({"query": " "}) == [
        {"error": "检索词应为1到1000字"}
    ]
    assert toolkit.tools["read_pdf_page"].invoke({"doc_id": row["doc_id"], "page": 3}) == [
        {"error": "文档或PDF页码不在允许范围"}
    ]
    result = toolkit.tools["read_pdf_page"].invoke({"doc_id": row["doc_id"], "page": 2})
    assert result[0]["excerpt"] == "page two"
