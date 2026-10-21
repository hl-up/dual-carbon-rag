from langchain_core.messages import AIMessage

from carbon_rag.agent import PolicyAgent


class FakePlanner:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.messages.append(list(messages))
        return next(self.replies)


def reply(name, args=None, identifier="call1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": identifier}])


def test_agent_tools_and_references(svc, metadata, pdf_bytes):
    row = svc.ingest(pdf_bytes("test evidence"), "a.pdf", metadata())["document"]
    planner = FakePlanner(
        [
            reply("list_documents"),
            reply("search_documents", {"query": "test", "doc_id": row["doc_id"]}, "call2"),
            AIMessage(content="ready"),
        ]
    )
    result = PolicyAgent(svc, planner=planner).ask("test", scope="national")
    assert result.status == "answered"
    assert [step["tool"] for step in result.steps] == ["list_documents", "search_documents"]
    assert result.citations[0]["doc_id"] == row["doc_id"]
    assert result.citations[0]["page"] == 1
    assert {t.name for t in planner.tools} == {
        "list_documents",
        "search_documents",
        "read_pdf_page",
    }


def test_agent_reads_real_pdf_page(svc, metadata, pdf_bytes):
    row = svc.ingest(pdf_bytes("Page one", "Page two evidence"), "a.pdf", metadata())["document"]
    planner = FakePlanner(
        [reply("read_pdf_page", {"doc_id": row["doc_id"], "page": 2}), AIMessage(content="done")]
    )
    result = PolicyAgent(svc, planner=planner).ask("second page")
    assert result.status == "answered"
    assert result.citations[0]["page"] == 2
    assert result.citations[0]["excerpt"] == "Page two evidence"


def test_model_direct_answer_never_bypasses_retrieval(svc, metadata, pdf_bytes):
    svc.ingest(pdf_bytes("real evidence"), "a.pdf", metadata())
    planner = FakePlanner([AIMessage(content="invented direct answer")])
    result = PolicyAgent(svc, planner=planner).ask("test")
    assert result.status == "answered" and "invented" not in result.text
    assert result.steps[0]["fallback"]
    assert result.citations


def test_region_and_pending_cannot_be_accessed_through_tools(svc, metadata, pdf_bytes):
    national = svc.ingest(pdf_bytes("national"), "a.pdf", metadata())["document"]
    foreign = svc.ingest(
        pdf_bytes("other region"), "b.pdf", metadata(jurisdiction="省级", province="广东省")
    )["document"]
    planner = FakePlanner(
        [
            reply("read_pdf_page", {"doc_id": foreign["doc_id"], "page": 1}),
            reply("search_documents", {"query": "region", "doc_id": foreign["doc_id"]}, "call2"),
            AIMessage(content="stop"),
        ]
    )
    result = PolicyAgent(svc, planner=planner).ask("test", scope="national")
    assert all(step["status"] == "error" for step in result.steps[:2])
    assert all(c["doc_id"] == national["doc_id"] for c in result.retrieved)


def test_unknown_tools_and_call_limit(svc, metadata, pdf_bytes):
    svc.ingest(pdf_bytes("test"), "a.pdf", metadata())
    planner = FakePlanner([reply("delete_document") for _ in range(10)])
    result = PolicyAgent(svc, planner=planner, max_rounds=10, max_calls=2).ask("test")
    assert len(result.steps) == 2
    assert result.status == "insufficient"
    assert len(svc.ledger.list()) == 1


def test_agent_empty_and_missing_region_do_not_call_model(svc):
    planner = FakePlanner([])
    agent = PolicyAgent(svc, planner=planner)
    assert agent.ask(" ").status == "empty"
    assert agent.ask("test", scope="province", province="上海市").status == "insufficient"
    assert not planner.messages


def test_agent_deduplicates_repeated_evidence(svc, metadata, pdf_bytes):
    svc.ingest(pdf_bytes("test"), "a.pdf", metadata())
    planner = FakePlanner(
        [
            reply("search_documents", {"query": "test"}),
            reply("search_documents", {"query": "test"}, "call2"),
            AIMessage(content="done"),
        ]
    )
    result = PolicyAgent(svc, planner=planner).ask("test")
    assert len(result.retrieved) == 1
    assert result.retrieved[0]["number"] == 1
