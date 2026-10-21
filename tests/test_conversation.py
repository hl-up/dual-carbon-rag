from langchain_core.messages import AIMessage

from carbon_rag.agent import PolicyAgent
from carbon_rag.conversation import contextual_question, is_followup, social_reply


def test_social_intent_is_only_for_standalone_utterances():
    assert "双碳政策资料助手" in social_reply("你好！")
    assert "原文" in social_reply("你是谁？")
    assert "不客气" in social_reply("谢谢")
    assert "双碳政策资料助手" in social_reply("你好，请问你是谁？")
    assert social_reply("你好，国家应对气候变化标准体系建设方案何时施行？") is None
    assert social_reply("谢谢，下一步如何申请？") is None


def test_social_reply_needs_no_documents_or_model(svc):
    for question in ("你好", "你是谁", "谢谢"):
        result = svc.ask(question)
        assert result.status == "social"
        assert not result.citations
        assert not svc.chat.calls
    assert svc.ask("你好，双碳政策有哪些？").status == "insufficient"


def test_agent_social_reply_skips_planner_and_retrieval(svc):
    class NoPlanner:
        def bind_tools(self, tools):
            raise AssertionError("社交话语不应进入工具规划")

    result = PolicyAgent(svc, planner=NoPlanner()).ask("您好")
    assert result.status == "social"
    assert result.steps == []


def test_followup_uses_previous_question_as_context_only(svc, metadata, pdf_bytes):
    svc.ingest(pdf_bytes("National evidence"), "a.pdf", metadata())
    previous = "国家应对气候变化标准体系建设方案有什么目标？"
    result = svc.ask("它什么时候施行？", previous_question=previous)
    assert result.status == "answered"
    sent_question, sent_evidence = svc.chat.calls[-1]
    assert previous in sent_question and "它什么时候施行" in sent_question
    assert all(previous not in item["excerpt"] for item in sent_evidence)
    assert is_followup("它什么时候施行？")
    assert contextual_question("《另一份文件》何时施行？", previous) == "《另一份文件》何时施行？"


def test_agent_planner_receives_followup_context_but_must_search(svc, metadata, pdf_bytes):
    svc.ingest(pdf_bytes("National evidence"), "a.pdf", metadata())

    class DirectAnswerPlanner:
        def __init__(self):
            self.messages = []

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content="无依据的直接回答")

    planner = DirectAnswerPlanner()
    previous = "国家应对气候变化标准体系建设方案有什么目标？"
    result = PolicyAgent(svc, planner=planner).ask("它何时施行？", previous_question=previous)
    assert result.status == "answered"
    assert result.citations
    assert result.steps[0]["fallback"]
    assert previous in planner.messages[1].content
    assert "无依据" not in result.text
