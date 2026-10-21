from carbon_rag.agent import PolicyAgent
from carbon_rag.conversation import validate_question


def test_clear_off_topic_input_is_rejected_before_any_model_call(svc):
    for question in ("今天天气怎么样？", "请写一个Python脚本爬取股票价格。", "Tell me a joke"):
        assert svc.ask(question).status == "out_of_scope"
        assert PolicyAgent(svc).ask(question).status == "out_of_scope"
    assert not svc.chat.calls


def test_policy_question_with_weather_word_is_not_rejected():
    assert validate_question("极端天气相关的气候适应政策有哪些？") is None
    assert validate_question("《国家应对气候变化标准体系建设方案》有哪些要求？") is None


def test_empty_and_oversized_input_is_rejected_without_retrieval(svc):
    assert svc.ask(" ").status == "empty"
    assert svc.ask("政策" * 251).status == "invalid"
    assert not svc.chat.calls
