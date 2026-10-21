"""Chat model adapter and provider error mapping."""

from threading import Lock

from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from .prompts import answer_messages


def safe_error(exc):
    if isinstance(exc, AuthenticationError):
        return "API 鉴权失败，请检查本地密钥与模型权限"
    if isinstance(exc, RateLimitError):
        return "API 限流或额度不足，请检查账户后重试"
    if isinstance(exc, APITimeoutError):
        return "API 请求超时，请稍后重试"
    if isinstance(exc, APIConnectionError):
        return "无法连接 API，请检查网络和 Base URL"
    return "操作失败，请检查文件、模型权限或网络后重试（诊断仅记录错误类型）"


def make_chat_model(settings, json_output=False):
    settings.require_api(chat=True)
    options = {}
    if settings.chat_model == "deepseek-flash" and settings.chat_base_url:
        options["extra_body"] = {"thinking": {"type": "disabled"}}
    model = ChatOpenAI(
        api_key=settings.chat_api_key if settings.chat_base_url else settings.api_key,
        base_url=settings.chat_base_url or settings.base_url,
        model=settings.chat_model,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        **options,
    )
    if json_output and settings.chat_model == "deepseek-flash":
        return model.bind(response_format={"type": "json_object"})
    return model


class AnswerModel:
    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._lock = Lock()

    def invoke(self, question, evidence):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = make_chat_model(self.settings, json_output=True)
        response = self._model.invoke(answer_messages(question, evidence))
        return response.content, response.usage_metadata or {}
