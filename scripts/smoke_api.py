"""仅显式运行时调用真实 API；不打印密钥和接口异常正文。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_openai import OpenAIEmbeddings

from carbon_rag.config import Settings
from carbon_rag.providers import AnswerModel, safe_error


def main():
    settings = Settings.from_env()
    try:
        settings.require_api(chat=True)
    except ValueError as exc:
        print(str(exc))
        return 1
    try:
        embedding = OpenAIEmbeddings(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dim,
            check_embedding_ctx_length=False,
            request_timeout=settings.timeout,
            max_retries=0,
        )
        result = embedding.embed_query("双碳政策接口连接测试")
        if len(result) != settings.embedding_dim:
            print("嵌入维度与配置不符，请核验模型和维度后重建索引。")
            return 1
        print(f"嵌入调用成功：{settings.embedding_model}，维度 {len(result)}")
        raw, usage = AnswerModel(settings).invoke(
            "测试材料说明什么？",
            [
                {
                    "number": 1,
                    "title": "合成联调材料（非政策）",
                    "excerpt": "本材料仅用于验证接口连接。",
                },
            ],
        )
        import json

        answer = json.loads(raw)
        if answer.get("supported") is not True or "[1]" not in answer.get("answer", ""):
            print("聊天接口可连接，但输出未通过 JSON/引用最小校验。")
            return 1
        print(f"聊天调用成功：{settings.chat_model}；输出格式通过；用量 {usage}")
        return 0
    except Exception as exc:
        print(safe_error(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
