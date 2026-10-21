"""Small conversational helpers; policy facts still require fresh PDF evidence."""

import re

OFF_TOPIC_REPLY = "我只回答已收录的双碳政策和相关资料问题。请换一个政策问题。"


def validate_question(question: str) -> tuple[str, str] | None:
    """Reject malformed and clearly unrelated requests before retrieval or model calls."""
    if not isinstance(question, str) or not question.strip():
        return "empty", "请输入有效问题"
    if len(question.strip()) > 500:
        return "invalid", "问题不能超过 500 字，请缩短后重试。"
    text = re.sub(r"\s+", "", question).casefold()
    unrelated = (
        r"^(?:请|帮我|给我|能否|你能)?(?:写|生成|编写).{0,12}(?:代码|脚本|程序|python|javascript)",
        r"^(?:请|帮我|给我|能否|你能)?(?:讲|说|写).{0,8}(?:笑话|故事|诗)",
        r"^(?:今天天气|明天天气|天气预报|股票价格|股价|体育比分|菜谱|食谱|法国首都|美国总统)",
        r"^(?:whatistheweather|tellmeajoke|write(?:a)?(?:python|javascript)(?:script|program)|recipefor)",
    )
    if any(re.search(pattern, text) for pattern in unrelated):
        return "out_of_scope", OFF_TOPIC_REPLY
    return None


def social_reply(question: str) -> str | None:
    """Handle only a complete social utterance, never a mixed policy question."""
    compact = re.sub(r"[\s，,。.!！?？~～]+", "", question).casefold()
    greetings = {
        "你好",
        "您好",
        "你好呀",
        "你好啊",
        "嗨",
        "哈喽",
        "hello",
        "hi",
        "早上好",
        "下午好",
        "晚上好",
        "在吗",
    }
    identity = {
        "你是谁",
        "你是什么",
        "介绍一下你自己",
        "请介绍一下你自己",
        "你叫什么名字",
        "你能做什么",
        "你会做什么",
        "你能帮我做什么",
        "你可以做什么",
        "你能回答什么问题",
        "怎么用",
    }
    if compact in greetings:
        return "你好！我是双碳政策资料助手。你可以问我政策要点、文件之间的差异，或让我帮你找到原文页码。"
    for greeting in greetings:
        if compact.startswith(greeting):
            compact = compact[len(greeting) :]
            break
    if compact.removeprefix("请问") in identity:
        return (
            "我是双碳政策资料助手。我可以在已核验的资料中查找政策原文、梳理要点，"
            "并附上可回查的文件和页码。你也可以接着上一问追问；资料里没有明确依据时，我会如实说明。"
        )
    if compact in {
        "谢谢",
        "谢谢你",
        "谢谢你的帮助",
        "感谢",
        "多谢",
        "辛苦了",
        "thanks",
        "thankyou",
    }:
        return "不客气！如果还想了解某条政策，直接接着问我就好。"
    if compact in {"再见", "拜拜", "bye", "goodbye"}:
        return "再见！需要查政策或核对原文时，随时来问我。"
    return None


def is_followup(question: str) -> bool:
    """Recognize short references to the previous policy question."""
    compact = re.sub(r"\s+", "", question).lstrip("，,。.!！?？")
    if not compact or len(compact) > 120 or "《" in compact:
        return False
    return compact.startswith(
        (
            "它",
            "他们",
            "它们",
            "这",
            "那",
            "该",
            "其",
            "上述",
            "前者",
            "后者",
            "第一份",
            "第二份",
            "这些",
            "还有",
            "另外",
            "再说说",
            "接着说",
            "什么时候",
            "何时",
            "谁发布",
            "适用范围",
            "生效日期",
            "施行日期",
        )
    )


def contextual_question(question: str, previous_question: str = "") -> str:
    """Use the prior question for reference resolution, never as factual evidence."""
    if not previous_question.strip() or not is_followup(question):
        return question
    return (
        f"上一轮问题（仅用于理解指代，不是事实依据）：{previous_question.strip()[:500]}\n"
        f"当前追问：{question.strip()}"
    )
