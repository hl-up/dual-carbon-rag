"""有次数上限的只读政策检索智能体；所有证据仍由后端读取与编号。"""

import json
from time import perf_counter

from langchain_core.messages import ToolMessage
from langsmith import traceable
from pydantic import ValidationError

from .conversation import contextual_question, social_reply, validate_question
from .prompts import agent_messages
from .providers import make_chat_model, safe_error
from .service import Answer
from .tools import PolicyTools


class PolicyAgent:
    def __init__(self, service, planner=None, max_rounds=3, max_calls=6, max_evidence=12):
        if min(max_rounds, max_calls, max_evidence) < 1:
            raise ValueError("智能体轮数、工具次数和证据上限必须大于零")
        self.service = service
        self.planner = planner
        self.max_rounds, self.max_calls, self.max_evidence = max_rounds, max_calls, max_evidence

    @traceable(run_type="chain", name="policy_agent")
    def ask(self, question, library="policy", scope="all", province="", previous_question=""):
        svc = self.service
        started = perf_counter()
        issue = validate_question(question)
        if issue is not None:
            result = Answer(*issue, elapsed=perf_counter() - started)
            svc.ledger.record("agent_ask", result.status, result.elapsed, details={"tool_calls": 0})
            return result
        reply = social_reply(question)
        if reply is not None:
            result = Answer("social", reply, elapsed=perf_counter() - started)
            svc.ledger.record("agent_ask", "social", result.elapsed, details={"tool_calls": 0})
            return result
        effective_question = contextual_question(question, previous_question)
        rows, stopped = svc.scope_documents(effective_question, library, scope, province)
        if stopped is not None:
            return stopped
        steps, evidence = [], []
        try:
            result = self._run(effective_question, rows, evidence, steps)
        except Exception as exc:
            svc.ledger.record("agent", "error", details={"error_type": type(exc).__name__})
            result = Answer("error", safe_error(exc), retrieved=evidence)
        result.steps = steps
        result.elapsed = perf_counter() - started
        svc.ledger.record(
            "agent_ask",
            result.status,
            result.elapsed,
            details={
                "model": svc.settings.chat_model,
                "tool_calls": len(steps),
                "evidence_count": len(evidence),
            },
        )
        return result

    def _run(self, question, rows, evidence, steps):
        svc = self.service
        toolkit = PolicyTools(svc, rows, evidence, self.max_evidence)
        tools = toolkit.tools
        planner = self.planner
        if planner is None:
            s = svc.settings
            s.require_api(chat=True)
            planner = make_chat_model(s)
        bound = planner.bind_tools(list(tools.values()))
        messages = agent_messages(question)
        call_count = 0
        for _ in range(self.max_rounds):
            model_started = perf_counter()
            response = bound.invoke(messages)
            usage = response.usage_metadata or {}
            svc.ledger.record(
                "agent_usage",
                "received",
                perf_counter() - model_started,
                details={
                    "model": svc.settings.chat_model,
                    **{
                        k: usage[k]
                        for k in ("input_tokens", "output_tokens", "total_tokens")
                        if k in usage
                    },
                },
            )
            messages.append(response)
            calls = response.tool_calls
            if not calls:
                break
            for call in calls:
                if call_count >= self.max_calls:
                    break
                name, args = call["name"], call["args"]
                call_count += 1
                tool_started = perf_counter()
                if name not in tools:
                    output = [{"error": "工具不在允许列表"}]
                else:
                    try:
                        output = tools[name].invoke(args)
                    except (ValidationError, TypeError):
                        output = [{"error": "工具参数无效，请按工具说明填写"}]
                steps.append(
                    {
                        "tool": name,
                        "query": args.get("query", ""),
                        "doc_id": args.get("doc_id", ""),
                        "page": args.get("page"),
                        "result_count": sum(1 for item in output if "error" not in item),
                        "status": "error" if any("error" in item for item in output) else "ok",
                        "elapsed": perf_counter() - tool_started,
                    }
                )
                messages.append(
                    ToolMessage(
                        content=json.dumps(output, ensure_ascii=False), tool_call_id=call["id"]
                    )
                )
            if call_count >= self.max_calls or len(evidence) >= self.max_evidence:
                break
        # 模型只返回文字/只查目录时，强制真正检索；不采用模型无依据的直接回答。
        if not evidence and call_count >= self.max_calls:
            return Answer(
                "insufficient", "我还没找到能确认这个问题的原文依据。可以换个问法或补充相关资料。"
            )
        if not evidence:
            toolkit.tools["search_documents"].invoke({"query": question})
            steps.append(
                {
                    "tool": "search_documents",
                    "query": question,
                    "result_count": len(evidence),
                    "status": "ok",
                    "fallback": True,
                }
            )
        return svc.answer_evidence(question, evidence)
