"""Send one real Agent run and verify its LangSmith trace, or inspect an existing run."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langsmith import Client, trace

from carbon_rag.agent import PolicyAgent
from carbon_rag.config import Settings
from carbon_rag.service import RAGService


async def verify_trace(client, project_name, run_id):
    project = await client.aread_project(project_name=project_name)
    for _ in range(8):
        try:
            trace_runs = await client.traces.list_runs(
                str(run_id),
                project_id=str(project.id),
                selects=[
                    "ID",
                    "NAME",
                    "RUN_TYPE",
                    "STATUS",
                    "PARENT_RUN_IDS",
                    "TOTAL_TOKENS",
                    "LATENCY_SECONDS",
                ],
            )
            items = trace_runs.items or []
            descendants = [
                item
                for item in items
                if run_id in {UUID(str(parent)) for parent in (item.parent_run_ids or [])}
            ]
            if descendants:
                llm_runs = [item for item in descendants if (item.run_type or "").lower() == "llm"]
                tool_runs = sum((item.run_type or "").lower() == "tool" for item in descendants)
                llm_tokens = sum(item.total_tokens or 0 for item in llm_runs)
                print(
                    f"LangSmith 已收到追踪：project={project_name} run_id={run_id} "
                    f"descendants={len(descendants)} llm_runs={len(llm_runs)} "
                    f"tool_runs={tool_runs} llm_tokens={llm_tokens}"
                )
                return 0
        except Exception:
            pass
        await asyncio.sleep(1)
    raise SystemExit(
        "模型执行完成，但 8 秒内未从 LangSmith 读到子运行；请检查密钥、Endpoint 和项目权限"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="仅核验既有追踪，不再调用模型")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("请先在 .env 填写 LANGSMITH_API_KEY；未调用任何远端接口")
    project = os.getenv("LANGSMITH_PROJECT", "dual-carbon-rag").strip()
    if not project:
        raise SystemExit("LANGSMITH_PROJECT 不能为空")
    os.environ["LANGSMITH_TRACING"] = "true"
    client = Client()
    if args.run_id:
        return asyncio.run(verify_trace(client, project, UUID(args.run_id)))
    settings = Settings.from_env()
    settings.require_api(chat=True)
    question = "国家应对气候变化标准体系包含哪些一级子体系？"
    with trace(
        "langsmith_agent_verification",
        inputs={"question": question},
        project_name=project,
        client=client,
    ) as run:
        result = PolicyAgent(RAGService(settings)).ask(question, scope="national")
        run.end(outputs={"status": result.status, "citations": len(result.citations)})
    print(f"Agent 答复状态：{result.status}")
    return asyncio.run(verify_trace(client, project, run.id))


if __name__ == "__main__":
    raise SystemExit(main())
