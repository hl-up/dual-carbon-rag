"""Send exactly N requests at once to the local FastAPI app and summarize latency."""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

import httpx


async def measure(url, count, concurrency, question, mode):
    gate = asyncio.Event()
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=120) as client:

        async def one():
            await gate.wait()
            started = perf_counter()
            try:
                response = await client.post(
                    url,
                    json={"question": question, "scope": "national", "mode": mode},
                )
                return response.status_code, perf_counter() - started, response.text[:200]
            except httpx.HTTPError as exc:
                return type(exc).__name__, perf_counter() - started, ""

        tasks = [asyncio.create_task(one()) for _ in range(count)]
        gate.set()
        return await asyncio.gather(*tasks)


def summary(results):
    latencies = sorted(elapsed for _, elapsed, _ in results)
    counts = Counter(str(status) for status, _, _ in results)
    rate_limited = sum(
        1
        for status, _, body in results
        if status in {429, 502} and ("限流" in body or "额度不足" in body)
    )
    return {
        "requests": len(results),
        "status_counts": dict(counts),
        "provider_rate_limit_responses": rate_limited,
        "latency_seconds": {
            "min": round(latencies[0], 3),
            "p50": round(median(latencies), 3),
            "p95": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3),
            "max": round(latencies[-1], 3),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/ask")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--question", default="国家应对气候变化标准体系包含哪些一级子体系？")
    parser.add_argument("--mode", choices=["direct", "agent"], default="direct")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.count, args.concurrency) < 1:
        parser.error("count 和 concurrency 必须为正数")
    report = summary(
        asyncio.run(measure(args.url, args.count, args.concurrency, args.question, args.mode))
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
