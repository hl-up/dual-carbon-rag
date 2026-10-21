# 评测自己的资料库

公开仓库只提供 `cases.example.jsonl` 的格式示例，不附带原项目的政策原件、文档 ID、答案记录或评测结果。示例中的 `doc_id` 和参考答案是占位内容，不能直接用于质量评估。

将示例复制为本地 `cases.jsonl`，然后按已核验并入库的 PDF 修改每题的 `doc_id`、PDF 页码、预期状态和参考答案。`cases.jsonl` 已被 `.gitignore` 排除。

```powershell
Copy-Item evaluation/cases.example.jsonl evaluation/cases.jsonl
uv run python scripts/evaluate.py --check
uv run python scripts/evaluate.py --check-corpus
uv run python scripts/evaluate.py --mode agent
```

`--check` 只检查题集格式；`--check-corpus` 检查本地已核验原件和页码；正式运行会调用配置的模型 API 并产生用量。报告默认写入被忽略的 `data/evaluation/results.json`，其中包含问题、回答和证据正文，应保留在本机。`--mode agent` 使用界面默认的智能体流程；不指定时使用直接检索流程。

每道题需包含唯一 `id`、`question`、`expected_status`（`answered`、`insufficient` 或 `out_of_scope`）和 `expected_evidence`。可回答题还需填写 `reference_answer`，证据条目使用入库后得到的 `doc_id` 与从 1 开始的 PDF 页序。可选字段包括 `category`、`scope`（`all`、`national`、`province`）和省级范围所需的 `province`。

脚本报告状态是否符合预期、目标页是否出现在候选和最终引用中，以及延迟。`--judge` 可额外调用模型判断事实支持与答案完整性；该判断仅供辅助。问答结论、地区与时间适用性仍应对照引用原文人工检查。
