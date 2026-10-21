# 双碳政策资料问答

一个可本地运行、也可部署为公开只读站点的政策资料检索与问答项目。程序按页提取 PDF 文字，使用嵌入模型建立 Chroma 索引，以 SQLite 记录文件与核验状态。问答可使用直接检索或带 `list_documents`、`search_documents`、`read_pdf_page` 工具的智能体；回答展示可回查的 PDF 页码和原文片段。

仓库包含公开资料快照 `public_data/`：57 份已入库 PDF、对应的台账和检索索引。其中 56 份已核验、可参与问答；1 份待核验原件仅供下载查阅。仓库还提供程序、离线测试、依赖锁文件和配置示例。**不包含本机工作目录 `data/`、其他候选资料、DOC/DOCX、原始工作簿、评测答案或 API Key。**

## 功能与边界

- PDF 签名与大小检查、按页解析、异常页确认、切片、嵌入、哈希去重和本地持久化。
- 文件台账支持修改元数据、撤销核验、删除和失败后重试；默认问答范围只使用已核验且入库完成的文件。
- 按国家、省级地区和资料类型限制检索；中国政策与参考资料分开选择。
- 答案附编号引用、原文和 PDF 页序；证据不足时返回相应状态。
- Streamlit 界面，以及带进程内并发限制的 FastAPI 问答接口。

当前解析器处理可提取文字的 PDF。整份扫描件需要 OCR；DOC/DOCX 转换和复杂版式校核尚未实现。引用编号检查无法单独证明政策结论正确，重要结论应由使用者核对原文、地区、日期与效力。本地管理界面面向单用户使用，不提供多用户鉴权，也不支持多个进程同时写入同一数据目录。公开站点只开放资料目录与问答，访客不能上传、修改或删除台账内容。

## 公开站点与访客使用

公开站点的入口为 `deploy/streamlit_app.py`，由 Streamlit Community Cloud 托管。访客可直接浏览目录并下载 57 份原件。问答需要访客在网页侧栏填写自己的 OpenAI 嵌入接口密钥，以及 DeepSeek 或 OpenAI 聊天接口密钥。密钥保存在当前 Streamlit 会话的服务器内存中，不写入 Git、数据库或日志；关闭会话或点击“清除本次会话密钥”后失效。访客应自行管理接口用量和预算。

公开索引使用 `text-embedding-3-small`、1536 维。访客的嵌入接口必须与该索引兼容。提问和检索到的原文片段会发送到访客选择的模型服务。Streamlit Community Cloud 的运行时存储为临时存储；仓库中的资料快照会在重新部署时恢复。

站点部署完成后，维护者应在此处填写实际访问地址并检查匿名访问。

## 从空白环境启动

需要 Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)。以下命令在项目根目录执行；Windows PowerShell 示例。下面的本地管理界面使用独立的 `data/` 目录，首次启动为空，不会修改公开资料快照。

```powershell
uv sync --frozen
Copy-Item .env.example .env
```

在本机编辑 `.env`，填写自己的嵌入接口和聊天接口密钥。`CLOSEAI_*` 是代码中嵌入接口的配置名，接口需兼容 OpenAI embeddings API；聊天接口使用 `DEEPSEEK_*` 配置。`.env.example` 只有空白密钥字段，不要把真实密钥写入模板或提交到 Git。

```dotenv
CLOSEAI_API_KEY=你的嵌入接口密钥
CLOSEAI_BASE_URL=https://你的嵌入接口地址/v1
DEEPSEEK_API_KEY=你的聊天接口密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-flash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
DATA_DIR=data
```

模型 ID、权限和返回维度须与实际账户一致。首次联调可以运行 `uv run python scripts/smoke_api.py`；这会发送合成测试文字并产生少量 API 用量。随后启动界面：

```powershell
uv run python -m streamlit run app.py --server.address 127.0.0.1
```

打开 <http://127.0.0.1:8501>。在“资料管理”页上传自己的 PDF，先查看解析页数及异常页，登记标题、发布机构、日期、来源链接、类型和地区。只有核对原件与来源后才确认“已核验”。入库后可在问答页选择范围，查看答案及逐条引用。密钥为空时仍可打开界面和本地预览，但入库与问答需要对应的接口。

## API、Docker 与测试

FastAPI 入口为 `carbon_rag.api:app`，仅建议在可信本机或受保护的内网使用；接口本身没有用户鉴权。直接运行时可用：

```powershell
uv run uvicorn carbon_rag.api:app --host 127.0.0.1 --port 8000
```

也可在配置好本地 `.env` 后运行 `docker compose up --build -d`。Compose 将容器端口映射到本机 `127.0.0.1:8501`，并把本地 `data/` 挂载为持久存储。不要将未加鉴权的问答接口直接暴露到公网。

离线测试使用合成 PDF、模型替身和本地临时目录，不需要正式政策语料或付费 API：

```powershell
uv run pytest -q
uv run ruff check app.py carbon_rag scripts tests
uv run ruff format --check app.py carbon_rag scripts tests
uv run python scripts/evaluate.py evaluation/cases.example.jsonl --check
```

如何为自己的资料建立题集、核对页码并运行真实模型评测，见 [评测说明](evaluation/README.md)。`evaluation/cases.example.jsonl` 只是格式示例，不是政策事实题集。

## 部署自己的公开站点

1. 将仓库推送到自己的 GitHub 账户，确认 `public_data/` 的资料有权公开。不要把 `.env` 或本地 `data/` 提交到仓库。
2. 在 [Streamlit Community Cloud](https://share.streamlit.io/) 用 GitHub 登录并创建应用，仓库选自己的副本，分支选 `main`，入口文件填 `deploy/streamlit_app.py`，Python 版本选 3.12。
3. 此公开版不需要在 Cloud Secrets 中填写站长的 API Key。部署后打开应用，检查目录、PDF 下载、访客密钥表单和问答。应用只在服务端保存访客当前会话的密钥；不要把密钥放进 URL、截图或 Git 提交。

`deploy/requirements.txt` 是云端入口的精简依赖。`.streamlit/config.toml` 不绑定本机地址，使云端可以监听其分配的端口。要更新公开资料，先在本地完成来源核验和入库，再运行 `uv run python -m scripts.build_public_corpus` 生成新快照；脚本会拒绝覆盖已有的 `public_data/`，请先备份并人工移走旧快照，再检查新快照与 `git status` 后提交。

## 数据与密钥

`.gitignore` 排除了 `.env`、本地资料目录 `pdf/`、本地数据库与向量索引 `data/`、本地评测报告及生成的 `docs/`；仅为人工核验后生成的 `public_data/` 开放了 PDF、台账和索引文件。提交前仍应运行 `git status --short` 和 `git diff --cached --name-only`，确认没有意外加入新位置的密钥、原件或个人数据。不要把密钥写进源代码、Issue、日志或截图。若密钥曾进入公开提交，即使随后删掉文件，也应在服务商处轮换该密钥。

问答会将问题和所选证据片段发送给聊天接口；入库会将 PDF 提取文字发送给嵌入接口。可选的 LangSmith 追踪关闭时不发送追踪数据；开启前应确认资料允许传输到该服务。停止程序后备份整个 `data/`，以便同时保留 SQLite、Chroma 和入库原件。更换嵌入模型或维度时应使用新的 `DATA_DIR` 并重建索引。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `app.py` | Streamlit 界面 |
| `deploy/streamlit_app.py` | 公开只读站点入口，访客自带模型密钥 |
| `public_data/` | 公开 PDF、去除运行日志的台账与检索索引 |
| `carbon_rag/` | 入库、检索、问答、台账和 API |
| `scripts/` | 接口联调、公开快照生成、资料盘点及评测工具；部分清单导入脚本需要自行提供对应工作簿 |
| `tests/` | 不使用真实密钥的离线测试 |
| `.env.example` | 本地配置模板 |

本仓库暂未添加开源许可证。公开可见不等于授予复制、修改或分发许可；若计划开放这些权限，需要由仓库所有者另行选择许可证。
