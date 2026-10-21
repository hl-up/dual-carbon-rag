"""Public, read-only entrypoint for Streamlit Community Cloud."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import question_page
from carbon_rag.config import Settings
from carbon_rag.ledger import Ledger
from carbon_rag.service import RAGService

DATA_DIR = Path(__file__).resolve().parents[1] / "public_data"


def clear_credentials():
    for key in ("embed_key", "chat_key", "chat_provider", "chat_model", "public_service"):
        st.session_state.pop(key, None)


def document_page(ledger):
    st.subheader("资料目录")
    rows = ledger.list()
    st.caption(f"共 {len(rows)} 份 PDF；待核验文件可下载查看，但不参与问答。")
    library = st.selectbox("资料类型", ["全部", "中国政策", "参考资料"])
    if library == "中国政策":
        rows = [row for row in rows if row["metadata"]["doc_type"] == "中国政策"]
    elif library == "参考资料":
        rows = [row for row in rows if row["metadata"]["doc_type"] != "中国政策"]
    rows.sort(key=lambda row: (row["metadata"]["title"], row["doc_id"]))
    if not rows:
        st.info("该类别暂无资料。")
        return
    selected = st.selectbox(
        "选择文件",
        range(len(rows)),
        format_func=lambda i: rows[i]["metadata"]["title"],
    )
    row = rows[selected]
    meta = row["metadata"]
    st.write(f"**标题：**{meta['title']}")
    st.write(f"**发布机构：**{meta['issuer']}　**发布日期：**{meta['publish_date']}")
    st.write(f"**类型：**{meta['doc_type']}　**核验状态：**{meta['verification_status']}")
    if meta["source_url"]:
        st.link_button("查看来源发布页", meta["source_url"])
    pdf_path = DATA_DIR / "originals" / f"{row['doc_id']}.pdf"
    if pdf_path.is_file():
        st.download_button(
            "下载入库原件",
            pdf_path.read_bytes(),
            file_name=row["filename"],
            mime="application/pdf",
        )
    else:
        st.error("原件暂不可用。")


def credentials_form():
    with st.sidebar:
        st.subheader("本次会话的模型接口")
        st.caption("密钥仅供您当前会话调用接口；本网站不提供共享密钥。")
        with st.form("visitor_credentials"):
            embed_key = st.text_input("嵌入接口 API Key", type="password", key="embed_key")
            st.caption("嵌入接口使用 OpenAI 官方 API。")
            chat_provider = st.selectbox("聊天接口", ["DeepSeek", "OpenAI"], key="chat_provider")
            chat_key = st.text_input("聊天接口 API Key", type="password", key="chat_key")
            chat_model = st.text_input(
                "聊天模型 ID",
                value="deepseek-chat" if chat_provider == "DeepSeek" else "gpt-4o-mini",
                key="chat_model",
            )
            save = st.form_submit_button("启用本次会话问答", type="primary")
        st.caption(
            "嵌入模型须兼容 text-embedding-3-small（1536 维）。问题与召回片段会发送至您填写的接口。"
        )
        st.button("清除本次会话密钥", on_click=clear_credentials)
    if save:
        if not embed_key.strip() or not chat_key.strip():
            st.sidebar.error("请填写两个接口的 API Key。")
            return st.session_state.get("public_service")
        settings = replace(
            Settings(data_dir=DATA_DIR),
            api_key=embed_key.strip(),
            base_url="https://api.openai.com/v1",
            chat_api_key=chat_key.strip(),
            chat_base_url=(
                "https://api.deepseek.com"
                if chat_provider == "DeepSeek"
                else "https://api.openai.com/v1"
            ),
            chat_model=chat_model.strip(),
        )
        try:
            settings.require_api(chat=True)
            st.session_state["public_service"] = RAGService(settings)
            st.session_state.pop("policy_history", None)
        except ValueError as exc:
            st.sidebar.error(str(exc))
    return st.session_state.get("public_service")


def main():
    st.title("🌱 双碳政策资料问答")
    st.caption("公开资料库 · 按核验状态检索 · 引用可回查")
    if not (DATA_DIR / "ledger.sqlite3").is_file():
        st.error("公开资料快照尚未配置。")
        st.stop()
    ledger = Ledger(DATA_DIR / "ledger.sqlite3")
    svc = credentials_form()
    page = st.sidebar.radio("功能", ["资料目录", "政策问答"])
    st.sidebar.metric("可检索政策", len(ledger.eligible()))
    if page == "资料目录":
        document_page(ledger)
    elif svc is None:
        st.info("请在左侧填写自己的模型接口密钥后开始问答。您仍可浏览和下载资料目录。")
    else:
        question_page(svc)


if __name__ == "__main__":
    main()
