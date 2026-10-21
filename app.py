import hashlib
import json

import streamlit as st

from carbon_rag.agent import PolicyAgent
from carbon_rag.config import Settings
from carbon_rag.conversation import is_followup
from carbon_rag.models import DOC_TYPES, PROVINCES, Metadata
from carbon_rag.pdf import parse_pdf
from carbon_rag.service import PartialPagesError, RAGService

st.set_page_config(page_title="双碳政策 · 资料问答", page_icon="🌱", layout="wide")


@st.cache_resource
def service(settings):
    return RAGService(settings)


def metadata_form(prefix, initial=None):
    m = initial or Metadata("", "", "", "")
    title = st.text_input("标题 *", m.title, key=f"{prefix}_title")
    left, right = st.columns(2)
    with left:
        issuer = st.text_input("发布机构 *", m.issuer, key=f"{prefix}_issuer")
        published = st.text_input(
            "发布日期 *（YYYY、YYYY-MM 或 YYYY-MM-DD）", m.publish_date, key=f"{prefix}_date"
        )
        kind = st.selectbox(
            "资料类型 *", DOC_TYPES, index=DOC_TYPES.index(m.doc_type), key=f"{prefix}_type"
        )
        jurisdictions = ["国家", "省级", "国际"]
        jurisdiction = st.selectbox(
            "范围 *",
            jurisdictions,
            index=jurisdictions.index(m.jurisdiction),
            key=f"{prefix}_jurisdiction",
        )
        regions = ["", *PROVINCES]
        province = st.selectbox(
            "省级地区（国家、国际文件留空）",
            regions,
            index=regions.index(m.province),
            key=f"{prefix}_province",
        )
    with right:
        source = st.text_input("来源发布页 URL *", m.source_url, key=f"{prefix}_source")
        number = st.text_input("文号", m.doc_number, key=f"{prefix}_number")
        effective = st.text_input(
            "施行日期（可选，YYYY-MM-DD）", m.effective_date, key=f"{prefix}_effective"
        )
        validity = st.text_input(
            "人工核验的效力说明（未知则留空）", m.validity_note, key=f"{prefix}_validity"
        )
        translated = st.checkbox("此文件为译文", m.is_translation, key=f"{prefix}_translated")
    language = st.text_input("语言", m.language, key=f"{prefix}_language")
    source_note = st.text_input(
        "来源/版本备注（网页转换 PDF、同名修订关系等）", m.source_note, key=f"{prefix}_note"
    )
    verified = st.checkbox(
        "我已核对发布来源与 PDF 正文，并确认上述元数据",
        m.verification_status == "已核验",
        key=f"{prefix}_verified",
    )
    return Metadata(
        title=title,
        issuer=issuer,
        publish_date=published,
        source_url=source,
        doc_type=kind,
        jurisdiction=jurisdiction,
        province=province,
        verification_status="已核验" if verified else "待核验",
        doc_number=number,
        effective_date=effective,
        validity_note=validity,
        is_translation=translated,
        language=language,
        source_note=source_note,
    )


def question_page(svc):
    st.subheader("双碳政策助手 · 可以对话，也能回到原文")
    mode = st.radio("问答方式", ["智能体问答", "直接检索问答"], horizontal=True)
    library_name = st.radio("资料库", ["中国政策", "参考资料"], horizontal=True)
    library = "policy" if library_name == "中国政策" else "reference"
    selected = st.selectbox("检索范围", ["全部已收录政策", "仅国家", "某省级地区＋国家"])
    scope = {"全部已收录政策": "all", "仅国家": "national", "某省级地区＋国家": "province"}[
        selected
    ]
    province = st.selectbox("省级地区", PROVINCES) if scope == "province" else ""
    eligible = svc.ledger.eligible(library, scope, province)
    st.caption(
        f"当前范围：{len(eligible)} 份已核验且已入库资料。政策不联网更新，未收录不代表没有相关政策。"
    )
    if eligible:
        with st.expander(f"查看当前可用的 {len(eligible)} 份资料"):
            for row in eligible:
                st.write(f"• {row['metadata']['title']}（{row['page_count']} 页）")
        for row in eligible:
            if row["failed_pages"]:
                st.info(
                    f"《{row['metadata']['title']}》PDF 第 {'、'.join(map(str, row['failed_pages']))} 页为图片或解析异常页，已确认跳过，未参与问答。"
                )
    if mode == "智能体问答":
        st.caption(
            "可以打招呼，也可以接着上一问追问。涉及政策事实时，助手会重新查找原文并给出引用。"
        )
    if library == "reference":
        st.info("参考资料用于阅读研究，不能作为中国地方政策依据。")
    if scope == "province" and not any(r["metadata"]["province"] == province for r in eligible):
        st.info("当前未收录资料：所选地区暂无可检索的已核验文件。")
    if any("融合排放清单" in r["metadata"]["title"] for r in eligible):
        examples = [
            "融合排放清单是否适用于碳市场交易？",
            "国家应对气候变化标准体系包含哪三类一级标准子体系？",
            "两份文件对地区温室气体清单工作分别有哪些要求？",
        ]
        for column, example in zip(st.columns(3), examples, strict=True):
            if column.button(example, width="stretch"):
                st.session_state["policy_question"] = example
    with st.form("question"):
        question = st.text_area(
            "问题",
            placeholder="例如：融合排放清单能否用于碳市场交易？标准体系方案提出哪些建设目标？",
            key="policy_question",
        )
        submit = st.form_submit_button("检索并回答", type="primary")
    context = (mode, library, scope, province)
    history = st.session_state.setdefault("policy_history", [])
    if submit:
        previous = next(
            (
                item["question"]
                for item in reversed(history)
                if item["context"] == context and item["result"].status == "answered"
            ),
            "",
        )
        with st.spinner("正在思考并查找资料…"):
            engine = PolicyAgent(svc) if mode == "智能体问答" else svc
            if previous and is_followup(question):
                result = engine.ask(question, library, scope, province, previous_question=previous)
            else:
                result = engine.ask(question, library, scope, province)
        history.append({"context": context, "question": question, "result": result})
        st.session_state["policy_history"] = history[-12:]
    history = st.session_state["policy_history"]
    if any(item["context"] == context for item in history):
        if st.button("清空当前对话"):
            st.session_state["policy_history"] = [
                item for item in history if item["context"] != context
            ]
            st.rerun()
    stale = False
    for turn, item in enumerate(history):
        if item["context"] != context:
            continue
        result = item["result"]
        if any(
            not svc.ledger.get(c["doc_id"]) or not (svc.raw_dir / f"{c['doc_id']}.pdf").exists()
            for c in result.citations
        ):
            stale = True
            continue
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            if result.status in {"answered", "social"}:
                st.markdown(result.text)
            elif result.status == "error":
                st.error(result.text)
            else:
                st.warning(result.text)
            if result.status != "social":
                st.caption(f"本次耗时 {result.elapsed:.2f} 秒；政策引用仍需人工核查。")
            if result.steps:
                labels = {
                    "list_documents": "查看资料目录",
                    "search_documents": "检索原文",
                    "read_pdf_page": "读取原文页",
                }
                with st.expander("查看本次资料查找记录"):
                    for step in result.steps:
                        label = labels.get(step["tool"], "工具调用")
                        suffix = f"：{step['query']}" if step.get("query") else ""
                        if step.get("page"):
                            suffix += f"，PDF 第 {step['page']} 页"
                        st.write(f"{label}{suffix} → {step['result_count']} 条结果")
            for citation in result.citations:
                region = citation["province"] or citation["jurisdiction"]
                with st.expander(
                    f"[{citation['number']}] {citation['title']} · {region} · PDF 第 {citation['page']} 页"
                ):
                    st.write(
                        f"发布机构：{citation['issuer']} ｜ 发布日期：{citation['publish_date']}"
                    )
                    st.text(citation["excerpt"])
                    st.link_button("打开来源发布页", citation["source_url"])
                    st.download_button(
                        "下载入库原件",
                        (svc.raw_dir / f"{citation['doc_id']}.pdf").read_bytes(),
                        file_name=f"{citation['doc_id']}.pdf",
                        mime="application/pdf",
                        key=f"cite_{turn}_{citation['number']}",
                    )
    if stale:
        st.session_state["policy_history"] = [
            item
            for item in history
            if all(
                svc.ledger.get(c["doc_id"]) and (svc.raw_dir / f"{c['doc_id']}.pdf").exists()
                for c in item["result"].citations
            )
        ]
        st.info("部分引用文件已变更，对应历史回答已移除。")


def manage_page(svc, settings):
    st.subheader("上传与核验")
    st.caption(
        "切片会发送至云端嵌入接口；问题与召回片段会发送至聊天接口。仅上传允许这样处理的资料。"
    )
    uploads = st.file_uploader(
        "选择 PDF（支持多个，每份单独填写元数据）", type=["pdf"], accept_multiple_files=True
    )
    if uploads:
        selected = st.selectbox(
            "本次登记的文件", range(len(uploads)), format_func=lambda i: uploads[i].name
        )
        uploaded = uploads[selected]
        content = uploaded.getvalue()
        upload_key = hashlib.sha256(content).hexdigest()
        if st.button("本地解析预览"):
            try:
                parsed = parse_pdf(content, uploaded.name, settings.max_upload_mb)
                st.write(
                    {
                        "PDF 页数": parsed.page_count,
                        "可提取页数": len(parsed.pages),
                        "异常页": parsed.failed_pages,
                        "提取字数": sum(len(t) for _, t in parsed.pages),
                    }
                )
                for page, text in parsed.pages[:3]:
                    with st.expander(f"PDF 第 {page} 页（预览前 2000 字）"):
                        st.text(text[:2000])
            except ValueError as exc:
                st.error(str(exc))
        with st.form(f"upload_{upload_key}"):
            meta = metadata_form(f"upload_{upload_key}")
            partial = st.checkbox("已检查解析预览，确认跳过无法提取文字的页面")
            submit = st.form_submit_button("入库", type="primary")
        if submit:
            try:
                with st.spinner("解析、切片和向量入库…"):
                    result = svc.ingest(content, uploaded.name, meta, partial)
                st.success("文件已存在，未重复入库" if result["duplicate"] else "入库成功")
                st.write(
                    {
                        "页数": result["document"]["page_count"],
                        "切片数": result["document"]["chunk_count"],
                    }
                )
            except (ValueError, RuntimeError, PartialPagesError) as exc:
                st.error(str(exc))

    st.subheader("文件台账")
    rows = svc.ledger.list()
    if not rows:
        st.info("尚未入库。pdf 目录中的原始资料不会自动上传至云端。")
        return
    table = [
        {
            "ID": r["doc_id"],
            "标题": r["metadata"]["title"],
            "类型": r["metadata"]["doc_type"],
            "地区": r["metadata"]["province"] or r["metadata"]["jurisdiction"],
            "核验": r["metadata"]["verification_status"],
            "入库状态": r["status"],
            "页数": r["page_count"],
            "切片数": r["chunk_count"],
        }
        for r in rows
    ]
    st.dataframe(table, width="stretch", hide_index=True)
    export = [{**{k: v for k, v in r.items() if k != "metadata"}, **r["metadata"]} for r in rows]
    st.download_button(
        "导出台账 JSON",
        json.dumps(export, ensure_ascii=False, indent=2),
        file_name="policy-ledger.json",
        mime="application/json",
    )
    chosen = st.selectbox(
        "管理文件",
        [r["doc_id"] for r in rows],
        format_func=lambda doc_id: next(
            r["metadata"]["title"] for r in rows if r["doc_id"] == doc_id
        ),
    )
    row = svc.ledger.get(chosen)
    if row["error"]:
        st.warning(row["error"])
    st.caption(
        f"嵌入模型：{row['embedding_model']}；维度：{row['embedding_dim']}；异常页：{row['failed_pages']}"
    )
    with st.expander("修改元数据 / 重新入库"):
        with st.form(f"edit_{chosen}"):
            meta = metadata_form(f"edit_{chosen}", Metadata(**row["metadata"]))
            save = st.form_submit_button("保存元数据")
        if save:
            try:
                svc.update_metadata(chosen, meta)
                st.session_state.pop("policy_history", None)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        partial = st.checkbox("重新入库时允许跳过异常页", key=f"retry_partial_{chosen}")
        if st.button("重新生成索引" if row["status"] == "ready" else "从保存的原件重试入库"):
            path = svc.raw_dir / f"{chosen}.pdf"
            if path.exists():
                try:
                    svc.reindex(chosen, allow_partial=partial)
                    st.session_state.pop("policy_history", None)
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
            else:
                st.error("原件未完整保存，请删除失败记录后重新上传。")
    confirm = st.checkbox("确认删除该文件及其索引", key=f"delete_{chosen}")
    if st.button("删除文件", disabled=not confirm):
        try:
            svc.delete(chosen)
            st.session_state.pop("policy_history", None)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


def main():
    st.title("🌱 双碳政策资料问答")
    st.caption("单用户本地资料库 · 根据已核验原文回答 · 引用可回查")
    try:
        settings = Settings.from_env()
        svc = service(settings)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    with st.sidebar:
        page = st.radio("功能", ["政策问答", "资料管理", "收录覆盖与诊断"])
        available = svc.ledger.eligible()
        st.metric("可检索政策", len(available))
        st.caption(
            f"最近入库：{max((r['ingested_at'] or '' for r in svc.ledger.list()), default='尚无')}"
        )
        st.caption(f"聊天模型：{settings.chat_model or '未配置'}")
        st.caption(f"嵌入模型：{settings.embedding_model}")
        if not settings.api_key:
            st.info("请在本地 .env 填写 API Key，重启后生效。")
    if page == "政策问答":
        question_page(svc)
    elif page == "资料管理":
        manage_page(svc, settings)
    else:
        st.subheader("收录覆盖")
        st.caption("覆盖仅指台账中的资料数量。可检索数量还要求入库成功，不表示政策已收齐。")
        st.dataframe(svc.ledger.coverage(), width="stretch", hide_index=True)
        st.subheader("运行诊断")
        with svc.ledger.connection() as db:
            operations = [
                dict(r) for r in db.execute("SELECT * FROM operations ORDER BY rowid DESC LIMIT 50")
            ]
        st.dataframe(operations, width="stretch", hide_index=True)
        st.caption("诊断仅记录状态、模型、耗时和请求用量，不记录 API Key、问题或正文。")


if __name__ == "__main__":
    main()
