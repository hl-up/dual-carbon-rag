import hashlib
import io
from dataclasses import replace

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from reportlab.pdfgen import canvas

from carbon_rag.config import Settings
from carbon_rag.models import Metadata
from carbon_rag.service import RAGService
from carbon_rag.vector_index import VectorIndex


class LocalTestEmbeddings(Embeddings):
    """仅供隔离测试，不能替代真实模型评测。"""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [(v + 1) / 256 for v in digest[:8]]


class FakeChat:
    def __init__(self):
        self.result = '{"supported":true,"answer":"测试原文的内容。[1]"}'
        self.calls = []
        self.error = None

    def invoke(self, question, citations):
        self.calls.append((question, citations))
        if self.error:
            raise self.error
        return self.result, {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}


@pytest.fixture
def pdf_bytes():
    def make(*pages):
        stream = io.BytesIO()
        pdf = canvas.Canvas(stream, invariant=1)
        for text in pages:
            if text:
                pdf.drawString(50, 750, text)
            pdf.showPage()
        pdf.save()
        return stream.getvalue()

    return make


@pytest.fixture
def metadata():
    def make(**changes):
        return replace(
            Metadata(
                "合成测试文件（不是政策）",
                "合成测试机构",
                "2022-01-01",
                "https://example.gov.cn/synthetic",
                verification_status="已核验",
            ),
            **changes,
        )

    return make


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "isolated",
        api_key="test-placeholder-not-real",
        base_url="https://example.invalid/v1",
        chat_model="test-only",
        embedding_model="test-only-hash",
        embedding_dim=8,
    )


def local_index(settings):
    index = VectorIndex(settings)
    index._store = Chroma(
        collection_name="carbon_policy_v1",
        embedding_function=LocalTestEmbeddings(),
        persist_directory=str(settings.data_dir / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )
    return index


@pytest.fixture
def svc(settings):
    return RAGService(settings, index=local_index(settings), chat=FakeChat())
