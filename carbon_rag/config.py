import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    chat_api_key: str = field(default="", repr=False)
    chat_base_url: str = ""
    chat_model: str = "deepseek-flash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    max_upload_mb: int = 50
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    timeout: float = 60
    max_retries: int = 1
    max_context_chars: int = 12000
    rerank_model: str = ""
    rerank_candidates: int = 15
    api_max_concurrent: int = 4
    api_queue_timeout: float = 10
    max_cosine_distance: float | None = None

    def __post_init__(self):
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE 且非负")
        if min(self.embedding_dim, self.max_upload_mb, self.top_k, self.timeout) <= 0:
            raise ValueError("维度、上传上限、Top-K 和超时必须大于零")
        if self.max_cosine_distance is not None and not 0 <= self.max_cosine_distance <= 2:
            raise ValueError("余弦距离上限应在 0 到 2 之间")
        if self.max_retries < 0 or self.max_context_chars < 0:
            raise ValueError("重试次数和上下文上限不能为负")
        if min(self.rerank_candidates, self.api_max_concurrent, self.api_queue_timeout) <= 0:
            raise ValueError("重排候选数、API 并发数和排队超时必须大于零")
        if self.rerank_candidates < self.top_k:
            raise ValueError("重排候选数不能小于 TOP_K")

    @classmethod
    def from_env(cls):
        load_dotenv()
        threshold = os.getenv("MAX_COSINE_DISTANCE", "").strip()
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            api_key=os.getenv("CLOSEAI_API_KEY", ""),
            base_url=os.getenv("CLOSEAI_BASE_URL", "").rstrip("/"),
            chat_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            chat_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            chat_model=os.getenv("CHAT_MODEL", "deepseek-flash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            top_k=int(os.getenv("TOP_K", "5")),
            timeout=float(os.getenv("API_TIMEOUT", "60")),
            max_retries=int(os.getenv("API_MAX_RETRIES", "1")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "12000")),
            rerank_model=os.getenv("RERANK_MODEL", "").strip(),
            rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "15")),
            api_max_concurrent=int(os.getenv("API_MAX_CONCURRENT", "4")),
            api_queue_timeout=float(os.getenv("API_QUEUE_TIMEOUT", "10")),
            max_cosine_distance=float(threshold) if threshold else None,
        )

    def require_api(self, chat=False):
        if not self.api_key or not self.base_url:
            raise ValueError("请在本地 .env 配置嵌入接口的 CLOSEAI_API_KEY 和 CLOSEAI_BASE_URL")
        if chat and (not self.chat_model or (self.chat_base_url and not self.chat_api_key)):
            raise ValueError("请在本地 .env 配置 CHAT_MODEL 和 DEEPSEEK_API_KEY")
        for url in (self.base_url, self.chat_base_url if chat else ""):
            if url and (urlparse(url).scheme != "https" or not urlparse(url).hostname):
                raise ValueError("模型 Base URL 必须为有效 HTTPS 接口地址")
