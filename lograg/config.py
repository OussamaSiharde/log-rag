"""Runtime settings, read from environment variables with local-first defaults."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    dsn: str
    ollama_url: str
    embed_model: str
    chat_model: str
    embed_dim: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            dsn=os.environ.get("LOGRAG_DSN", "postgresql://lograg:lograg@localhost:5442/lograg"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            embed_model=os.environ.get("LOGRAG_EMBED_MODEL", "nomic-embed-text"),
            chat_model=os.environ.get("LOGRAG_CHAT_MODEL", "qwen2.5:7b"),
            embed_dim=int(os.environ.get("LOGRAG_EMBED_DIM", "768")),
        )
