import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 256
    chunk_size: int = 700
    chunk_overlap: int = 80
    retriever_k: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", cls.max_new_tokens)),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            retriever_k=int(os.getenv("RETRIEVER_K", cls.retriever_k)),
        )
