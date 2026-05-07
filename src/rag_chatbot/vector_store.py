import chainlit as cl
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings
from .document_loader import load_and_split_file


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_kwargs={"device": "cpu"})


def build_vector_store(file, settings: Settings) -> Chroma:
    docs = load_and_split_file(file, settings)
    cl.user_session.set("docs", docs)

    return Chroma.from_documents(documents=docs, embedding=get_embeddings())
