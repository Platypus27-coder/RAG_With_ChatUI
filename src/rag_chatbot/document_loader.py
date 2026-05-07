from chainlit.types import AskFileResponse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


def load_and_split_file(file: AskFileResponse, settings: Settings):
    if file.type == "text/plain":
        loader = TextLoader(file.path, encoding="utf-8")
    elif file.type == "application/pdf":
        loader = PyPDFLoader(file.path)
    else:
        raise ValueError(f"Unsupported file type: {file.type}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)

    for index, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{index}"

    return docs
