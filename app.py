import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_XET_CACHE", str(Path(".cache/huggingface/xet").resolve()))

import chainlit as cl

from src.rag_chatbot.chain import build_qa_chain
from src.rag_chatbot.config import Settings
from src.rag_chatbot.vector_store import build_vector_store

WELCOME_MESSAGE = """Welcome to the PDF QA!

1. Upload a PDF or text file.
2. Ask a question about the uploaded file.
"""


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = Settings.from_env()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    vector_store = await cl.make_async(build_vector_store)(file, settings)
    chain = await cl.make_async(build_qa_chain)(vector_store, settings)

    cl.user_session.set("chain", chain)
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="Please upload a PDF or text file first.").send()
        return

    callback = cl.AsyncLangchainCallbackHandler()
    result = await chain.ainvoke(message.content, callbacks=[callback])

    answer = result.get("answer", "")
    source_documents = result.get("source_documents", [])
    text_elements = []

    for source_idx, source_doc in enumerate(source_documents):
        source_name = source_doc.metadata.get("source", f"source_{source_idx}")
        text_elements.append(cl.Text(content=source_doc.page_content, name=source_name))

    if text_elements:
        answer += f"\nSources: {', '.join(text_el.name for text_el in text_elements)}"
    else:
        answer += "\nNo sources found."

    await cl.Message(content=answer, elements=text_elements).send()
