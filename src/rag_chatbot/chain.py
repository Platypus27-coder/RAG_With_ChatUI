from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from .config import Settings
from .llm import build_llm


def build_qa_chain(vector_store, settings: Settings):
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retriever_k},
    )

    return ConversationalRetrievalChain.from_llm(
        llm=build_llm(settings),
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
