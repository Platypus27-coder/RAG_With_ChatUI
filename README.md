# RAG Chatbot with Chainlit

A local Retrieval-Augmented Generation (RAG) chatbot for asking questions over uploaded PDF or text files. The app uses Chainlit for the chat UI, LangChain for the RAG pipeline, ChromaDB for vector search, HuggingFace embeddings, and a quantized HuggingFace chat model for generation.

## Features

- Upload PDF or plain text files from the chat UI.
- Split documents into chunks and index them in ChromaDB.
- Ask follow-up questions with conversation memory.
- Return answers with source snippets.
- Run a local HuggingFace model with 4-bit quantization.

## Tech Stack

- UI: Chainlit
- RAG framework: LangChain
- Vector database: ChromaDB
- Loader: PyPDF / LangChain document loaders
- Embeddings: HuggingFace Embeddings
- Default LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Quantization: bitsandbytes 4-bit NF4

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- .env.example
|-- chainlit.md
|-- notebooks/
|   `-- solution_rag_with_chatui.ipynb
`-- src/
    `-- rag_chatbot/
        |-- chain.py
        |-- config.py
        |-- document_loader.py
        |-- llm.py
        `-- vector_store.py
```

## Setup

Create and activate a Conda environment:

```powershell
conda create -n chatbot python=3.11.7 -y
conda activate chatbot
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The included requirements use PyTorch CUDA 12.1 because this is a broadly compatible target for the pinned bitsandbytes version on Windows. If your platform needs a different PyTorch build, install the matching PyTorch package from the official PyTorch index and adjust `requirements.txt`.

Verify CUDA support:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -m bitsandbytes
```

Copy the example environment file if you want to override defaults:

```powershell
copy .env.example .env
```

## Configuration

Default values are defined in `.env.example`:

```env
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
MAX_NEW_TOKENS=256
CHUNK_SIZE=700
CHUNK_OVERLAP=80
RETRIEVER_K=3
HF_HOME=.cache/huggingface
```

For lower-memory GPUs, reduce `MAX_NEW_TOKENS` or use a smaller chat model. For larger GPUs, you can increase token limits or switch to a stronger model.

## Run

Start the app:

```powershell
conda activate chatbot
$env:DEBUG="false"
chainlit run app.py --headless --host 127.0.0.1 --port 8000
```

Open the app at:

```text
http://127.0.0.1:8000
```

The first run may take time because HuggingFace model and embedding files are downloaded into the local cache.

## Notes

- The app caches the LLM inside the running process to avoid reloading the model for every upload.
- Embeddings are configured to run on CPU to leave more GPU memory for generation.
- If you repeatedly upload large files and run out of GPU memory, restart the Chainlit server and consider lowering `MAX_NEW_TOKENS`.
- Runtime artifacts such as `.cache/`, `.chainlit/`, `.files/`, logs, and Python caches are ignored by Git.
