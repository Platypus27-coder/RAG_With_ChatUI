#  RAG Chatbot with UI (LangChain + Vicuna)

##  Giới thiệu
Một chatbot sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)** cho phép:
- Chat với tài liệu (PDF)
- Trả lời dựa trên dữ liệu riêng


---

## Công nghệ sử dụng
- LLM: Vicuna / HuggingFace Transformers  
- Framework: LangChain  
- Vector Database: ChromaDB  
- UI: Chainlit  
- Embedding: HuggingFace Embeddings  

---

##  Cài đặt

### 1. Clone repo
```bash
git clone https://github.com/Platypus27-coder/rag-chatbot.git
cd rag-chatbot
```
### 2. Thư viện 
```bash
pip install -r requirements.txt 
```
### 3. Chạy app
```bash
chainlit run app.py
```
