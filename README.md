# 🧠 LLMonTheGo

> Your offline personal document assistant — powered by local LLMs. No internet needed.

## What is this?

LLMonTheGo is a personal AI assistant that runs completely offline.
Drop in your documents (PDF, DOCX, TXT) and ask questions about them — 
perfect for travelling, commuting, or anywhere without internet.

If the answer isn't in your documents, it falls back to the local LLM's own knowledge.

## Features

- 📂 Supports PDF, DOCX, and TXT files (up to 5)
- 🔒 100% offline — powered by Ollama + Llama 3.2
- 🧠 Smart routing — answers from your docs first, LLM as fallback
- 🚀 Built with LangChain + LangGraph

## Tech Stack

- [LangChain](https://python.langchain.com) — document loading, chains
- [LangGraph](https://langchain-ai.github.io/langgraph) — stateful graph routing
- [Ollama](https://ollama.com) — local LLM runner
- [Llama 3.2](https://ollama.com/library/llama3.2) — local language model
- [ChromaDB](https://www.trychroma.com) — local vector database
- [HuggingFace Embeddings](https://huggingface.co) — local embeddings

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/LLMonTheGo.git
cd LLMonTheGo
```

### 2. Create virtual environment
```bash
python -m venv env
source env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama & pull Llama 3.2
```bash
brew install ollama
ollama pull llama3.2
```

### 5. Add your documents
```bash
mkdir docs
# Copy your PDF, DOCX, or TXT files into the docs/ folder
```

### 6. Run
```bash
python main.py
```

## Project Structure
