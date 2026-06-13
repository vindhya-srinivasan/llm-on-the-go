from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from src.config import (
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS
)

# ─── State ────────────────────────────────────────────
class State(TypedDict):
    question: str
    context: str
    answer: str
    found_in_docs: bool

# ─── LLM & Embeddings ─────────────────────────────────
llm = ChatOllama(model=OLLAMA_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ─── Index Documents ──────────────────────────────────
def index_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Indexed {len(chunks)} chunks\n")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

# ─── Nodes ────────────────────────────────────────────
def retrieve(state: State, retriever) -> State:
    docs = retriever.invoke(state["question"])
    state["context"] = "\n\n".join([
        f"[{d.metadata['source']}]: {d.page_content}" for d in docs
    ])
    return state

def answer_from_docs(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent document assistant. Your job is to answer the user's question using the provided document context.

Follow this process:
1. Carefully analyze the user's question to understand what they are asking
2. Thoroughly search the context for any information relevant to the question
3. If relevant information exists — answer the question fully:
   - For summaries → synthesize and condense key points from the context
   - For specific facts → extract and quote directly from the context
   - For analysis → reason over the context to form a complete answer
   - For bullet points → structure your answer clearly from the context
4. If the context is partially relevant — use what is available and clearly state what is missing
5. Only respond with exactly: I_DONT_KNOW
   if after thorough analysis the context contains absolutely no information relevant to the question

Important rules:
- Always prioritize the document context over your own knowledge
- Never fabricate or assume information not present in the context
- Be concise but complete in your answers
- If multiple documents are referenced in context, synthesize across all of them

Context:
{context}"""),
        ("human", "{question}")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "question": state["question"],
        "context": state["context"]
    })

    if "I_DONT_KNOW" in result.content:
        print("→ Not found in docs, falling back to LLM...")
        state["found_in_docs"] = False
    else:
        state["answer"] = result.content
        state["found_in_docs"] = True

    return state

def answer_from_llm(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. The user's uploaded documents did not contain relevant information for this question.
Answer using your own knowledge clearly and accurately.
Always be transparent by starting your response with: [From LLM — not found in your documents]"""),
        ("human", "{question}")
    ])
    chain = prompt | llm
    result = chain.invoke({"question": state["question"]})
    state["answer"] = result.content
    return state

def route(state: State) -> str:
    return "end" if state["found_in_docs"] else "llm"

# ─── Build Graph ──────────────────────────────────────
def build_graph(retriever):
    graph = StateGraph(State)

    graph.add_node("retrieve", lambda s: retrieve(s, retriever))
    graph.add_node("answer_from_docs", answer_from_docs)
    graph.add_node("answer_from_llm", answer_from_llm)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer_from_docs")

    graph.add_conditional_edges(
        "answer_from_docs",
        route,
        {
            "end": END,
            "llm": "answer_from_llm"
        }
    )

    graph.add_edge("answer_from_llm", END)
    return graph.compile()
