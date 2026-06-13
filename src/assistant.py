from src.loaders import load_all_documents
from src.graph import index_documents, build_graph

def run():
    print("=" * 50)
    print("   🧠 LLMonTheGo — Your Offline Assistant")
    print("=" * 50)

    # Load and index documents
    docs = load_all_documents()
    retriever = index_documents(docs)
    app = build_graph(retriever)

    print("💬 Ask anything (type 'quit' to exit)\n")

    while True:
        question = input("You: ")
        if question.lower() == "quit":
            print("\n👋 Goodbye! Travel safe.")
            break

        result = app.invoke({
            "question": question,
            "context": "",
            "answer": "",
            "found_in_docs": False
        })

        print(f"\n🤖 Assistant: {result['answer']}\n")
        print("-" * 50)
