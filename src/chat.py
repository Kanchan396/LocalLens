import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ---------- Config ----------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "locallens"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
TOP_K = 3
MAX_HISTORY = 6          # keep last 6 messages (3 exchanges)

# ---------- Load components ----------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to local vector database...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

print("LocalLens chat is ready. Type 'exit' to quit.\n")

# Conversation history
history = []

# ---------- Chat loop ----------
while True:
    question = input("You: ").strip()
    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break
    if not question:
        continue

    # 1. Embed the question
    query_embedding = embedder.encode([question]).tolist()

    # 2. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K
    )

    context_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n".join(context_chunks)

    # 3. Build conversation history text
    history_text = ""
    for msg in history[-MAX_HISTORY:]:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # 4. Build better prompt
    prompt = f"""You are a helpful assistant that answers questions using the provided context from the user's personal documents and the conversation history.
- Use the conversation history to understand follow-up questions.
- Prefer information from the context.
- If the answer is not in the context, say so clearly.

Conversation history:
{history_text}

Context from documents:
{context}

Current question: {question}

Answer:"""

    # 5. Ask the local LLM
    print("\nThinking...\n")
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]
    print("LocalLens:", answer)
    print(f"\n(Sources: {', '.join(set(sources))})\n")

    # 6. Save to history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})