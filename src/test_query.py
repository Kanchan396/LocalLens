from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("locallens")

query = "What is LocalLens?"
query_embedding = embedder.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=1
)

print("\n=== Query Result ===")
print("Question:", query)
print("\nMatching text:")
print(results["documents"][0][0])
print("\nSource:", results["metadatas"][0][0]["source"])