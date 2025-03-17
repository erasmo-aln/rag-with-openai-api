import os
import dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Load API key
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="documents")

# Initialize OpenAI Embedding Model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

def answer_query(question, model="gpt-4o-mini", top_k=3, max_tokens=200, temperature=0.5):
    """Retrieves relevant chunks and generates a response using GPT chat completion API, including document sources."""

    # Generate embedding for query
    query_embedding = embedding_model.embed_query(question)

    # Retrieve top_k most relevant chunks and their metadata (filenames)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas"])

    retrieved_chunks = results["documents"][0] if results["documents"] else ["No relevant context found."]
    source_documents = results["metadatas"][0] if results["metadatas"] else [{}]

    # Format sources with document names
    unique_sources = list(set([source_documents[i].get('filename', 'Unknown document') for i in range(len(source_documents))]))
    sources = "\n".join([f"- {source}" for source in unique_sources])

    # Format retrieved context
    context = "\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": "You are an AI assistant answering questions based on the provided document context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    final_answer = response.choices[0].message.content.strip()

    return f"{final_answer}\n\n**Sources:**\n{sources}"

if __name__ == "__main__":
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = answer_query(query)
        print("\n", answer)
