import os
import dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="pdf_embeddings")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

def answer_query(question, model="gpt-4o-mini", top_k=3, max_tokens=200, temperature=0.5):
    """Retrieves relevant chunks and generates a response using GPT chat completion API."""

    query_embedding = embedding_model.embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_chunks = results["documents"][0] if results["documents"] else ["No relevant context found."]

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

    return response.choices[0].message.content.strip(), retrieved_chunks

if __name__ == "__main__":
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer, retrieved = answer_query(query)
        print("\nAnswer:\n", answer)
