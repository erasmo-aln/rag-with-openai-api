import os
import fitz
import chromadb
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Load API key
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")


def extract_text_from_pdfs(directory):
    """Extracts text from all PDFs in a folder."""
    extracted_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            doc = fitz.open(filepath)
            text = "\n".join(page.get_text("text") for page in doc)
            extracted_data.append((filename, text))
    return extracted_data


def chunk_and_store_text():
    """Splits text, embeds, and stores in ChromaDB."""
    pdf_texts = extract_text_from_pdfs("files")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for doc_name, text in pdf_texts:
        chunks = text_splitter.split_text(text)
        embeddings = embedding_model.embed_documents(chunks)

        for i, emb in enumerate(embeddings):
            collection.add(
                ids=[f"{doc_name}_{i}"],
                documents=[chunks[i]],
                embeddings=[emb]
            )

    print("Embedding storage complete.")


if __name__ == "__main__":
    chunk_and_store_text()
