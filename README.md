# PDF Question-Answering Chatbot

## Overview
This project is a Retrieval-Augmented Generation (RAG) system that enables users to ask questions about a collection of PDF documents and receive accurate, context-aware responses. The solution extracts text from PDFs, stores vector embeddings in a database, retrieves relevant content based on user queries, and generates answers using OpenAI's GPT models.

## Features
- Extracts text from PDFs and stores embeddings for retrieval.
- Uses OpenAI's `text-embedding-3-small` model to generate vector representations.
- Stores and retrieves embeddings using ChromaDB.
- Supports multiple GPT models (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`).
- Allows customization of retrieval parameters, including temperature, max tokens, and top-k retrieval.
- Provides a web-based interface using Streamlit.

## Technologies Used
### Backend
- **Python**: Core programming language.
- **OpenAI API**: GPT models for text generation.

### Vector Storage & Retrieval
- **ChromaDB**: Stores and retrieves document embeddings.
- **LangChain & LangChain-OpenAI**: Embedding and retrieval framework.
- **PyMuPDF (Fitz)**: Extracts text from PDFs.

### Frontend
- **Streamlit**: Provides an interactive web interface for querying and displaying results.

## Project Structure
```
rag-with-openai-api/
│── files/                     # Folder containing the PDFs
│── chroma_db/                 # Folder containing the embeddings
│── app.py                      # Streamlit web app
│── ingest.py                    # Extracts text from PDFs and stores embeddings in ChromaDB
│── query.py                     # Handles retrieval and GPT Model response generation
│── .env                          # Stores OpenAI API Key
│── requirements.txt              # Python dependencies
│── README.md                     # Documentation
```

## Installation
### Prerequisites
- Python Version: 3.9.13
- OpenAI API Key

### Requirements
```python
openai==1.6.1
langchain==0.1.9
chromadb==0.4.22
pymupdf==1.23.5
python-dotenv==1.0.1
streamlit==1.31.1
tiktoken==0.5.2
httpx==0.27
```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/erasmo-aln/rag-with-openai-api.git
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/scripts/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up the OpenAI API key:
   Create a `.env` file in the root directory and add:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage
### Step 1: Process PDFs and Store Embeddings
Run the ingestion script to process PDFs and store their embeddings:
```bash
python ingest.py
```

### Step 2: Start the Web Application
Run the Streamlit web application:
```bash
streamlit run app.py
```

### Step 3: Ask Questions
- Enter a question in the Streamlit interface.
- Customize model selection, temperature, max tokens, and retrieved chunks.
- Click "Get Answer" to receive a response.

## Configuration Options
The Streamlit app provides customization options:
- **GPT Model**: Choose from `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`.
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = high variability).
- **Max Tokens**: Defines response length.
- **Top-K Retrieval**: Determines how many document chunks are fetched.

## Contact
- **Linkedin**: [https://www.linkedin.com/in/erasmoneto/]
