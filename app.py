import streamlit as st
from query import answer_query


st.set_page_config(page_title="RAG-Powered PDF Chatbot", layout="wide")

st.title("PDF Q&A Chatbot")

st.markdown("""
This chatbot retrieves relevant content from your PDFs and provides accurate answers.
You can customize the settings below.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    # Model Selection
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    model = st.selectbox("Choose GPT Model", model_options)

    # Temperature slider
    temperature = st.slider("Temperature (randomness)", 0.0, 1.0, 0.5)

    # Max tokens input
    max_tokens = st.slider("Max Tokens (response length)", 50, 1000, 200)

    # Top-K retrieval
    top_k = st.slider("Number of relevant chunks to retrieve", 1, 5, 3)

# User input
question = st.text_input("Ask a question about the PDFs:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Fetching relevant information..."):
            answer = answer_query(question, model=model, temperature=temperature, max_tokens=max_tokens, top_k=top_k)

        # st.subheader("Retrieved Context:")
        # for i, chunk in enumerate(retrieved_chunks):
        #     st.markdown(f"**Source {i+1}:**")
        #     st.info(chunk)

        st.subheader("Answer:")
        st.write(answer)

if st.button("Clear"):
    st.experimental_rerun()
