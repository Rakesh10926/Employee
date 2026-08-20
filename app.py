import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Employee Management KT Chatbot")
st.title("📘 Employee Management KT Chatbot")

# Read KT document
doc = Document("data/KT_Document.docx")
text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60
)
chunks = splitter.split_text(text)

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Search function
def retrieve(query):
    q = model.encode([query])
    _, ids = index.search(np.array(q), 3)
    return [chunks[i] for i in ids[0]]

# UI
question = st.text_input("Ask a question about the KT document")

if question:
    results = retrieve(question)

    st.subheader("Answer")
    st.write(results[0])

    with st.expander("Retrieved KT Sections"):
        for i, r in enumerate(results, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(r)
