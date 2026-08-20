import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os

st.set_page_config(page_title="Employee Management KT Chatbot")
st.title("📘 Employee Management KT Chatbot")

if not os.path.exists("data/KT_Document.docx"):
    st.error("KT_Document.docx not found inside data folder.")
    st.stop()

doc = Document("data/KT_Document.docx")
text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks = splitter.split_text(text)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

emb = model.encode(chunks)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(np.array(emb))

question = st.text_input("Ask a question")

if question:
    q = model.encode([question])
    _, ids = index.search(np.array(q), 3)

    st.subheader("Answer")
    st.write(chunks[ids[0][0]])

    with st.expander("Retrieved Context"):
        for i in ids[0]:
            st.write(chunks[i])
