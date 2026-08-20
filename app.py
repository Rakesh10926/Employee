import streamlit as st

st.set_page_config(page_title="EMS RAG Chatbot")

st.title("📘 Employee Management KT Chatbot")

question = st.text_input("Ask a question about the KT document")

if question:
    # Replace this with your RAG function later
    st.success(f"You asked: {question}")
    st.write("This is where the retrieved answer will appear.")
