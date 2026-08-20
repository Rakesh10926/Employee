from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss, pickle, numpy as np
from pathlib import Path

doc=Document("data/KT_Document.docx")
text="\n".join([p.text for p in doc.paragraphs if p.text.strip()])
splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=60)
chunks=splitter.split_text(text)
model=SentenceTransformer("all-MiniLM-L6-v2")
emb=model.encode(chunks)
index=faiss.IndexFlatL2(emb.shape[1])
index.add(np.array(emb))
Path("vectorstore").mkdir(exist_ok=True)
faiss.write_index(index,"vectorstore/index.faiss")
pickle.dump(chunks,open("vectorstore/metadata.pkl","wb"))
print("Indexed",len(chunks),"chunks")
