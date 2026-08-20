import faiss,pickle,numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

model=SentenceTransformer("all-MiniLM-L6-v2")
index=faiss.read_index("vectorstore/index.faiss")
chunks=pickle.load(open("vectorstore/metadata.pkl","rb"))
client=OpenAI()

def retrieve(q,k=3):
    e=model.encode([q])
    _,ids=index.search(np.array(e),k)
    return [chunks[i] for i in ids[0]]

def ask(question):
    ctx="\n\n".join(retrieve(question))
    prompt=f"Answer only from the KT.\\n\\nContext:\\n{ctx}\\n\\nQuestion:{question}"
    r=client.chat.completions.create(model="gpt-4.1-mini",messages=[{"role":"user","content":prompt}])
    return r.choices[0].message.content
