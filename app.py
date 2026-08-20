from retriever import ask
print("=== EMS RAG Chatbot ===")
while True:
    q=input("You: ")
    if q.lower() in ["exit","quit"]:
        break
    print("Bot:",ask(q))
