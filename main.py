from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import google.generativeai as genai
import chromadb
import os

app = FastAPI()

# 1. SECURITY: The "Handshake" (CORS)
# This allows your HTML bubble to talk to this Python script
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. THE SECRET KEY: Getting the key from Render's private memory
# No hardcoding here! We use 'os.getenv'
gemini_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# 3. THE SMART WAREHOUSE: Setup ChromaDB (Vector DB)
client = chromadb.Client()
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
name="stellaria_docs",
embedding_function=default_ef
)

# 4. DATA LOADING: Putting your CSV into the Brain
# This runs once when the server starts
try:
 df = pd.read_csv("stellchatbotver1(in).csv").fillna("")
 documents = (df['question'] + " " + df['answer']).tolist()
 ids = [str(i) for i in range(len(documents))]
 collection.add(documents=documents, ids=ids)
 print("Successfully loaded CSV.")
except Exception as e:
 print(f"Error loading CSV: {e}")

# This defines the "shape" of the message coming from your HTML
class ChatRequest(BaseModel):
    message: str

# 5. THE ENDPOINT: The "Service Window"
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Search the CSV for the 2 most relevant rows
        results = collection.query(query_texts=[request.message], n_results=2)
        context = "\n".join(results["documents"][0])
        # You'll likely want to return something here
        return {"context": context}
    except Exception as e:
        # A try block MUST have an except or finally block
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
