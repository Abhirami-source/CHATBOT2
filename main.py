import os
import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 1. SETUP & APP INITIALIZATION
app = FastAPI()

# SECURITY: Enable CORS so your frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CONFIGURATION
gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# 3. VECTOR DATABASE SETUP
client = chromadb.Client()
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="stellaria_docs",
    embedding_function=default_ef
)

# 4. DATA LOADING (Runs on Startup)
try:
    df = pd.read_csv("stellchatbotver1(in).csv").fillna("")
    documents = (df['question'] + " " + df['answer']).tolist()
    ids = [str(i) for i in range(len(documents))]
    collection.add(documents=documents, ids=ids)
    print("✅ Successfully loaded CSV and populated Vector DB.")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")

# DATA MODEL
class ChatRequest(BaseModel):
    message: str

# 5. THE CHAT ENDPOINT
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Search the CSV for the 2 most relevant rows
        results = collection.query(query_texts=[request.message], n_results=2)
        context = "\n".join(results["documents"][0])
        
        # Crafting the response using Gemini
        prompt = f"Context: {context}\n\nUser Question: {request.message}\n\nAnswer based on context:"
        response = model.generate_content(prompt)
        
        return {"reply": response.text, "context_used": context}
    except Exception as e:
        print(f"Error during chat: {e}")
        return {"error": str(e)}

# 6. RENDER DEPLOYMENT GUARD
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
