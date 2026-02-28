from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import google.generativeai as genai
import chromadb
import os

app = FastAPI()

# VERY IMPORTANT: This allows your HTML file to talk to this Python code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Setup Gemini
genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-1.5-flash")

# 2. Setup ChromaDB (The Vector DB)
client = chromadb.Client()
gemini_ef = chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key="YOUR_API_KEY_HERE",
    model_name="models/text-embedding-004"
)
collection = client.get_or_create_collection(name="stellaria_docs", embedding_function=gemini_ef)

# 3. Load your CSV data into the DB
df = pd.read_csv("stellchatbotver1(in).csv").fillna("")
documents = (df['question'] + " " + df['answer']).tolist()
ids = [str(i) for i in range(len(documents))]
collection.add(documents=documents, ids=ids)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    # Search the CSV for the best match
    results = collection.query(query_texts=[request.message], n_results=2)
    context = "\n".join(results["documents"][0])
    
    # Ask Gemini to answer using that context
    prompt = f"Context: {context}\n\nUser: {request.message}\nAnswer as Stellaria Club Assistant:"
    response = model.generate_content(prompt)
    
    return {"reply": response.text}
