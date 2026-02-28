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
gemini_ef = chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=gemini_key,
  model_name="models/embedding-001"
)
collection = client.get_or_create_collection(name="stellaria_docs", embedding_function=gemini_ef)

# 4. DATA LOADING: Putting your CSV into the Brain
# This runs once when the server starts
df = pd.read_csv("stellchatbotver1(in).csv").fillna("")
documents = (df['question'] + " " + df['answer']).tolist()
ids = [str(i) for i in range(len(documents))]
collection.add(documents=documents, ids=ids)

# This defines the "shape" of the message coming from your HTML
class ChatRequest(BaseModel):
    message: str

# 5. THE ENDPOINT: The "Service Window"
@app.post("/chat")
async def chat(request: ChatRequest):
    # Search the CSV for the 2 most relevant rows
    results = collection.query(query_texts=[request.message], n_results=2)
    context = "\n".join(results["documents"][0])
    
    # Send the CSV info + User question to Gemini
    prompt = f"Context: {context}\n\nUser: {request.message}\nAnswer as Stellaria Club Assistant:"
    response = model.generate_content(prompt)
    
    # Send the answer back to your HTML bubble
    return {"reply": response.text}
