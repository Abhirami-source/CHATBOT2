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
model = genai.GenerativeModel("models/gemini-1.5-flash")

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
        # 1. Search the CSV
        results = collection.query(query_texts=[request.message], n_results=2)
        documents = results.get("documents", [[]])[0]
        context = "\n".join(documents) if documents else "No specific context from CSV."

        # 2. Improved "Helpful" Prompt
        prompt = (
            f"You are the Stellaria Assistant. Use the following context from our internal docs "
            f"to answer the question. If the information isn't in the context, use your general "
            f"knowledge to provide a helpful, space-themed answer.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {request.message}\n\n"
            f"ASSISTANT:"
        )

        # 3. Get response from Gemini
        response = model.generate_content(prompt)
        
        # Ensure we return a string, not 'undefined'
        bot_reply = response.text if response.text else "I'm having trouble focusing my sensors. Try again?"
        return {"reply": bot_reply}

    except Exception as e:
        print(f"Error: {e}")
        return {"reply": "Connection to the space station lost. Please try again later!"}
# 6. RENDER DEPLOYMENT GUARD
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
