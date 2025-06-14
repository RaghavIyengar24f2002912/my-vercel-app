import base64
import json
import os
import tempfile
import httpx
import requests
from pydantic import BaseModel
from fastapi import FastAPI
import numpy as np
from fastapi.middleware.cors import CORSMiddleware



# Keys
GEMINI_API_KEY = "AIzaSyDBh4pgfo95E8h2XKr82LqtQqd7TMq7hS0"
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDI5MTJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.d3EQS-25nWGXSTjwhpP7q3bEiint3sFZd6a5OJ4039c"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QuestionRequest(BaseModel):
    question: str
    image: str = None  # base64-encoded

def describe_base64_image(image_b64: str):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Describe this image in one sentence for educational documentation."},
                    {"inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64
                    }}
                ]
            }]
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"]
        else:
            return None
    except Exception as e:
        print(f"Image description failed: {e}")
        return None

def embed(text: str) -> list[float]:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://aipipe.org/openai/v1/embeddings",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "text-embedding-3-small",
                    "input": text
                }
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding failed: {e}")
        raise

def load_embeddings(path="combined_embeddings.json"):
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_chunks = []
        loaded_embeddings = []

        for key, value in data.items():
            # Basic field check
            if not isinstance(value, dict):
                print(f"⚠️ Skipping {key}: not a valid dictionary.")
                continue

            url = value.get("url")
            text = value.get("text")
            embedding = value.get("embedding")

            if not url or not text or not embedding:
                print(f"⚠️ Skipping {key}: missing url/text/embedding.")
                continue

            if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                print(f"⚠️ Skipping {key}: invalid embedding format.")
                continue

            if len(embedding) < 100:  # Optional: check vector length sanity
                print(f"⚠️ Skipping {key}: embedding too short ({len(embedding)} floats).")
                continue

            loaded_chunks.append({
                "source": key,
                "url": url,
                "text": text
            })
            loaded_embeddings.append(embedding)

        print(f"✅ Loaded {len(loaded_chunks)} valid chunks from {path}.")
        return loaded_chunks, loaded_embeddings

    except Exception as e:
        print(f"❌ Failed to load embeddings from {path}: {e}")
        return [], []


def answer(question: str, image_b64: str = None):
    if image_b64:
        image_description = describe_base64_image(image_b64)
        if image_description:
            question += f"\n\nImage context: {image_description}"

    loaded_chunks, loaded_embeddings = load_embeddings()
    if not loaded_chunks or not loaded_embeddings:
        return {
            "answer": "No knowledge base loaded. Please check if `combined_embeddings.json` exists and is valid.",
            "links": []
        }

    try:
        vector = embed(question)
    except Exception as e:
        return {
            "answer": f"Sorry, embedding failed: {e}",
            "links": []
        }

    if not vector or len(vector) == 0:
        return {
            "answer": "Sorry, something went wrong while embedding your question.",
            "links": []
        }

    similarities = []
    for i, emb in enumerate(loaded_embeddings):
        try:
            a = np.array(emb)
            b = np.array(vector)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            similarities.append((sim, i))
        except Exception as e:
            print(f"⚠️ Skipping index {i} due to similarity error: {e}")
            continue

    if not similarities:
        return {
            "answer": "No valid matches found in the knowledge base.",
            "links": []
        }

    top_3_indices = [i for _, i in sorted(similarities, reverse=True)[:3]]
    top_3_chunks = [loaded_chunks[i] for i in top_3_indices]

    context = "\n\n".join(chunk["text"] for chunk in top_3_chunks)[:3000]

    answer_text = generate_llm_response(question, context)

    return {
        "answer": answer_text.strip(),
        "links": [
            {"url": chunk["url"], "text": chunk["text"][:160].strip()}
            for chunk in top_3_chunks
        ]
    }


def generate_llm_response(question: str, context: str) -> str:
    prompt = f"""You are a helpful teaching assistant. Use the context below to answer the question briefly and clearly.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://aipipe.org/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a helpful teaching assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 300  # limit output to ~1-2 paragraphs
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return f"Error generating response: {e}"


@app.post("/api/")
async def api_answer(request: QuestionRequest):
    try:
        return answer(request.question, request.image)
    except Exception as e:
        return {"error": str(e)}

