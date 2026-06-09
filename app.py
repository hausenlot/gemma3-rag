import os, httpx, chromadb, re, json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any
import PyPDF2, io

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Startup ───────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Middleware with Tool Use")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer(EMBED_MODEL)
chroma = chromadb.PersistentClient(path="/data/chroma")
collection = chroma.get_or_create_collection("knowledge")

# ── Helpers (same as before) ─────────────────────────────────────────────────
def chunk_text(text: str) -> List[str]:
    """Improved chunking that respects paragraph boundaries."""
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        
        if para_words > CHUNK_SIZE:
            words = para.split()
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = " ".join(words[i:i + CHUNK_SIZE])
                if len(chunk.strip()) > 20:
                    chunks.append(chunk)
            continue
        
        if current_length + para_words > CHUNK_SIZE and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_paras = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
            current_chunk = overlap_paras.copy()
            current_length = sum(len(p.split()) for p in current_chunk)
        
        current_chunk.append(para)
        current_length += para_words
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return [c for c in chunks if len(c.strip()) > 50]

def extract_text(filename: str, data: bytes) -> str:
    """Improved PDF extraction that preserves structure."""
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            lines = page_text.split('\n')
            
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    cleaned_lines.append('')
                    continue
                
                if (line.isupper() and len(line) > 5) or line.endswith(':') or line.startswith('#'):
                    cleaned_lines.append(f"\n### {line}\n")
                elif line.startswith(('•', '-', '*', '○')):
                    cleaned_lines.append(f"  • {line[1:].strip()}")
                else:
                    cleaned_lines.append(line)
            
            text_parts.append('\n'.join(cleaned_lines))
        
        return '\n\n'.join(text_parts)
    
    text = data.decode("utf-8", errors="ignore")
    text = re.sub(r'^(#{1,3}\s)', r'\n\1\n', text, flags=re.MULTILINE)
    return text

def retrieve_context(query: str) -> List[str]:
    """Pure retrieval function - no prompt building."""
    if collection.count() == 0:
        return []
    
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb, 
        n_results=min(TOP_K, collection.count())
    )
    return results["documents"][0] if results["documents"] else []

# ── NEW: Tool/Function Calling System Prompt ─────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant for a File Server Control Panel dashboard.

You have access to a `search_documents` tool that can find information in uploaded documents (PDFs, text files).

TOOL USAGE:
- When a user asks a question that might be answered by documents they've uploaded, call `search_documents` with their question
- For general conversation, help requests, or questions about how YOU work, answer directly from your knowledge
- Example: "What's the storage limit?" → check documents first (call tool)
- Example: "Can you help me?" → answer directly (no tool needed)
- Example: "How do I upload a file?" → answer directly (you know this)

RESPONSE GUIDELINES:
- Be conversational and friendly
- If search_documents returns relevant info, use it to answer
- If search_documents returns nothing, use your training to help
- Never say "I cannot find this information" unless you've truly exhausted options
- For file server questions you know (uploading, API keys, navigation), just answer

Use the search_documents tool when you need specific document information.
"""

def build_tool_prompt(user_question: str, tool_results: Optional[List[str]] = None) -> str:
    """Build prompt that includes tool results if available."""
    if tool_results:
        context = "\n\n---\n\n".join(tool_results)
        return f"""{SYSTEM_PROMPT}

I searched the documents for: "{user_question}"

Here's what I found in the documents:
{context}

Now answer the user's question using this information. If the documents don't fully answer it, supplement with your general knowledge.

User: {user_question}

Assistant:"""
    else:
        # No tool used - just answer naturally
        return f"""{SYSTEM_PROMPT}

User: {user_question}

Assistant:"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "docs_indexed": collection.count()}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """Upload one or more documents into the vector store."""
    added = 0
    for f in files:
        data = await f.read()
        text = extract_text(f.filename, data)
        chunks = chunk_text(text)
        
        if not chunks:
            continue
            
        print(f"File: {f.filename}, extracted {len(chunks)} chunks")
        
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{f.filename}::chunk{i}" for i in range(len(chunks))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": f.filename, "chunk_id": i} for i, _ in enumerate(chunks)],
        )
        added += len(chunks)
    
    return {"indexed_chunks": added, "total_in_db": collection.count()}

@app.delete("/ingest")
def clear_knowledge():
    """Wipe the entire knowledge base."""
    chroma.delete_collection("knowledge")
    global collection
    collection = chroma.get_or_create_collection("knowledge")
    return {"status": "cleared"}

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[dict]
    max_tokens: int = 512
    stream: bool = False
    temperature: float = 0.3
    tools: Optional[List[dict]] = None  # For OpenAI compatibility

async def call_ollama(prompt: str, model: str, temperature: float, max_tokens: int, stream: bool):
    """Call Ollama with a prompt."""
    async with httpx.AsyncClient(timeout=120) as client:
        if stream:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            return StreamingResponse(resp.iter_bytes(), media_type="text/event-stream")
        else:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            return resp.json()

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    """OpenAI-compatible endpoint with tool/function calling pattern."""
    # Extract the latest user message
    user_msg = next(
        (m["content"] for m in reversed(req.messages) if m["role"] == "user"),
        None
    )
    if not user_msg:
        raise HTTPException(400, "No user message found")

    # Determine if this query needs document search
    # Simple heuristic: questions about document content vs general chat
    doc_keywords = ["document", "pdf", "uploaded", "manual", "guide", "policy", 
                    "according to", "what does it say", "find in", "storage limit",
                    "api key", "file size", "max upload"]
    
    needs_search = any(keyword in user_msg.lower() for keyword in doc_keywords)
    
    # Also search if there are documents and query isn't obviously general
    has_docs = collection.count() > 0
    general_phrases = ["help", "hi", "hello", "thanks", "how are you", "what can you do"]
    is_general = any(phrase in user_msg.lower() for phrase in general_phrases)
    
    tool_results = None
    if has_docs and (needs_search or not is_general):
        # Try to retrieve relevant content
        tool_results = retrieve_context(user_msg)
        print(f"Retrieved {len(tool_results) if tool_results else 0} chunks for: {user_msg}")
    
    # Build appropriate prompt
    prompt = build_tool_prompt(user_msg, tool_results if tool_results else None)
    
    # Call Ollama
    model = req.model or OLLAMA_MODEL
    
    if req.stream:
        return await call_ollama(prompt, model, req.temperature, req.max_tokens, True)
    else:
        result = await call_ollama(prompt, model, req.temperature, req.max_tokens, False)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result.get("response", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            }
        }

@app.get("/v1/models")
async def list_models():
    """List available models from Ollama."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{OLLAMA_URL}/api/tags")
        if resp.status_code == 200:
            models = resp.json()
            return {
                "object": "list",
                "data": [{"id": m["name"], "object": "model"} for m in models.get("models", [])]
            }
        return {"error": "Could not fetch models"}