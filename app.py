import os, httpx, chromadb, re, json, hashlib, time, asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any, Tuple
import PyPDF2, io

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")  # Updated model
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "256"))

# ── Startup ───────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Middleware with Function Calling")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer(EMBED_MODEL)
chroma = chromadb.PersistentClient(path="/data/chroma")
collection = chroma.get_or_create_collection("knowledge")

# ── Embedding Cache ───────────────────────────────────────────────────────────
_embed_cache: Dict[str, list] = {}

def get_query_embedding(query: str) -> list:
    """Encode a query, returning cached result if available."""
    key = hashlib.sha256(query.lower().strip().encode()).hexdigest()
    if key not in _embed_cache:
        if len(_embed_cache) >= EMBED_CACHE_SIZE:
            _embed_cache.pop(next(iter(_embed_cache)))
        _embed_cache[key] = embedder.encode([query]).tolist()
    return _embed_cache[key]

@app.on_event("startup")
async def warmup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: embedder.encode(["warmup"]))
    print(f"Embedder warmed up ✓")
    print(f"Using model: {OLLAMA_MODEL}")

# ── Document Processing Helpers ──────────────────────────────────────────────
def chunk_text(text: str) -> List[str]:
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
    if collection.count() == 0:
        return []
    
    q_emb = get_query_embedding(query)
    results = collection.query(
        query_embeddings=q_emb, 
        n_results=min(TOP_K, collection.count())
    )
    return results["documents"][0] if results["documents"] else []

async def retrieve_context_async(query: str) -> List[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, retrieve_context, query)

# ── Function Calling Setup ────────────────────────────────────────────────────
# Define the tool that the model can call
SEARCH_DOCUMENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search through uploaded documents (PDFs, text files) to find relevant information. Use this when the user asks about file server operations, upload limits, storage policies, API keys, or any documented content.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information in the documents"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

# System prompt for function calling mode
SYSTEM_PROMPT = """You are a helpful assistant for a File Server Control Panel dashboard.

You have access to a function called `search_documents` that can find information in uploaded documents.

IMPORTANT RULES:
1. If the user asks about information that might be in their documents (file operations, upload limits, storage, API keys, how-to guides, policies), you MUST call the `search_documents` function.
2. For general conversation (greetings, help, what you can do, basic file server knowledge like "how to upload" which you already know), answer directly without calling the function.
3. When you receive search results, incorporate them naturally into your response.
4. If search results don't answer the question, use your general knowledge.

Examples:
- User: "What's the storage limit?" → Call search_documents
- User: "How do I upload a file?" → Answer directly (you know this)
- User: "Hello" → Answer directly
- User: "What does the manual say about API keys?" → Call search_documents

Be conversational and helpful."""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_indexed": collection.count(),
        "embed_cache_entries": len(_embed_cache),
        "model": OLLAMA_MODEL,
    }

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    added = 0
    for f in files:
        data = await f.read()
        text = extract_text(f.filename, data)
        chunks = chunk_text(text)
        
        if not chunks:
            continue
            
        print(f"File: {f.filename}, extracted {len(chunks)} chunks")
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, lambda c=chunks: embedder.encode(c).tolist())
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
    tools: Optional[List[dict]] = None

async def call_ollama_with_tools(
    messages: List[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    tools: List[dict]
):
    """Call Ollama with function calling support."""
    
    # Prepare the request body for Ollama
    request_body = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "tools": tools  # Ollama supports tools in the API
    }
    
    if stream:
        async def stream_generator():
            client = httpx.AsyncClient(timeout=120)
            try:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",  # Use /api/chat for messages format
                    json=request_body
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=request_body
            )
            return resp.json()

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    """OpenAI-compatible endpoint with proper function calling and timing breakdown."""
    
    # Start total timer
    total_start = time.perf_counter()
    timing = {}
    
    # Prepare messages with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + req.messages
    
    model = req.model or OLLAMA_MODEL
    tools = req.tools or [SEARCH_DOCUMENTS_TOOL]
    
    # First call: Let the model decide if it needs to search
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "num_predict": req.max_tokens
                },
                "tools": tools
            }
        )
        
        timing["llm_first_call_ms"] = (time.perf_counter() - t0) * 1000
        
        if response.status_code != 200:
            raise HTTPException(500, f"Ollama error: {response.text}")
        
        result = response.json()
        
        # Check if the model wants to call a tool
        message = result.get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        if tool_calls:
            print(f"Tool call detected: {tool_calls}")
            
            tool_results = []
            retrieval_time_ms = 0
            
            for tool_call in tool_calls:
                if tool_call.get("function", {}).get("name") == "search_documents":
                    # Parse the query
                    function_args = tool_call.get("function", {}).get("arguments", {})
                    if isinstance(function_args, str):
                        function_args = json.loads(function_args)
                    
                    query = function_args.get("query", "")
                    
                    # Perform the search with timing
                    t_retrieval = time.perf_counter()
                    search_results = await retrieve_context_async(query)
                    retrieval_time_ms = (time.perf_counter() - t_retrieval) * 1000
                    
                    print(f"Searched for '{query}': {len(search_results)} chunks in {retrieval_time_ms:.2f}ms")
                    
                    # Check cache status
                    cache_key = hashlib.sha256(query.lower().strip().encode()).hexdigest()
                    was_cached = cache_key in _embed_cache
                    
                    timing["retrieval_ms"] = retrieval_time_ms
                    timing["embedding_cached"] = was_cached
                    timing["chunks_retrieved"] = len(search_results)
                    
                    # Format results as a tool response
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", "search_documents"),
                        "name": "search_documents",
                        "content": json.dumps({
                            "query": query,
                            "results": search_results,
                            "count": len(search_results)
                        })
                    })
            
            # Append tool results to conversation and make second call
            messages.append(message)  # Add the assistant's tool call message
            messages.extend(tool_results)  # Add tool responses
            
            # Second call: Get final answer with search results
            t_llm = time.perf_counter()
            async with httpx.AsyncClient(timeout=120) as client2:
                final_response = await client2.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": req.stream,
                        "options": {
                            "temperature": req.temperature,
                            "num_predict": req.max_tokens
                        }
                    }
                )
                
                timing["llm_second_call_ms"] = (time.perf_counter() - t_llm) * 1000
                timing["llm_total_ms"] = timing.get("llm_first_call_ms", 0) + timing.get("llm_second_call_ms", 0)
                
                if req.stream:
                    # Handle streaming response
                    async def stream_final():
                        client_stream = httpx.AsyncClient(timeout=120)
                        try:
                            async with client_stream.stream(
                                "POST",
                                f"{OLLAMA_URL}/api/chat",
                                json={
                                    "model": model,
                                    "messages": messages,
                                    "stream": True,
                                    "options": {
                                        "temperature": req.temperature,
                                        "num_predict": req.max_tokens
                                    }
                                }
                            ) as resp:
                                async for chunk in resp.aiter_bytes():
                                    # Parse chunk to inject timing info (complex for streaming)
                                    yield chunk
                        finally:
                            await client_stream.aclose()
                    
                    return StreamingResponse(stream_final(), media_type="text/event-stream")
                else:
                    final_result = final_response.json()
                    final_message = final_result.get("message", {})
                    
                    timing["total_ms"] = (time.perf_counter() - total_start) * 1000
                    
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": final_message.get("content", "")
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": final_result.get("usage", {}),
                        "timing_breakdown": timing  # <-- HERE'S YOUR TIMING DATA
                    }
        else:
            # No tool call, just return the response
            timing["llm_total_ms"] = timing.get("llm_first_call_ms", 0)
            timing["no_tool_called"] = True
            timing["total_ms"] = (time.perf_counter() - total_start) * 1000
            
            if req.stream:
                # Handle streaming
                async def stream_direct():
                    client = httpx.AsyncClient(timeout=120)
                    try:
                        async with client.stream(
                            "POST",
                            f"{OLLAMA_URL}/api/chat",
                            json={
                                "model": model,
                                "messages": messages,
                                "stream": True,
                                "options": {
                                    "temperature": req.temperature,
                                    "num_predict": req.max_tokens
                                }
                            }
                        ) as resp:
                            async for chunk in resp.aiter_bytes():
                                yield chunk
                    finally:
                        await client.aclose()
                
                return StreamingResponse(stream_direct(), media_type="text/event-stream")
            else:
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": message.get("content", "")
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": result.get("usage", {}),
                    "timing_breakdown": timing  # <-- HERE'S YOUR TIMING DATA
                }

@app.post("/test/retrieval")
async def test_retrieval_only(request: dict):
    query = request.get("query")
    if not query:
        raise HTTPException(400, "Missing 'query' field")
    
    t0 = time.monotonic()
    tool_results = await retrieve_context_async(query)
    elapsed = time.monotonic() - t0
    
    cache_key = hashlib.sha256(query.lower().strip().encode()).hexdigest()
    was_cached = cache_key in _embed_cache
    
    return {
        "query": query,
        "retrieval_time_ms": round(elapsed * 1000, 2),
        "chunks_retrieved": len(tool_results) if tool_results else 0,
        "embedding_cached": was_cached,
        "total_docs_in_db": collection.count(),
        "sample_chunks": tool_results[:2] if tool_results else [],
    }

@app.get("/v1/models")
async def list_models():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{OLLAMA_URL}/api/tags")
        if resp.status_code == 200:
            models = resp.json()
            return {
                "object": "list",
                "data": [{"id": m["name"], "object": "model"} for m in models.get("models", [])]
            }
        return {"error": "Could not fetch models"}