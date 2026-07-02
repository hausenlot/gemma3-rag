import os, httpx, chromadb, re, json, hashlib, time, asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import PyPDF2, io

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")  # Updated model
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "256"))

# ── BM25 Index Helper ─────────────────────────────────────────────────────────
class BM25Searcher:
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.ids = []
        self.metadatas = []
        
    def rebuild(self):
        """Synchronously pull all documents from ChromaDB and rebuild the BM25 index."""
        if collection.count() == 0:
            self.bm25 = None
            self.documents = []
            self.ids = []
            self.metadatas = []
            return
            
        data = collection.get()
        self.documents = data.get("documents", []) or []
        self.ids = data.get("ids", []) or []
        self.metadatas = data.get("metadatas", []) or []
        
        if not self.documents:
            self.bm25 = None
            return
            
        tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.documents:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        scored_indices = []
        for idx, score in enumerate(scores):
            if score > 0:
                scored_indices.append((idx, score))
                
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_indices[:top_k]:
            results.append({
                "id": self.ids[idx],
                "document": self.documents[idx],
                "metadata": self.metadatas[idx] if self.metadatas else {},
                "score": round(score, 4)
            })
        return results

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
bm25_searcher = BM25Searcher()

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
    await loop.run_in_executor(None, bm25_searcher.rebuild)
    print(f"Embedder warmed up ✓")
    print(f"BM25 index built with {len(bm25_searcher.documents)} documents ✓")
    print(f"Using model: {OLLAMA_MODEL}")

# ── Document Processing Helpers ──────────────────────────────────────────────
def detect_heading(para: str) -> Optional[str]:
    """Detect if a paragraph is a heading or section title."""
    lines = [l.strip() for l in para.split('\n') if l.strip()]
    if not lines:
        return None
    
    first_line = lines[0]
    m = re.match(r'^#{1,6}\s+(.+)$', first_line)
    if m:
        return m.group(1).strip()
        
    for line in lines:
        if re.match(r'^={3,}|^[-#]{3,}', line):
            continue
        if re.match(r'^(?:SECTION|CHAPTER|PART)\s+\d+', line, re.IGNORECASE):
            return line.strip()
            
    if len(first_line) < 100 and first_line.isupper() and not first_line.endswith('.'):
        return first_line
        
    return None

def chunk_text(text: str, filename: str) -> List[Dict[str, Any]]:
    """Chunk text into semantic pieces, tracking position and heading metadata."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.py', '.js', '.json', '.csv', '.yaml', '.yml']:
        chunk_size = 300
        overlap = 50
    elif ext in ['.txt', '.md', '.pdf']:
        chunk_size = 200
        overlap = 40
    else:
        chunk_size = CHUNK_SIZE
        overlap = CHUNK_OVERLAP
        
    paragraphs = re.split(r'\n\s*\n', text)
    chunks_data = []
    current_chunk_paras = []
    current_length = 0
    current_heading = "General"
    
    for para in paragraphs:
        para_str = para.strip()
        if not para_str:
            continue
            
        heading = detect_heading(para_str)
        if heading:
            current_heading = heading
            
        para_words = len(para_str.split())
        
        if para_words > chunk_size:
            words = para_str.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_str = " ".join(words[i:i + chunk_size])
                if len(chunk_str.strip()) > 20:
                    chunks_data.append({
                        "text": chunk_str,
                        "heading": current_heading
                    })
            continue
            
        if current_length + para_words > chunk_size and current_chunk_paras:
            chunks_data.append({
                "text": "\n\n".join(current_chunk_paras),
                "heading": current_heading
            })
            overlap_paras = current_chunk_paras[-2:] if len(current_chunk_paras) > 2 else current_chunk_paras
            current_chunk_paras = overlap_paras.copy()
            current_length = sum(len(p.split()) for p in current_chunk_paras)
            
        current_chunk_paras.append(para_str)
        current_length += para_words
        
    if current_chunk_paras:
        chunks_data.append({
            "text": "\n\n".join(current_chunk_paras),
            "heading": current_heading
        })
        
    filtered_chunks = [c for c in chunks_data if len(c["text"].strip()) > 50]
    total_chunks = len(filtered_chunks)
    final_chunks = []
    
    for i, chunk in enumerate(filtered_chunks):
        if total_chunks == 1:
            position = "first"
        elif i == 0:
            position = "first"
        elif i == total_chunks - 1:
            position = "last"
        else:
            position = "middle"
            
        final_chunks.append({
            "text": chunk["text"],
            "metadata": {
                "source": filename,
                "chunk_id": i,
                "heading": chunk["heading"],
                "position": position,
                "total_chunks": total_chunks
            }
        })
        
    return final_chunks

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

def classify_query(query: str) -> Dict[str, Any]:
    """Classify the user query to adapt hybrid search weights dynamically."""
    tokens = re.findall(r'\b\w+\b', query.lower())
    num_tokens = len(tokens)
    
    has_error_code = bool(re.search(r'\berr(?:or)?\b|status\s*\d{3}|\b\d{3}\b|\b0x[0-9a-fA-F]+\b|\bERR_[A-Z0-9_]+\b|cve-\d{4}-\d{4,7}', query, re.IGNORECASE))
    has_version = bool(re.search(r'\bv?\d+\.\d+(?:\.\d+)*\b', query))
    
    question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'can', 'is', 'are', 'do', 'does', 'should', 'would', 'could', 'please'}
    is_question = query.strip().endswith('?') or (num_tokens > 0 and tokens[0] in question_words)
    
    is_short = num_tokens <= 3
    
    category = "default"
    vector_weight = 1.0
    bm25_weight = 1.0
    
    if has_error_code or has_version:
        category = "exact_identifier"
        vector_weight = 0.2
        bm25_weight = 1.0
    elif is_short:
        category = "short_keyword"
        vector_weight = 0.5
        bm25_weight = 1.0
    elif is_question or num_tokens > 6:
        category = "natural_language"
        vector_weight = 1.0
        bm25_weight = 0.2
        
    return {
        "category": category,
        "vector_weight": vector_weight,
        "bm25_weight": bm25_weight,
        "features": {
            "num_tokens": num_tokens,
            "has_error_code": has_error_code,
            "has_version": has_version,
            "is_question": is_question,
            "is_short": is_short
        }
    }

def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]], 
    bm25_results: List[Dict[str, Any]], 
    k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0
) -> List[Dict[str, Any]]:
    """Merge ranked results from multiple search methods using Reciprocal Rank Fusion."""
    rrf_scores = {}
    doc_map = {}
    
    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + vector_weight * (1.0 / (k + (rank + 1)))
        
    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + bm25_weight * (1.0 / (k + (rank + 1)))
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for rank, (doc_id, score) in enumerate(sorted_docs):
        doc_info = doc_map[doc_id].copy()
        doc_info["rrf_score"] = round(score, 6)
        doc_info["rrf_rank"] = rank + 1
        
        v_rank = next((i + 1 for i, d in enumerate(vector_results) if d["id"] == doc_id), None)
        b_rank = next((i + 1 for i, d in enumerate(bm25_results) if d["id"] == doc_id), None)
        doc_info["vector_rank"] = v_rank
        doc_info["bm25_rank"] = b_rank
        
        final_results.append(doc_info)
        
    return final_results

def compute_term_overlap(query: str, chunk_text: str) -> Dict[str, Any]:
    """Compute word overlap metrics between query and chunk text."""
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    chunk_words = set(re.findall(r'\b\w+\b', chunk_text.lower()))
    overlap_words = query_words.intersection(chunk_words)
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'he', 'she', 'him', 'her', 'his', 'hers', 'you', 'your', 'yours', 'we', 'us', 'our', 'ours'}
    meaningful_overlap = overlap_words - stop_words
    
    return {
        "overlap_count": len(overlap_words),
        "overlap_words": list(overlap_words),
        "meaningful_overlap_count": len(meaningful_overlap),
        "meaningful_overlap_words": list(meaningful_overlap)
    }

def retrieve_hybrid_context_details(query: str) -> Dict[str, Any]:
    """Perform dense vector + sparse BM25 search, apply RRF, and return comprehensive details."""
    classification = classify_query(query)
    v_weight = classification["vector_weight"]
    b_weight = classification["bm25_weight"]
    
    t_start = time.perf_counter()
    
    # 1. Vector Search
    t_vec = time.perf_counter()
    vector_docs = []
    vector_latency_ms = 0.0
    
    if collection.count() > 0:
        q_emb = get_query_embedding(query)
        # Retrieve candidate pool for RRF
        n_candidates = min(max(TOP_K * 2, 10), collection.count())
        results = collection.query(
            query_embeddings=q_emb,
            n_results=n_candidates
        )
        vector_latency_ms = (time.perf_counter() - t_vec) * 1000
        
        if results and results["documents"]:
            docs = results["documents"][0]
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            
            for idx, text in enumerate(docs):
                vector_docs.append({
                    "id": ids[idx],
                    "document": text,
                    "metadata": metadatas[idx] if metadatas[idx] is not None else {},
                    "score": round(1.0 - distances[idx], 4)
                })
                
    # 2. BM25 Search
    t_bm25 = time.perf_counter()
    bm25_docs = []
    bm25_latency_ms = 0.0
    
    if bm25_searcher.bm25:
        bm25_docs = bm25_searcher.search(query, top_k=max(TOP_K * 2, 10))
        bm25_latency_ms = (time.perf_counter() - t_bm25) * 1000
        
    # 3. RRF Fusion
    t_fusion = time.perf_counter()
    fused_results = reciprocal_rank_fusion(
        vector_docs, 
        bm25_docs, 
        k=60, 
        vector_weight=v_weight, 
        bm25_weight=b_weight
    )
    fusion_latency_ms = (time.perf_counter() - t_fusion) * 1000
    
    # 4. Limit to TOP_K and compute overlap
    final_results = fused_results[:TOP_K]
    for doc in final_results:
        doc["overlap"] = compute_term_overlap(query, doc["document"])
        
    total_latency_ms = (time.perf_counter() - t_start) * 1000
    
    return {
        "query": query,
        "classification": classification,
        "vector_latency_ms": round(vector_latency_ms, 2),
        "bm25_latency_ms": round(bm25_latency_ms, 2),
        "fusion_latency_ms": round(fusion_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "vector_results": vector_docs[:TOP_K],
        "bm25_results": bm25_docs[:TOP_K],
        "final_results": final_results
    }

def retrieve_context(query: str) -> List[str]:
    details = retrieve_hybrid_context_details(query)
    return [doc["document"] for doc in details["final_results"]]

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
        chunks = chunk_text(text, f.filename)
        
        if not chunks:
            continue
            
        print(f"File: {f.filename}, extracted {len(chunks)} chunks")
        
        loop = asyncio.get_event_loop()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = await loop.run_in_executor(None, lambda c=chunk_texts: embedder.encode(c).tolist())
        ids = [f"{f.filename}::chunk{i}" for i in range(len(chunks))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=[c["metadata"] for c in chunks],
        )
        added += len(chunks)
        
    # Rebuild BM25 searcher index after new uploads
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, bm25_searcher.rebuild)
    
    return {"indexed_chunks": added, "total_in_db": collection.count()}

@app.delete("/ingest")
def clear_knowledge():
    chroma.delete_collection("knowledge")
    global collection
    collection = chroma.get_or_create_collection("knowledge")
    bm25_searcher.rebuild()
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
    
    debug_mode = request.get("debug", True)
    
    loop = asyncio.get_event_loop()
    details = await loop.run_in_executor(None, retrieve_hybrid_context_details, query)
    
    cache_key = hashlib.sha256(query.lower().strip().encode()).hexdigest()
    was_cached = cache_key in _embed_cache
    
    response = {
        "query": query,
        "total_docs_in_db": collection.count(),
        "embedding_cached": was_cached,
    }
    
    if debug_mode:
        response.update({
            "classification": details["classification"],
            "latencies": {
                "vector_search_ms": details["vector_latency_ms"],
                "bm25_search_ms": details["bm25_latency_ms"],
                "rrf_fusion_ms": details["fusion_latency_ms"],
                "total_retrieval_ms": details["total_latency_ms"]
            },
            "standalone_vector_results": [
                {
                    "id": d["id"],
                    "document": d["document"],
                    "metadata": d["metadata"],
                    "score": d["score"]
                } for d in details["vector_results"]
            ],
            "standalone_bm25_results": [
                {
                    "id": d["id"],
                    "document": d["document"],
                    "metadata": d["metadata"],
                    "score": d["score"]
                } for d in details["bm25_results"]
            ],
            "merged_hybrid_results": [
                {
                    "id": d["id"],
                    "document": d["document"],
                    "metadata": d["metadata"],
                    "rrf_score": d["rrf_score"],
                    "rrf_rank": d["rrf_rank"],
                    "vector_rank": d["vector_rank"],
                    "bm25_rank": d["bm25_rank"],
                    "term_overlap": d["overlap"]
                } for d in details["final_results"]
            ]
        })
    else:
        response.update({
            "retrieval_time_ms": details["total_latency_ms"],
            "chunks_retrieved": len(details["final_results"]),
            "sample_chunks": [d["document"] for d in details["final_results"][:2]]
        })
        
    return response

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

# Add this new endpoint to your app.py (place it after your existing endpoints)

class GenerateRequest(BaseModel):
    """Flexible request model for stateless text generation"""
    prompt: Optional[str] = None  # Direct prompt (simplified mode)
    messages: Optional[List[dict]] = None  # Chat messages (advanced mode)
    system: Optional[str] = None  # System prompt (applies to messages mode)
    model: Optional[str] = None  # Override model
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    format: Optional[str] = None  # "json" for JSON mode
    tools: Optional[List[dict]] = None  # Optional function calling
    tool_choice: Optional[str] = None  # "auto", "none", or specific tool
    seed: Optional[int] = None  # For reproducibility
    raw: bool = False  # If True, return raw Ollama response

@app.post("/v1/generate")
async def generate_text(req: GenerateRequest):
    """
    Stateless generation endpoint - direct LLM API for Qwen2.5.
    
    Usage examples:
    
    1. Simple prompt:
        POST /v1/generate
        {"prompt": "Write a haiku about programming"}
    
    2. Chat format with system prompt:
        POST /v1/generate
        {
            "messages": [
                {"role": "user", "content": "Explain quantum computing"}
            ],
            "system": "You are a physics professor",
            "temperature": 0.3
        }
    
    3. JSON mode:
        POST /v1/generate
        {
            "prompt": "List 3 programming languages and their uses",
            "format": "json"
        }
    
    4. With function calling:
        POST /v1/generate
        {
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{"type": "function", "function": {...}}],
            "tool_choice": "auto"
        }
    
    5. Raw access to Ollama:
        POST /v1/generate
        {
            "messages": [...],
            "raw": true,
            "stream": false
        }
    """
    
    model = req.model or OLLAMA_MODEL
    start_time = time.perf_counter()
    
    # Convert simple prompt to messages format
    if req.prompt and not req.messages:
        messages = [{"role": "user", "content": req.prompt}]
    elif req.messages:
        messages = req.messages.copy()
        # Prepend system prompt if provided
        if req.system:
            messages.insert(0, {"role": "system", "content": req.system})
    else:
        raise HTTPException(400, "Either 'prompt' or 'messages' must be provided")
    
    # Prepare request body for Ollama
    ollama_request = {
        "model": model,
        "messages": messages,
        "stream": req.stream,
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
            "top_p": req.top_p,
            "frequency_penalty": req.frequency_penalty,
            "presence_penalty": req.presence_penalty,
        }
    }
    
    # Add optional parameters
    if req.stop:
        ollama_request["options"]["stop"] = req.stop
    if req.format == "json":
        ollama_request["format"] = "json"
    if req.seed is not None:
        ollama_request["options"]["seed"] = req.seed
    if req.tools:
        ollama_request["tools"] = req.tools
    if req.tool_choice:
        ollama_request["tool_choice"] = req.tool_choice
    
    # Handle streaming
    if req.stream:
        async def stream_generator():
            async with httpx.AsyncClient(timeout=120) as client:
                try:
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_URL}/api/chat",
                        json=ollama_request
                    ) as resp:
                        if resp.status_code != 200:
                            error_text = await resp.aread()
                            yield f"data: {json.dumps({'error': f'Ollama error: {error_text.decode()}'})}\n\n"
                            return
                        
                        async for line in resp.aiter_lines():
                            if line:
                                yield f"data: {line}\n\n"
                        yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    # Non-streaming request
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json=ollama_request
        )
        
        if response.status_code != 200:
            raise HTTPException(500, f"Ollama error: {response.text}")
        
        result = response.json()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Check for tool calls
        message = result.get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        # If raw mode, return complete Ollama response
        if req.raw:
            return {
                **result,
                "timing_ms": round(elapsed_ms, 2)
            }
        
        # Standard OpenAI-compatible response
        response_data = {
            "id": f"gen_{hashlib.md5(str(time.time()).encode()).hexdigest()[:10]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message.get("content", "")
                },
                "finish_reason": result.get("done_reason", "stop")
            }],
            "usage": result.get("usage", {}),
            "timing_ms": round(elapsed_ms, 2)
        }
        
        # Include tool calls if present
        if tool_calls:
            response_data["choices"][0]["message"]["tool_calls"] = tool_calls
        
        return response_data


@app.post("/v1/generate/batch")
async def generate_batch(requests: List[GenerateRequest]):
    """
    Batch generation - process multiple generation requests concurrently.
    
    Useful for generating multiple texts in parallel.
    
    Example:
        POST /v1/generate/batch
        [
            {"prompt": "Write about AI", "temperature": 0.8},
            {"prompt": "Write about cats", "temperature": 0.5},
            {"prompt": "Write about space", "temperature": 1.0}
        ]
    """
    
    async def process_one(req: GenerateRequest, idx: int):
        try:
            # Create a mini request without streaming
            req.stream = False
            result = await generate_text(req)
            if hasattr(result, "body"):
                # If it's a Response object, parse it
                return {"index": idx, "success": True, "result": result}
            return {"index": idx, "success": True, "result": result}
        except Exception as e:
            return {"index": idx, "success": False, "error": str(e)}
    
    # Process all requests concurrently
    tasks = [process_one(req, i) for i, req in enumerate(requests)]
    results = await asyncio.gather(*tasks)
    
    # Sort by index
    results.sort(key=lambda x: x["index"])
    
    return {
        "total": len(requests),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }


@app.get("/v1/generate/stream")
async def stream_generate_example():
    """
    OpenAPI documentation for streaming usage.
    This endpoint provides an example of how to use streaming.
    """
    return {
        "streaming_example": {
            "method": "POST",
            "endpoint": "/v1/generate",
            "body": {
                "prompt": "Write a story about a robot",
                "stream": True,
                "temperature": 0.8
            },
            "notes": "Streaming returns Server-Sent Events (SSE) format"
        }
    }