from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
import json
import logging
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the actual graph system
from .graph.graph_builder import DynamicAgentGraph
from app.services.pdf_utils import PDFProcessor
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService
from app.services.file_utils import load_structured_file, answer_structured_query, toolbox_dispatch
from app.graph.nodes.answer_formatter import format_llm_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic Agent System API",
    description="FastAPI backend for LangGraph-based multi-agent system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the graph system
graph_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the graph system on startup"""
    global graph_system
    try:
        # Configuration for the graph system
        config = {
            "pdf_config": {
                "extract_images": True,
                "extract_tables": True
            },
            "pinecone_config": {
                "api_key": os.getenv("PINECONE_API_KEY", ""),
                "environment": os.getenv("PINECONE_ENVIRONMENT", ""),
                "index_name": os.getenv("PINECONE_INDEX_NAME", "")
            },
            "data_config": {
                "default_data_path": "./data/"
            }
        }
        
        graph_system = DynamicAgentGraph(config)
        logger.info("✅ Dynamic Agent Graph initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize graph system: {str(e)}")
        # Continue with mock responses if graph initialization fails
        graph_system = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    persona: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    formatted_response: Dict[str, Any]
    metadata: Dict[str, Any]
    status: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "graph_system_available": graph_system is not None,
        "system_mode": "actual_graph" if graph_system is not None else "mock_fallback",
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    import openai
    start_time = time.time()
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        # 1. Use LLM to classify the query
        classification_prompt = (
            "Classify this question as:\n"
            "- 'structured' if it is about tables, columns, numbers, or data analysis\n"
            "- 'math_financial' if it asks for financial calculations like moving averages, stock analysis, or time series calculations\n"
            "- 'unstructured' if it is about general document content\n"
            f"Question: {request.query}\n"
            "Answer with only 'structured', 'math_financial', or 'unstructured'."
        )
        classification_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a classifier that only answers with 'structured', 'math_financial', or 'unstructured'."},
                {"role": "user", "content": classification_prompt}
            ],
            max_tokens=20,
            temperature=0
        )
        classification_content = classification_response.choices[0].message.content
        if classification_content is not None:
            classification = classification_content.strip().lower()
        else:
            classification = "unstructured"
        logger.info(f"LLM classified query as: {classification}")

        # --- Structured Query Handling (CSV, numpy, column correction) ---
        if classification == "structured":
            # Use the new DbNode logic
            try:
                # Try to extract a function from the query (simple keyword match)
                function_keywords = ['mean', 'sum', 'median', 'std', 'min', 'max', 'count']
                function = None
                for kw in function_keywords:
                    if kw in request.query.lower():
                        function = kw
                        break
                if not function:
                    function = 'mean'  # Default to mean if not specified
                # Use the latest uploaded CSV (None means auto-detect latest)
                if graph_system is None or not hasattr(graph_system, 'db_node'):
                    return QueryResponse(
                        formatted_response={"error": "No graph system initialized. Please upload a CSV first."},
                        metadata={},
                        status="error"
                    )
                result = graph_system.db_node.process_query(request.query, filename="", function=function)
                if result.get("success"):
                    return QueryResponse(
                        formatted_response={
                            "result": result["result"],
                            "used_column": result["used_column"],
                            "used_file": result["used_file"],
                            "correction_info": result["correction_info"]
                        },
                        metadata={},
                        status="success"
                    )
                else:
                    return QueryResponse(
                        formatted_response={"error": result.get("error", "Unknown error")},
                        metadata={},
                        status="error"
                    )
            except Exception as e:
                logger.error(f"Error in structured CSV/numpy pipeline: {str(e)}")
                return QueryResponse(
                    formatted_response={"error": f"Structured CSV/numpy processing failed: {str(e)}"},
                    metadata={},
                    status="error"
                )
        # --- Unstructured Query Handling (PDF, text, etc.) ---
        elif classification == "unstructured":
            try:
                if graph_system is None:
                    return QueryResponse(
                        formatted_response={"error": "No graph system initialized. Please upload a document first."},
                        metadata={},
                        status="error"
                    )
                
                # 1. Find the latest uploaded document
                unstructured_dir = './data/uploads/unstructured/'
                if not os.path.exists(unstructured_dir):
                    return QueryResponse(
                        formatted_response={"error": "No unstructured documents directory found. Please upload a document first."},
                        metadata={},
                        status="error"
                    )
                
                files = [f for f in os.listdir(unstructured_dir) if f.endswith(('.pdf', '.txt'))]
                if not files:
                    return QueryResponse(
                        formatted_response={"error": "No documents found. Please upload a PDF or text document first."},
                        metadata={},
                        status="error"
                    )
                
                # Use the most recently uploaded document
                files.sort(key=lambda x: os.path.getmtime(os.path.join(unstructured_dir, x)), reverse=True)
                latest_file = files[0]
                logger.info(f"Using latest document: {latest_file}")
                
                # 2. Create embeddings for the user query
                from app.services.embedding_service import EmbeddingService
                embedding_service = EmbeddingService()
                query_embedding = embedding_service.embed_texts([request.query])[0]
                
                if not query_embedding:
                    return QueryResponse(
                        formatted_response={"error": "Failed to create query embedding."},
                        metadata={},
                        status="error"
                    )
                
                # 3. Search Pinecone for relevant chunks
                from app.services.pinecone_service import PineconeService
                pinecone_service = PineconeService()
                
                # Connect to Pinecone index
                connect_result = pinecone_service.connect_to_index()
                if not connect_result or connect_result.get("error"):
                    return QueryResponse(
                        formatted_response={"error": f"Failed to connect to Pinecone: {connect_result.get('error', 'Unknown error')}"},
                        metadata={},
                        status="error"
                    )
                
                # Search for relevant chunks  
                search_result = pinecone_service.semantic_search(
                    query_embedding=query_embedding,
                    top_k=5  # Get top 5 most relevant chunks
                    # document_id not specified = search all documents
                )
                
                if not search_result.get("success"):
                    return QueryResponse(
                        formatted_response={"error": f"Pinecone search failed: {search_result.get('error', 'Unknown error')}"},
                        metadata={},
                        status="error"
                    )
                
                relevant_chunks = search_result.get("results", [])
                if not relevant_chunks:
                    return QueryResponse(
                        formatted_response={"error": "No relevant content found in the document for your query."},
                        metadata={},
                        status="error"
                    )
                
                # 4. Use LLM to generate an answer based on retrieved chunks
                import openai
                
                # Prepare context from retrieved chunks
                context_text = ""
                references = []
                for i, chunk in enumerate(relevant_chunks):
                    chunk_text = chunk.get("text", "")
                    chunk_id = chunk.get("chunk_id", f"chunk_{i}")
                    relevance_score = chunk.get("relevance_score", 0.0)
                    
                    context_text += f"\n[Chunk {i+1}] (Relevance: {relevance_score:.3f})\n{chunk_text}\n"
                    references.append({
                        "chunk_id": chunk_id,
                        "relevance_score": relevance_score,
                        "text_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                    })
                
                # Generate answer using LLM
                rag_prompt = f"""
                Based on the following document excerpts, answer the user's question: "{request.query}"
                
                Document Context:
                {context_text}
                
                Instructions:
                - Provide a comprehensive answer based on the document content
                - Reference specific chunks when relevant (e.g., "According to Chunk 1...")
                - If the document doesn't contain enough information to answer fully, say so
                - Be specific and cite the relevant parts of the document
                """
                
                try:
                    rag_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on document content. Always cite your sources from the provided chunks."},
                            {"role": "user", "content": rag_prompt}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    )
                    
                    answer = rag_response.choices[0].message.content
                    
                    # 5. Format the response with references
                    formatted_result = {
                        "type": "document_analysis",
                        "answer": answer,
                        "query": request.query,
                        "document_used": latest_file,
                        "relevant_chunks_found": len(relevant_chunks),
                        "references": references,
                        "search_metadata": {
                            "total_chunks_searched": search_result.get("query_info", {}).get("top_k", 0),
                            "embedding_model": "text-embedding-ada-002"
                        }
                    }
                    
                    return QueryResponse(
                        formatted_response=formatted_result,
                        metadata={
                            "processing_type": "unstructured_rag",
                            "document_file": latest_file,
                            "chunks_retrieved": len(relevant_chunks)
                        },
                        status="success"
                    )
                    
                except Exception as llm_error:
                    logger.error(f"LLM generation failed: {str(llm_error)}")
                    return QueryResponse(
                        formatted_response={"error": f"Failed to generate answer: {str(llm_error)}"},
                        metadata={},
                        status="error"
                    )
                
            except Exception as e:
                logger.error(f"Error in unstructured document pipeline: {str(e)}")
                return QueryResponse(
                    formatted_response={"error": f"Unstructured document processing failed: {str(e)}"},
                    metadata={},
                    status="error"
                )
        # --- Fallback for other query types ---
        else:
            return QueryResponse(
                formatted_response={"error": "Only structured CSV/numpy and unstructured document queries are supported in this mode."},
                metadata={},
                status="error"
            )
    except Exception as e:
        logger.error(f"Error in /query endpoint: {str(e)}")
        return QueryResponse(
            formatted_response={"error": f"/query endpoint failed: {str(e)}"},
            metadata={},
            status="error"
        )

async def save_uploaded_file(file: UploadFile) -> tuple[str, bytes, str]:
    """Save uploaded file and return (safe_filename, file_bytes, file_path)."""
    upload_dir = "./data/uploads/"
    os.makedirs(upload_dir, exist_ok=True)
    safe_filename = file.filename if file.filename else f"uploaded_{int(time.time())}.dat"
    file_path = os.path.join(upload_dir, safe_filename)
    file_bytes = await file.read()
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    print(f"File saved to: {file_path}")
    return safe_filename, file_bytes, file_path

def detect_file_type(safe_filename: str, file_bytes: bytes, content_type: str) -> str:
    ext = os.path.splitext(safe_filename)[1].lower()
    # Robust structured file detection
    is_structured = (
        ext in [".csv", ".xls", ".xlsx"] or
        "csv" in content_type or
        (ext == "" and b"," in file_bytes[:1024] and b"\n" in file_bytes[:1024])
    )
    is_pdf_by_header = file_bytes[:4] == b'%PDF'
    is_pdf = (
        ext == ".pdf" or 
        content_type == "application/pdf" or 
        "pdf" in content_type or 
        is_pdf_by_header
    )
    is_text = (
        ext in [".txt", ".text"] or 
        content_type.startswith("text/") or
        "text" in content_type
    )
    print(f"File extension: '{ext}' (type: {type(ext)})")
    print(f"Content type: '{content_type}'")
    print(f"PDF header detected: {is_pdf_by_header}")
    print(f"Detected as PDF: {is_pdf}")
    print(f"Detected as Text: {is_text}")
    print(f"Detected as Structured: {is_structured}")
    if is_structured:
        return "structured"
    elif is_pdf:
        return "pdf"
    elif is_text:
        return "text"
    else:
        return "unsupported"

def save_structured_file(file_bytes: bytes, ext: str) -> str:
    structured_dir = './data/'
    os.makedirs(structured_dir, exist_ok=True)
    for prev_ext in ['.csv', '.xls', '.xlsx']:
        prev_path = os.path.join(structured_dir, f'structured{prev_ext}')
        if os.path.exists(prev_path):
            os.remove(prev_path)
    structured_path = os.path.join(structured_dir, f'structured{ext}')
    with open(structured_path, 'wb') as f:
        f.write(file_bytes)
    print(f"Structured file saved to: {structured_path}")
    return structured_path

def extract_text_from_file(file_path: str, file_type: str) -> str:
    if file_type == "structured":
        return load_structured_file(file_path).to_string(index=False)
    elif file_type == "pdf":
        pdf_processor = PDFProcessor()
        extraction_result = pdf_processor.extract_text_from_pdf(file_path)
        if not extraction_result.get("success"):
            raise Exception(extraction_result.get("error", "Text extraction failed"))
        return extraction_result["text"]
    elif file_type == "text":
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    else:
        raise Exception("Unsupported file type for text extraction")

def chunk_text(text: str) -> list[str]:
    pdf_processor = PDFProcessor()
    return [chunk["text"] for chunk in pdf_processor.chunk_text(text)]

def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    embedding_service = EmbeddingService()
    return embedding_service.embed_texts(chunks)

def upsert_chunks_to_pinecone(chunks, embeddings, safe_filename):
    pinecone_service = PineconeService()
    pinecone_connect_result = pinecone_service.connect_to_index()
    if pinecone_connect_result and pinecone_connect_result.get("error"):
        print(f"Pinecone connection error: {pinecone_connect_result['error']}")
        return
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{safe_filename}_chunk_{i}"
        metadata = {"text": chunk, "filename": safe_filename, "chunk_index": i}
        upsert_result = pinecone_service.upsert_vector(vector_id, embedding, metadata=metadata)
        if upsert_result.get("success"):
            print(f"✅ Upserted chunk {i+1}/{len(chunks)}: {vector_id}")
        else:
            print(f"❌ Failed to upsert chunk {i+1}: {upsert_result.get('error')}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"=== FILE UPLOAD DEBUG ===")
        print(f"Filename: {file.filename}")
        print(f"Content Type: {file.content_type}")
        print(f"File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
        safe_filename = file.filename if file.filename else f"uploaded_{int(time.time())}.dat"
        ext = os.path.splitext(safe_filename)[1].lower()
        content_type = (file.content_type or "").lower()
        file_bytes = await file.read()

        # Detect file type
        file_type = detect_file_type(safe_filename, file_bytes, content_type)
        if file_type == "unsupported":
            print("=== UNSUPPORTED FILE TYPE ===")
            return {
                "status": "error",
                "message": f"Unsupported file type. Extension: '{ext}', Content-Type: '{content_type}'. Only PDF, TXT, CSV, XLS, XLSX files are supported."
            }

        # Handle structured files (CSV/XLS/XLSX)
        if file_type == "structured":
            structured_dir = './data/uploads/structured/'
            os.makedirs(structured_dir, exist_ok=True)
            # Delete all previous files in structured_dir
            for f in os.listdir(structured_dir):
                try:
                    os.remove(os.path.join(structured_dir, f))
                except Exception as e:
                    print(f"Error deleting file {f}: {e}")
            # Save new file
            file_path = os.path.join(structured_dir, safe_filename)
            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            print(f"Structured file saved to: {file_path}")
            text = extract_text_from_file(file_path, file_type)
            print(f"Extracted text length: {len(text)}")
            print("Skipping embedding for structured file.")
            # --- Update CSV/column cache using DbNode ---
            try:
                if graph_system is not None:
                    graph_system.db_node.update_column_names_cache(safe_filename)
                    print("[UPLOAD] CSV columns cached successfully.")
                else:
                    from app.graph.nodes.db_node import DbNode
                    db_node = DbNode()
                    db_node.update_column_names_cache(safe_filename)
                    print("[UPLOAD] CSV columns cached successfully (local instance).")
            except Exception as meta_exc:
                print(f"[UPLOAD] Error updating CSV column cache: {meta_exc}")
            # --- End CSV/column cache update ---
            return {
                "status": "success",
                "filename": safe_filename,
                "file_type": file_type.upper(),
                "text_length": len(text),
                "message": "Structured file uploaded and columns cached successfully (no embeddings created)"
            }

        # Handle unstructured files (PDF)
        if file_type == "pdf":
            unstructured_dir = './data/uploads/unstructured/'
            os.makedirs(unstructured_dir, exist_ok=True)
            # Delete all previous files in unstructured_dir
            for f in os.listdir(unstructured_dir):
                try:
                    os.remove(os.path.join(unstructured_dir, f))
                except Exception as e:
                    print(f"Error deleting file {f}: {e}")
            # Save new file
            file_path = os.path.join(unstructured_dir, safe_filename)
            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            print(f"Unstructured file saved to: {file_path}")
            # Clear Pinecone index (delete all vectors)
            pinecone_service = PineconeService()
            pinecone_service.connect_to_index()
            try:
                pinecone_service.delete_all_vectors()
                print("Cleared Pinecone index.")
            except Exception as e:
                print(f"Error clearing Pinecone: {e}")
            # Optionally clear local structured data if needed (delete all files in ./data/structured/)
            structured_dir = './data/structured/'
            if os.path.exists(structured_dir):
                for f in os.listdir(structured_dir):
                    try:
                        os.remove(os.path.join(structured_dir, f))
                    except Exception as e:
                        print(f"Error deleting structured file {f}: {e}")
            # Continue with embedding, chunking, etc.
            text = extract_text_from_file(file_path, file_type)
            print(f"Extracted text length: {len(text)}")
            chunks = chunk_text(text)
            print(f"Number of chunks created: {len(chunks)}")
            embeddings = generate_embeddings(chunks)
            print(f"Generated {len(embeddings)} embeddings")
            print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
            upsert_chunks_to_pinecone(chunks, embeddings, safe_filename)
            print("=== SUCCESS ===")
            return {
                "status": "success",
                "filename": safe_filename,
                "file_type": file_type.upper(),
                "text_length": len(text),
                "num_chunks": len(chunks),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "message": "Unstructured file uploaded, Pinecone cleared, and embeddings created successfully"
            }

        # For other file types (e.g., text), keep previous logic or return error
        return {
            "status": "error",
            "message": f"File type '{file_type}' not supported for this operation."
        }
    except Exception as e:
        print(f"=== UPLOAD ERROR ===")
        print(f"Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Dynamic Agent System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "upload": "/upload"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
