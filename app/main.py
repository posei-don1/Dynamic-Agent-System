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
from app.graph.services.pdf_utils import PDFProcessor
from app.graph.services.embedding_service import EmbeddingService
from app.graph.services.pinecone_service import PineconeService
from app.graph.services.file_utils import load_structured_file, answer_structured_query, toolbox_dispatch
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
    start_time = time.time()
    try:
        logger.info(f"Processing query through LangGraph: {request.query[:50]}...")
        
        # Check if graph system is available
        if graph_system is None:
            return QueryResponse(
                formatted_response={"error": "LangGraph system not initialized. Please check your API keys and restart the server."},
                metadata={"system_mode": "error", "error": "Graph system unavailable"},
                status="error"
            )
        
        # Prepare context for the graph
        context = request.context or {}
        
        # Add persona information if provided
        if request.persona:
            context["requested_persona"] = request.persona
        
        # Process query through LangGraph system
        print(f"Processing query: {request.query}")
        result = await graph_system.process_query(request.query, context)
        
        if result.get("success"):
            # Extract the formatted response from the graph result
            formatted_response = result.get("response", {})
            metadata = result.get("metadata", {})
            
            # Add processing metadata
            metadata.update({
                "processing_time": time.time() - start_time,
                "system_mode": "langgraph",
                "graph_flow": "persona_selector → router → processor → suggestion_generator → answer_formatter"
            })
            
            # Create the final response
            final_response = QueryResponse(
                formatted_response=formatted_response,
                metadata=metadata,
                status="success"
            )
            
            # Print the response before sending to frontend
            print("\n" + "="*80)
            print("🚀 BACKEND RESPONSE BEING SENT TO FRONTEND")
            print("="*80)
            print(f"📝 Query: {request.query}")
            print(f"👤 Persona: {request.persona}")
            print(f"⏱️  Processing Time: {metadata['processing_time']:.3f}s")
            print(f"🎯 Status: {final_response.status}")
            print(f"🔄 Route Selected: {metadata.get('route_selected', 'Unknown')}")
            print(f"📊 Route Confidence: {metadata.get('route_confidence', 0):.2f}")
            print(f"🤖 Persona Selected: {metadata.get('persona_selected', 'Unknown')}")
            print(f"📄 Response Type: {formatted_response.get('type', 'Unknown')}")
            
            # Print sections if available
            if 'sections' in formatted_response:
                print(f"📋 Sections Found: {len(formatted_response['sections'])}")
                for i, section in enumerate(formatted_response['sections']):
                    print(f"   {i+1}. {section.get('title', 'No title')} ({section.get('type', 'No type')})")
            
            # Print suggestions if available
            suggestions_section = next((s for s in formatted_response.get('sections', []) if s.get('type') == 'suggestions'), None)
            if suggestions_section and 'suggestions' in suggestions_section:
                print(f"💡 Suggestions: {len(suggestions_section['suggestions'])} found")
                for i, suggestion in enumerate(suggestions_section['suggestions'][:3]):  # Show first 3
                    print(f"   {i+1}. {suggestion.get('title', 'No title')}")
            
            # Print references if available
            references_section = next((s for s in formatted_response.get('sections', []) if s.get('type') == 'references'), None)
            if references_section and 'references' in references_section:
                print(f"📚 References: {len(references_section['references'])} found")
                for i, ref in enumerate(references_section['references'][:3]):  # Show first 3
                    print(f"   {i+1}. Score: {ref.get('relevance_score', 'N/A'):.3f}")
            
            print("="*80)
            print("📤 SENDING RESPONSE TO FRONTEND")
            print("="*80 + "\n")
            
            # Final print before sending
            print(f"🎯 FINAL RESPONSE : {final_response}")
            
            return final_response
        else:
            # Handle graph processing errors
            error_msg = result.get("error", "Unknown error in LangGraph processing")
            logger.error(f"LangGraph processing failed: {error_msg}")
            
            error_response = QueryResponse(
                formatted_response={"error": error_msg},
                metadata={
                    "system_mode": "langgraph_error",
                    "error": error_msg,
                    "processing_time": time.time() - start_time
                },
                status="error"
            )
            
            # Print error response
            print("\n" + "="*80)
            print("❌ BACKEND ERROR RESPONSE")
            print("="*80)
            print(f"📝 Query: {request.query}")
            print(f"👤 Persona: {request.persona}")
            print(f"⏱️  Processing Time: {error_response.metadata['processing_time']:.3f}s")
            print(f"🎯 Status: {error_response.status}")
            print(f"🚨 Error: {error_msg}")
            print("="*80)
            print("📤 SENDING ERROR RESPONSE TO FRONTEND")
            print("="*80 + "\n")
            
            # Final print before sending
            print(f"🎯 FINAL RESPONSE : {error_response}")
            
            return error_response
            
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        exception_response = QueryResponse(
            formatted_response={"error": f"Query processing failed: {str(e)}"},
            metadata={
                "system_mode": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            },
            status="error"
        )
        
        # Print exception response
        print("\n" + "="*80)
        print("💥 BACKEND EXCEPTION RESPONSE")
        print("="*80)
        print(f"📝 Query: {request.query}")
        print(f"👤 Persona: {request.persona}")
        print(f"⏱️  Processing Time: {exception_response.metadata['processing_time']:.3f}s")
        print(f"🎯 Status: {exception_response.status}")
        print(f"💥 Exception: {str(e)}")
        print("="*80)
        print("📤 SENDING EXCEPTION RESPONSE TO FRONTEND")
        print("="*80 + "\n")
            
            # Final print before sending
        print(f"🎯 FINAL RESPONSE : {exception_response}")
            
        return exception_response


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
        # print(f"=== FILE UPLOAD DEBUG ===")
        # print(f"Filename: {file.filename}")
        # print(f"Content Type: {file.content_type}")
        # print(f"File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
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
            # print(f"Structured file saved to: {file_path}")
            text = extract_text_from_file(file_path, file_type)
            # print(f"Extracted text length: {len(text)}")
            # print("Skipping embedding for structured file.")
            # --- Update CSV/column cache using DbNode ---
            try:
                if graph_system is not None:
                    graph_system.db_node.update_column_names_cache(safe_filename)
                    # print("[UPLOAD] CSV columns cached successfully.")
                else:
                    from app.graph.nodes.db_node import DbNode
                    db_node = DbNode()
                    db_node.update_column_names_cache(safe_filename)
                    # print("[UPLOAD] CSV columns cached successfully (local instance).")
            except Exception as meta_exc:
                # print(f"[UPLOAD] Error updating CSV column cache: {meta_exc}")
                pass
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
            # print(f"Unstructured file saved to: {file_path}")
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
            # print(f"Extracted text length: {len(text)}")
            chunks = chunk_text(text)
            # print(f"Number of chunks created: {len(chunks)}")
            embeddings = generate_embeddings(chunks)
            # print(f"Generated {len(embeddings)} embeddings")
            # print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
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
