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
    """Process a query through the dynamic agent system"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.query[:50]}... with persona: {request.persona}")
        
        # Use actual graph system if available
        if graph_system is not None:
            # Prepare context for the graph system
            context = request.context or {}
            
            # Add persona to context
            context["preferred_persona"] = request.persona
            
            # Add file paths for sample data
            context["data_file"] = "./data/msft_2024.csv"
            context["contract_file"] = "./data/sample_contract.pdf"
            
            # Process query through the actual graph system
            result = await graph_system.process_query(request.query, context)
            
            # Extract response and format appropriately
            formatted_response = result.get("formatted_response", {})
            metadata = result.get("metadata", {})
            
            # Ensure the response has the expected structure
            if "response" not in formatted_response:
                formatted_response["response"] = result.get("processed_data", {}).get("summary", "No response generated")
            
            if "suggestions" not in formatted_response:
                formatted_response["suggestions"] = result.get("suggestions", {}).get("recommendations", [])
            
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                formatted_response=formatted_response,
                metadata={
                    "persona_selected": metadata.get("persona_selected", request.persona),
                    "route_selected": metadata.get("route_selected", "unknown"),
                    "route_confidence": metadata.get("route_confidence", 0.0),
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "system_mode": "actual_graph"
                },
                status="success"
            )
            
            logger.info(f"Query processed via graph system in {processing_time:.2f}s")
            return response
            
        else:
            # Fall back to mock responses if graph system is not available
            logger.warning("Graph system not available, using mock responses")
            
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Mock response based on persona
            persona_responses = {
                "financial_analyst": {
                    "response": f"[MOCK] Financial analysis of your query: '{request.query}'. Based on the current market data, I recommend...",
                    "suggestions": [
                        "Review quarterly earnings reports",
                        "Analyze competitor performance",
                        "Consider market volatility factors"
                    ]
                },
                "legal_advisor": {
                    "response": f"[MOCK] Legal analysis of your query: '{request.query}'. From a legal perspective, the key considerations are...",
                    "suggestions": [
                        "Review relevant regulations",
                        "Consult with specialized counsel",
                        "Document compliance requirements"
                    ]
                },
                "data_scientist": {
                    "response": f"[MOCK] Data analysis of your query: '{request.query}'. The statistical analysis reveals...",
                    "suggestions": [
                        "Gather additional data points",
                        "Validate data quality",
                        "Consider machine learning models"
                    ]
                },
                "business_consultant": {
                    "response": f"[MOCK] Business analysis of your query: '{request.query}'. Strategic recommendations include...",
                    "suggestions": [
                        "Conduct market research",
                        "Analyze business metrics",
                        "Develop implementation plan"
                    ]
                }
            }
            
            formatted_response = persona_responses.get(request.persona, {
                "response": f"[MOCK] General response to: '{request.query}'",
                "suggestions": ["Consider consulting with a specialist"]
            })
            
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                formatted_response=formatted_response,
                metadata={
                    "persona_selected": request.persona,
                    "route_selected": "mock_processor",
                    "route_confidence": 0.95,
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "system_mode": "mock_fallback"
                },
                status="success"
            )
            
            logger.info(f"Query processed via mock system in {processing_time:.2f}s")
            return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing"""
    try:
        logger.info(f"File upload: {file.filename} ({file.content_type})")
        
        # Read file content
        content = await file.read()
        
        # For now, just return file info
        # In a real implementation, this would process the file
        
        return {
            "status": "success",
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "message": "File uploaded successfully (mock response)"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
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
