"""
Document Processing Node
Handles PDF extraction, document analysis, and content processing with RAG capabilities
"""
from typing import Dict, Any, List, Optional
import logging
import os
from pathlib import Path
import openai

logger = logging.getLogger(__name__)

class DocNode:
    """Processes documents for analysis and information extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def process_document(self, file_path: str, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process document and extract relevant information using RAG
        
        Args:
            file_path: Path to the document file
            query: User query for context
            context: Additional context information
            
        Returns:
            Processed document results with RAG answer
        """
        logger.info(f"Processing document with RAG: {file_path}")
        
        try:
            # If no specific file path, find the latest uploaded document
            if not file_path:
                file_path = self._find_latest_document()
                if not file_path:
                    return {"error": "No documents found. Please upload a PDF or text document first."}
            
            # Validate file exists and is supported
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return {"error": f"Unsupported file format: {file_ext}"}
            
            # Use RAG processing for unstructured queries
            return self._process_with_rag(file_path, query, context)
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": f"Document processing failed: {str(e)}"}
    
    def _find_latest_document(self) -> Optional[str]:
        """Find the document in data/uploads/unstructured"""
        unstructured_dir = './data/uploads/unstructured/'
        if not os.path.exists(unstructured_dir):
            print("Unstructured directory not found")
            return None
        
        files = [f for f in os.listdir(unstructured_dir) if f.endswith(('.pdf', '.txt'))]
        if not files:
            print("No files found in unstructured directory")
            return None
        
        # Since there's only one document, just pick the first one
        file_path = os.path.join(unstructured_dir, files[0])
        print(f"Found document: {file_path}")
        return file_path
    
    def _process_with_rag(self, file_path: str, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process document using RAG (Retrieval-Augmented Generation)"""
        try:
            # 1. Create embeddings for the user query
            from app.graph.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            query_embedding = embedding_service.embed_texts([query])[0]
            
            if not query_embedding:
                return {"error": "Failed to create query embedding."}
            
            # 2. Search Pinecone for relevant chunks
            from app.graph.services.pinecone_service import PineconeService
            pinecone_service = PineconeService()
            
            # Connect to Pinecone index
            connect_result = pinecone_service.connect_to_index()
            if not connect_result or connect_result.get("error"):
                return {"error": f"Failed to connect to Pinecone: {connect_result.get('error', 'Unknown error')}"}
            
            # Search for relevant chunks
            search_result = pinecone_service.semantic_search(
                query_embedding=query_embedding,
                top_k=5  # Get top 5 most relevant chunks
            )
            
            if not search_result.get("success"):
                return {"error": f"Pinecone search failed: {search_result.get('error', 'Unknown error')}"}
            
            relevant_chunks = search_result.get("results", [])
            if not relevant_chunks:
                return {"error": "No relevant content found in the document for your query."}
            
            # 3. Use LLM to generate an answer based on retrieved chunks
            answer = self._generate_rag_answer(query, relevant_chunks)
            
            # 4. Format the response
            return {
                "success": True,
                "type": "document_analysis",
                "answer": answer,
                "query": query,
                "document_used": os.path.basename(file_path),
                "relevant_chunks_found": len(relevant_chunks),
                "references": self._format_references(relevant_chunks),
                "search_metadata": {
                    "total_chunks_searched": search_result.get("query_info", {}).get("top_k", 0),
                    "embedding_model": "text-embedding-ada-002"
                }
            }
            
        except Exception as e:
            logger.error(f"RAG processing failed: {str(e)}")
            return {"error": f"RAG processing failed: {str(e)}"}
    
    def _generate_rag_answer(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM based on retrieved chunks"""
        try:
            # Prepare context from retrieved chunks
            context_text = ""
            for i, chunk in enumerate(relevant_chunks):
                chunk_text = chunk.get("text", "")
                relevance_score = chunk.get("relevance_score", 0.0)
                context_text += f"\n[Chunk {i+1}] (Relevance: {relevance_score:.3f})\n{chunk_text}\n"
            
            # Generate answer using LLM
            rag_prompt = f"""
            Based on the following document excerpts, answer the user's question: "{query}"
            
            Document Context:
            {context_text}
            
            Instructions:
            - Provide a comprehensive answer based on the document content
            - Reference specific chunks when relevant (e.g., "According to Chunk 1...")
            - If the document doesn't contain enough information to answer fully, say so
            - Be specific and cite the relevant parts of the document
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on document content. Always cite your sources from the provided chunks."},
                    {"role": "user", "content": rag_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content or "No answer generated."
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return f"Failed to generate answer: {str(e)}"
    
    def _format_references(self, relevant_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format references for the response"""
        references = []
        for i, chunk in enumerate(relevant_chunks):
            chunk_text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            relevance_score = chunk.get("relevance_score", 0.0)
            
            references.append({
                "chunk_id": chunk_id,
                "relevance_score": relevance_score,
                "text_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            })
        
        return references
    
    def _extract_text(self, file_path: str, file_ext: str) -> str:
        """Extract text from different file types"""
        try:
            if file_ext == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            elif file_ext == '.docx':
                return self._extract_docx_text(file_path)
            else:
                return ""
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return ""
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using OCR if necessary"""
        # Placeholder for PDF text extraction
        # In a real implementation, you would use libraries like:
        # - PyPDF2 for text-based PDFs
        # - pytesseract + pdf2image for OCR
        return f"[PDF TEXT EXTRACTION PLACEHOLDER for {file_path}]"
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        # Placeholder for DOCX text extraction
        # In a real implementation, you would use python-docx
        return f"[DOCX TEXT EXTRACTION PLACEHOLDER for {file_path}]"
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
        
        return chunks
    
    def _find_relevant_chunks(self, chunks: List[str], query: str) -> List[Dict[str, Any]]:
        """Find chunks most relevant to the query"""
        query_words = set(query.lower().split())
        relevant_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            relevance_score = overlap / len(query_words) if query_words else 0
            
            if relevance_score > 0.1:  # Threshold for relevance
                relevant_chunks.append({
                    "chunk_id": i,
                    "content": chunk,
                    "relevance_score": relevance_score
                })
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return relevant_chunks[:5]  # Return top 5 relevant chunks
    
    def _generate_summary(self, relevant_chunks: List[Dict[str, Any]], query: str) -> str:
        """Generate a summary based on relevant chunks and query"""
        if not relevant_chunks:
            return "No relevant content found in the document."
        
        # Placeholder for summary generation
        # In a real implementation, you would use an LLM for summarization
        chunk_contents = [chunk["content"][:200] + "..." for chunk in relevant_chunks]
        
        return f"Based on the query '{query}', found {len(relevant_chunks)} relevant sections. " + \
               f"Key content includes: {'; '.join(chunk_contents)}"
    
    def analyze_contract(self, file_path: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Specialized contract analysis"""
        logger.info(f"Analyzing contract: {file_path}")
        
        # Extract contract text
        file_ext = Path(file_path).suffix.lower()
        text = self._extract_text(file_path, file_ext)
        
        if not text:
            return {"error": "Could not extract contract text"}
        
        # Contract-specific analysis
        analysis_results = {
            "parties": self._extract_parties(text),
            "key_terms": self._extract_key_terms(text),
            "obligations": self._extract_obligations(text),
            "dates": self._extract_dates(text),
            "financial_terms": self._extract_financial_terms(text)
        }
        
        return {
            "success": True,
            "analysis_type": analysis_type,
            "results": analysis_results
        }
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract contract parties"""
        # Placeholder for party extraction logic
        return ["Party A", "Party B"]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key contract terms"""
        # Placeholder for key terms extraction
        return ["Term 1", "Term 2", "Term 3"]
    
    def _extract_obligations(self, text: str) -> List[str]:
        """Extract obligations and responsibilities"""
        # Placeholder for obligations extraction
        return ["Obligation 1", "Obligation 2"]
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract important dates"""
        # Placeholder for date extraction
        return ["2024-01-01", "2024-12-31"]
    
    def _extract_financial_terms(self, text: str) -> List[str]:
        """Extract financial terms and amounts"""
        # Placeholder for financial terms extraction
        return ["$100,000", "Monthly payment"] 